# Documentación Técnica de mini-llm

Este documento detalla las decisiones de arquitectura, fundamentos de ingeniería de sistemas y principios matemáticos detrás del runtime de inferencia **mini-llm**, implementado desde cero en Rust. Este documento decidi llevarlo a cabo con un workflow de IA el cual iba documentando lo que hacia cada cierto tiempo.

---

## 1. Representación de un Tensor en Memoria

### Memoria Plana y Direccionamiento Multidimensional
En hardware moderno, la memoria física direccionable (DRAM y jerarquías de caché L1/L2/L3) es un espacio lineal unidimensional contiguo de bytes. Los tensores, sin embargo, representan arreglos matemáticos multidimensionales de rango $N$ con dimensiones $(d_0, d_1, \dots, d_{N-1})$.

Para mapear una coordenada lógica $(i_0, i_1, \dots, i_{N-1})$ a un índice escalar plano en memoria, utilizamos un vector de **strides** (pasos) $(s_0, s_1, \dots, s_{N-1})$:

$$\text{offset} = \text{base\_offset} + \sum_{k=0}^{N-1} i_k \cdot s_k$$

### Layout Row-Major (C-Contiguous)
mini-llm utiliza por defecto una distribución **Row-Major** (orden C):
* La dimensión más interna ($d_{N-1}$) tiene stride $s_{N-1} = 1$. Los elementos adyacentes en la última dimensión están físicamente contiguos en memoria (separados exactamente por `sizeof(dtype)` bytes).
* Para cualquier dimensión $k < N-1$, el stride se calcula mediante el producto acumulado de las dimensiones subsiguientes:
  $$s_k = \prod_{j=k+1}^{N-1} d_j$$

### Caché Lines y Prefetching Espacial
Una línea de caché en la arquitectura x86-64 moderna típica es de **64 bytes**. Con elementos `f32` (4 bytes cada uno), una sola línea de caché carga exactamente **16 valores contiguos**.
* **Acceso Secuencial**: Iterar sobre el eje contiguo ($stride = 1$) garantiza que 1 de cada 16 accesos provoca un *cache miss*, mientras que los 15 restantes resultan en *L1 cache hits* inmediatos (~1 ns de latencia). Además, el *hardware stream prefetcher* de la CPU detecta el patrón y precarga las líneas siguientes antes de que las instrucciones las soliciten.
* **Acceso Discontiguo (Strided)**: Si un algoritmo itera a lo largo de una dimensión con un stride grande (por ejemplo, recorriendo una columna de una matriz grande sin transponer), cada acceso carga 64 bytes pero utiliza solo 4 bytes, desperdiciando el 93.75% del ancho de banda del bus de memoria y expulsando líneas útiles de la caché L1.

### Vistas Zero-Copy vs. Contigüidad
El desacoplamiento entre `Shape`, `Strides`, `Offset` y el búfer subyacente permite operaciones de costo $O(1)$ sin copia:
1. **Transpose (2D / Permute)**: Intercambiar dos dimensiones equivale a intercambiar sus respectivos strides. Los datos no se mueven.
2. **Slicing**: Tomar un subrango a lo largo de una dimensión ajusta el `offset` base y las dimensiones del tensor sin reubicar la memoria subyacente.
3. **Contiguous**: Cuando una operación de bajo nivel (como vectorización SIMD o llamadas a BLAS) requiere memoria estrictamente secuencial, se invoca `.contiguous()`, que reordena y compacta los datos en un nuevo búfer contiguo.

---

## 2. Multiplicación de Matrices (`matmul`)

### Complejidad y Aritmética
La multiplicación de dos matrices $A \in \mathbb{R}^{M \times K}$ y $B \in \mathbb{R}^{K \times N}$ genera una matriz $C \in \mathbb{R}^{M \times N}$ según:

$$C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} \cdot B_{k, j}$$

* Complejidad temporal: $O(M \cdot N \cdot K)$ operaciones de punto flotante (FLOPs: $2MNK$ operaciones contando multiplicación y suma).
* Complejidad espacial: $O(M \cdot N)$ para el resultado.

### El Desafío del Layout en Matmul Ingenuo
En una implementación ingenua con tres bucles anidados $(i, j, k)$:
```text
for i in 0..M:
    for j in 0..N:
        for k in 0..K:
            C[i, j] += A[i, k] * B[k, j]
```
* Para la matriz $A$, el acceso $A[i, k]$ varía en $k$, que es la dimensión interna contigua (buena localidad espacial).
* Para la matriz $B$, el acceso $B[k, j]$ varía en $k$, saltando filas completas de longitud $N$ en cada paso del bucle más interno. El paso en memoria es de $N \cdot \text{sizeof}(f32)$ bytes. Si $N$ es grande, cada acceso a $B$ provoca un fallo de caché (*cache miss*).

### Optimizaciones Clave
1. **Reordenamiento de Bucles ($i, k, j$)**:
   Al permutar el orden de los bucles a $(i, k, j)$, el bucle más interno varía $j$:
   ```text
   for i in 0..M:
       for k in 0..K:
           let a_val = A[i, k]; // Constante en el bucle j
           for j in 0..N:
               C[i, j] += a_val * B[k, j];
   ```
   Tanto $B[k, j]$ como $C[i, j]$ son accedidos de forma puramente secuencial con $stride = 1$. Esto transforma drásticamente el rendimiento multiplicando la tasa de aciertos en caché.
2. **Tiling / Blocking (Partición en Bloques para Jerarquía de Memoria)**:
   Dividir las matrices en subbloques de tamaño $B_M \times B_K \times B_N$ que quepan simultáneamente en la memoria caché L1/L2 (e.g. 32 KB para L1D). De este modo, los datos de un bloque de $B$ se reutilizan múltiples veces antes de ser desalojados de la caché.
3. **Vectorización SIMD**:
   Utilizar instrucciones vectoriales (AVX2 / AVX-512 / NEON) para calcular 8 o 16 productos acumulados en paralelo por ciclo de reloj utilizando FMA (*Fused Multiply-Add*).
4. **Paralelismo de Datos**:
   Distribuir las filas independientes de $M$ entre los núcleos disponibles de la CPU.

---

## 3. Mecanismo de Attention (Multi-Head & Causal)

### Formulación Matemática
Para cada cabeza de atención con dimensión de cabeza $d_k$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Donde:
* $Q \in \mathbb{R}^{S_q \times d_k}$ (Queries: preguntas del token actual).
* $K \in \mathbb{R}^{S_k \times d_k}$ (Keys: etiquetas del contexto pasado y presente).
* $V \in \mathbb{R}^{S_k \times d_v}$ (Values: contenido de información semántica).
* $\frac{1}{\sqrt{d_k}}$: Factor de escala que previene que los productos punto crezcan desproporcionadamente con $d_k$, lo cual empujaría la función softmax a regiones de gradientes saturados con varianza descontrolada.
* $M$: Máscara causal triangular inferior, donde $M_{i, j} = -\infty$ si $j > i$ (para evitar que tokens pasados atiendan al futuro durante el entrenamiento y la generación autorregresiva).

### Softmax Numéricamente Estable
El cálculo directo de $e^{x_i}$ provoca rápidamente *overflow* numérico (e.g., $e^{89} \approx \infty$ en IEEE 754 `f32`). Para garantizar estabilidad matemática absoluta:
$$m = \max_j (x_j)$$
$$P_i = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}$$
Dado que $x_i - m \le 0$, los exponentes están estrictamente en el intervalo $(-\infty, 0]$, garantizando $e^{x_i - m} \in (0, 1]$, eliminando el riesgo de desbordamiento por infinito.

---

## 4. RoPE (Rotary Position Embeddings)

### Fundamento
A diferencia de los embeddings posicionales absolutos que se suman linealmente a las entradas ($x + p$), RoPE incorpora la posición relativa multiplicando los vectores $Q$ y $K$ por matrices de rotación ortogonales en bloques de dimensión 2:

Para un vector $x = (x_0, x_1) \in \mathbb{R}^2$ en la posición $m$ con frecuencia angular $\theta$:
$$R_{\Theta, m} x = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix} = \begin{pmatrix} x_0 \cos(m\theta) - x_1 \sin(m\theta) \\ x_0 \sin(m\theta) + x_1 \cos(m\theta) \end{pmatrix}$$

Donde $\theta_i = \text{base}^{-2i/d}$, típicamente con $\text{base} = 10000.0$ (o $500000.0$ en modelos recientes).

### Propiedad Fundamental
El producto escalar entre un Query en la posición $m$ y una Key en la posición $n$ satisface:
$$\langle R_{\Theta, m} Q, R_{\Theta, n} K \rangle = Q^T R_{\Theta, m}^T R_{\Theta, n} K = Q^T R_{\Theta, n - m} K$$
La atención entre dos tokens depende únicamente de su **distancia relativa** $(m - n)$, no de sus posiciones absolutas en la secuencia.

---

## 5. KV Cache: Memoria y Complejidad en Inferencia

### El Problema de la Inferencia Autorregresiva
En generación autorregresiva token por token:
* En el paso $t$, se introduce únicamente el nuevo token $x_t$.
* Las representaciones de Keys y Values de los tokens previos $x_0, \dots, x_{t-1}$ son invariantes bajo decodificación causal.
* Sin KV Cache, recomputar $K$ y $V$ para toda la secuencia en cada paso requiere $O(T^2)$ cómputo total acumulado para una secuencia de longitud $T$.
* Con KV Cache, almacenamos las matrices $K$ y $V$ calculadas previamente. En el paso $t$, únicamente calculamos $Q_t, K_t, V_t$, anexamos $(K_t, V_t)$ al caché y evaluamos la atención contra el histórico en $O(T)$ por paso.

### Consumo de Memoria del KV Cache
Para un modelo con:
* $L$ capas (layers)
* $H_{kv}$ cabezas de KV (en Multi-Query o Grouped-Query Attention, $H_{kv} \le H_q$)
* $D_{head}$ dimensión por cabeza
* $S_{max}$ tokens máximos soportados en el contexto
* $B$ tamaño de lote (batch size, típicamente 1 para inferencia local)
* `sizeof(dtype)` (e.g. 4 bytes para `f32`, 2 bytes para `f16`)

$$\text{Memoria KV Cache} = 2 \times L \times B \times H_{kv} \times S_{max} \times D_{head} \times \text{sizeof(dtype)}$$
*(El factor 2 proviene de almacenar tanto Keys como Values)*.

---

## 6. Carga de Pesos y Formato GGUF

### Estructura de Archivo GGUF
GGUF (GPT-Generated Unified Format) es un formato binario estructurado en tres secciones:
1. **Header**: Magic number (`0x46554747` = `"GGUF"` en little-endian), versión del formato (v2 o v3), contador de tensores y contador de pares clave-valor de metadatos.
2. **Metadata Key-Value Store**: Tipos de datos serializados (strings, enteros, floats, arrays) que contienen hiperparámetros de la arquitectura (`general.architecture`, `llama.block_count`, `llama.embedding_length`, `tokenizer.ggml.tokens`, etc.).
3. **Tensor Info Index**: Nombre de cada tensor, número de dimensiones, dimensiones, tipo de cuantización (e.g. `GGML_TYPE_F32`, `GGML_TYPE_Q4_0`), y offset relativo al inicio del bloque de datos binarios.
4. **Binary Tensor Data**: Búfer continuo de pesos con alineación configurable (por defecto 32 bytes).

### Memory-Mapping (`mmap`) y Zero-Copy
En lugar de invocar `read()` para volcar gigabytes de pesos a través del kernel hacia búferes en espacio de usuario (lo que duplicaría la huella de memoria RAM e incurriría en overhead de copias de memoria):
* Se realiza una llamada a `mmap` del archivo en modo solo lectura.
* El sistema operativo mapea las páginas del archivo directamente en el espacio de direcciones virtuales del proceso.
* Las páginas se cargan desde el almacenamiento mediante *page faults* bajo demanda conforme el runtime accede a los pesos de cada capa.

---

## 7. Ciclo de Inferencia Autorregresiva

El flujo de ejecución autorregresivo desacopla la fase de **Prefill** de la fase de **Decode**:

```text
[Texto Prompt]
      │
      ▼ (BPE Tokenize)
[Token IDs: t_0, ..., t_p]
      │
      ├─► [FASE 1: PREFILL (Paralelo)]
      │   Procesa t_0..t_p simultáneamente en paralelo
      │   Llena el KV Cache hasta la posición p
      │   Calcula logits para t_p
      │
      ▼
[Logits del último token]
      │
      ▼ (Sampler: Temperature / Top-P / Greedy)
[Nuevo Token ID: t_{p+1}]
      │
      ├─► Emitir token a la salida
      │
      ▼
┌─► [FASE 2: DECODE (Secuencial, token a token)]
│   Posición: pos = pos + 1
│   Input: Únicamente t_{actual} (sec_len = 1)
│   Forward pass:
│     - Embedding lookup
│     - Layers Transformer con KV Cache append
│     - RMSNorm final + Proyección a Vocabulario
│   Sampler ──► t_{siguiente}
│   ¿t_{siguiente} == EOS o alcanzado max_tokens?
│     ├─► SÍ: Finalizar inferencia
│     └─► NO: Repetir bucle Decode
└────────┘
```

---

## 8. Gestión de Memoria y Perfil de Asignaciones

### El Problema de las Asignaciones en el Bucle Crítico
En C++ o Rust, invocar `Vec::new()` o asignar búferes dentro del bucle token por token (`Decode`) destruye el rendimiento:
* Cada asignación invoca el allocator general del sistema (`malloc` / `jemalloc`), introduciendo contención de bloqueos (locks) y fragmentación del heap.
* Se invalida la localidad de la caché L1 al obtener punteros en direcciones dispersas.

### Estrategia de Asignación en mini-llm
1. **Pre-reserva de Búferes de Activación**: Todos los tensores intermedios de una capa Transformer ($Q, K, V$, atención, proyecciones MLP) tienen dimensiones fijas ($1 \times \text{hidden\_dim}$). Se preasignan antes de iniciar el bucle de generación.
2. **Scratchpad / Arena Allocator**: Durante el paso hacia adelante (`forward`), los tensores temporales reutilizan espacio de un búfer de trabajo compartido reajustable mediante un simple puntero lineal que se reinicia a cero al finalizar la evaluación del token.
3. **Invariante de Allocations**: El bucle `while pos < max_tokens` ejecuta con **cero asignaciones dinámicas en el heap** (`0 allocations per token`).

---

## 9. Cuantización: Representación y Error Numérico

### Principio Matemático
La cuantización reduce el número de bits necesarios para almacenar cada peso, reduciendo drásticamente la saturación del bus de memoria:

#### Cuantización Uniforme por Bloques (ejemplo Q8_0 / Q4_0)
Para un bloque contiguo de $B$ pesos (típicamente $B = 32$ en formatos GGML):
1. Se localiza el valor absoluto máximo en el bloque:
   $$x_{max} = \max_{i} |x_i|$$
2. Se calcula el factor de escala en punto flotante `f16` o `f32`:
   $$d = \frac{x_{max}}{2^{b-1} - 1}$$
   *(donde $b=8$ para 8 bits con rango $[-127, 127]$, o $b=4$ para 4 bits con rango $[-8, 7]$)*.
3. Cada peso se cuantiza discretamente como:
   $$q_i = \text{round}\left(\frac{x_i}{d}\right)$$
4. Almacenamiento: Se guarda la escala $d$ compartida por el bloque y los enteros comprimidos $q_i$.

#### Cuantized Dot Product (`quantized_matmul`)
Al evaluar el producto punto entre activaciones en `f32` y pesos cuantizados en `i8`/`i4`, se evita descuantizar a memoria intermedia:
$$\sum_{i} x_i \cdot w_i \approx d \cdot \sum_{i} x_i \cdot q_i$$
Se acumulan los productos enteros o escalares directamente en registros de la CPU, reduciendo la transferencia de memoria en un factor de $4\times$ (para INT8) u $8\times$ (para INT4).

---

## 10. Jerarquía de Impacto de Optimizaciones

| Nivel | Optimización | Mecanismo Físico | Aceleración Típica |
| :--- | :--- | :--- | :--- |
| **1** | **KV Cache** | Evita recalcular atención histórica ($O(T^2) \to O(T)$) | $\mathbf{10\times - 100\times}$ en secuencias largas |
| **2** | **Reordenamiento de Bucles MatMul ($ikj$)** | Garantiza acceso contiguo a memoria (L1 cache hit rate $>98\%$) | $\mathbf{8\times - 15\times}$ vs. ingenuo $ijk$ |
| **3** | **Zero Allocations en Inferencia** | Elimina llamadas a `malloc` en el bucle caliente | $\mathbf{2\times - 4\times}$ latencia y variabilidad |
| **4** | **Vectorización SIMD (AVX2 / FMA)** | Ejecuta 8 operaciones `f32` simultáneas por ciclo por núcleo | $\mathbf{3\times - 6\times}$ en kernels densos |
| **5** | **Cuantización de Pesos (INT8 / INT4)** | Reduce el cuello de botella del ancho de banda de memoria DRAM | $\mathbf{2\times - 3\times}$ tokens/segundo en CPU |
| **6** | **Paralelismo Multihilo (Rayon)** | Distribuye proyección de cabezas y filas MLP entre núcleos | $\mathbf{2\times - 6\times}$ según el número de núcleos físicos |
