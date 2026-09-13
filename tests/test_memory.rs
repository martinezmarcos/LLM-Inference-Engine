use mini_llm::memory::{DoubleBuffer, ScratchArena};

#[test]
fn test_scratch_arena_allocation_and_reset() {
    let mut arena = ScratchArena::new(100);
    assert_eq!(arena.capacity(), 100);
    assert_eq!(arena.current_usage(), 0);

    let t1 = arena.alloc(&[10, 5]).unwrap();
    assert_eq!(t1.dims(), &[10, 5]);
    assert_eq!(t1.offset(), 0);
    assert_eq!(arena.current_usage(), 50);

    let t2 = arena.alloc(&[5, 5]).unwrap();
    assert_eq!(t2.dims(), &[5, 5]);
    assert_eq!(t2.offset(), 50);
    assert_eq!(arena.current_usage(), 75);

    // Exceed capacity
    let err = arena.alloc(&[30]);
    assert!(err.is_err());

    // Reset arena
    arena.reset();
    assert_eq!(arena.current_usage(), 0);

    // Reuse without new allocations
    let t3 = arena.alloc(&[50]).unwrap();
    assert_eq!(t3.offset(), 0);
    assert_eq!(arena.current_usage(), 50);
}

#[test]
fn test_double_buffering() {
    let mut db = DoubleBuffer::new(&[2, 4]);

    {
        let (_in_buf, out_buf) = db.step();
        out_buf.set(&[0, 0], 123.0).unwrap();
    }
    assert_eq!(db.current().get(&[0, 0]).unwrap(), 123.0);

    {
        let (_in_buf, out_buf) = db.step();
        out_buf.set(&[0, 0], 456.0).unwrap();
    }
    assert_eq!(db.current().get(&[0, 0]).unwrap(), 456.0);
}
