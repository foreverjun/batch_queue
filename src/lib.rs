extern crate alloc;

use alloc::boxed::Box;
use core::iter::IntoIterator;
use core::ptr;
use core::sync::atomic::{AtomicPtr as StdAtomicPtr, Ordering};
use haphazard::raw::Pointer;
use haphazard::{AtomicPtr, Domain, HazardPointer};
use std::mem::MaybeUninit;
use std::ptr::null_mut;

const ANN_TAG: usize = 1;

#[inline(always)]
fn tag_ann<T>(ann_ptr: *mut Ann<T>) -> *mut Node<T> {
    (ann_ptr as usize | ANN_TAG) as *mut Node<T>
}

#[inline(always)]
fn untag<T>(ptr: *mut Node<T>) -> usize {
    ptr as usize & !ANN_TAG
}

#[inline(always)]
fn is_ann<T>(ptr: *mut Node<T>) -> bool {
    (ptr as usize & ANN_TAG) == ANN_TAG
}

#[inline(always)]
fn get_ann_ptr<T>(ptr: *mut Node<T>) -> *mut Ann<T> {
    untag(ptr) as *mut Ann<T>
}

#[inline(always)]
fn get_node_ptr<T>(ptr: *mut Node<T>) -> *mut Node<T> {
    ptr
}

struct Node<T> {
    item: MaybeUninit<T>,
    next: AtomicPtr<Node<T>>,
}

type PtrOrAnn<T> = StdAtomicPtr<T>;

struct InternalBatchRequest<T> {
    first_enq: *mut Node<T>,
    last_enq: *mut Node<T>,
}

struct Ann<T> {
    batch_req: InternalBatchRequest<T>,
    old_head_node: *mut Node<T>,
    old_tail_node: AtomicPtr<Node<T>>,
}
unsafe impl<T: Send> Send for Ann<T> {}

pub struct BQueue<T>
where
    T: Send + Sync,
{
    head: PtrOrAnn<Node<T>>,
    tail: AtomicPtr<Node<T>>,
}

trait RetireHelpers {
    unsafe fn retire_node<T: Send>(&self, node: *mut Node<T>);
    unsafe fn retire_ann<T: Send>(&self, ann: *mut Ann<T>);
}

impl<F: 'static> RetireHelpers for Domain<F> {
    unsafe fn retire_node<T: Send>(&self, node: *mut Node<T>) {
        unsafe {
            self.retire_ptr::<Node<T>, Box<Node<T>>>(node);
        }
    }
    unsafe fn retire_ann<T: Send>(&self, ann: *mut Ann<T>) {
        unsafe {
            self.retire_ptr::<Ann<T>, Box<Ann<T>>>(ann);
        }
    }
}

impl<T> Node<T> {
    fn new(item: T) -> Self {
        Self {
            item: MaybeUninit::new(item),
            next: unsafe { AtomicPtr::new(core::ptr::null_mut()) },
        }
    }

    fn empty() -> Self {
        Self {
            item: MaybeUninit::uninit(),
            next: unsafe { AtomicPtr::new(core::ptr::null_mut()) },
        }
    }
}

impl<T> BQueue<T>
where
    T: Send + Sync,
{
    pub fn new() -> Self {
        let dummy_node_ptr = Box::new(Node::empty()).into_raw();
        BQueue {
            head: StdAtomicPtr::new(dummy_node_ptr),
            tail: unsafe { AtomicPtr::new(dummy_node_ptr) },
        }
    }

    fn help_ann_and_get_head<'hp>(&self, hp: &'hp mut HazardPointer) -> *mut Node<T>
    where
        T: 'hp,
    {
        loop {
            let head_raw = self.head.load(Ordering::Acquire);

            if !is_ann(head_raw) {
                let node_ptr = get_node_ptr(head_raw);
                hp.protect_raw(node_ptr);
                if node_ptr.is_null() {
                    continue;
                }
                let current_head_raw = self.head.load(Ordering::Acquire);
                if current_head_raw != head_raw {
                    hp.reset_protection();
                    continue;
                }
                return node_ptr;
            } else {
                let ann_ptr = get_ann_ptr(head_raw);
                hp.protect_raw(ann_ptr as *mut Ann<T>);
                let current_head_raw = self.head.load(Ordering::Acquire);
                if current_head_raw != head_raw {
                    hp.reset_protection();
                    continue;
                }
                self.execute_ann(ann_ptr);
                hp.reset_protection();
            }
        }
    }

    pub fn enqueue(&self, item: T) {
        let new_node_ptr = Box::into_raw(Box::new(Node::new(item)));
        let mut hp_tail = HazardPointer::new();
        let mut hp_ann = HazardPointer::new();

        loop {
            let tail_node = self.tail.safe_load(&mut hp_tail).unwrap();
            let tail_node_ptr = tail_node as *const _ as *mut _;
            let std_next_atomic_ptr = unsafe { tail_node.next.as_std() };
            match std_next_atomic_ptr.compare_exchange(
                ptr::null_mut(),
                new_node_ptr,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let _ = unsafe { self.tail.compare_exchange_ptr(tail_node_ptr, new_node_ptr) };
                    return;
                }
                Err(actual_next_ptr) => {
                    let head_raw = self.head.load(Ordering::Acquire);
                    if is_ann(head_raw) {
                        let ann_ptr = get_ann_ptr(head_raw);
                        hp_ann.protect_raw(ann_ptr);
                        let current_head_raw = self.head.load(Ordering::Acquire);
                        if current_head_raw == head_raw {
                            self.execute_ann(ann_ptr);
                        }
                        hp_ann.reset_protection();
                    } else {
                        if !actual_next_ptr.is_null() {
                            let _ = unsafe {
                                self.tail
                                    .compare_exchange_ptr(tail_node_ptr, actual_next_ptr)
                            };
                        }
                    }
                }
            }
        }
    }

    pub fn dequeue(&self) -> Option<T> {
        let mut hp_head = HazardPointer::new();
        let mut hp_next = HazardPointer::new();

        loop {
            let head_node_ptr = self.help_ann_and_get_head(&mut hp_head);
            let head_node_ref = unsafe { &*head_node_ptr };

            let next_node_ptr = head_node_ref.next.load_ptr();
            if next_node_ptr.is_null() {
                return None;
            }
            let next_node_opt = head_node_ref.next.safe_load(&mut hp_next).unwrap();
            if let Ok(_) = self.head.compare_exchange(
                head_node_ptr,
                next_node_ptr,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                unsafe { Domain::global().retire_node(head_node_ptr) };
                return Some(unsafe {
                    std::ptr::read(next_node_opt.item.assume_init_ref() as *const _)
                });
            }
            hp_head.reset_protection();
            hp_next.reset_protection();
        }
    }

    pub fn deq_batch(&self, num_deqs_requested: usize) -> DeqBatchIterator<T> {
        let mut hp_head = HazardPointer::new();
        let mut hp1 = HazardPointer::new();
        let mut hp2 = HazardPointer::new();

        let mut successful_deqs;
        let mut old_head_ptr: *mut Node<T>;
        let mut last_deq_item_ptr: *const MaybeUninit<T>;

        loop {
            successful_deqs = 0;
            old_head_ptr = self.help_ann_and_get_head(&mut hp_head);

            let mut new_head_node = unsafe { old_head_ptr.as_ref().unwrap() };
            while successful_deqs < num_deqs_requested {
                let head_next_node = new_head_node.next.safe_load(&mut hp1);

                if let Some(head_next) = head_next_node {
                    successful_deqs += 1;
                    new_head_node = head_next;

                    if successful_deqs >= num_deqs_requested {
                        break;
                    }

                    let head_next_node_2 = new_head_node.next.safe_load(&mut hp2);
                    if let Some(head_next_2) = head_next_node_2 {
                        successful_deqs += 1;
                        new_head_node = head_next_2;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            if successful_deqs == 0 {
                return DeqBatchIterator::new(0, None, null_mut());
            }

            last_deq_item_ptr = &new_head_node.item;
            if self
                .head
                .compare_exchange(
                    old_head_ptr,
                    new_head_node as *const _ as *mut _,
                    Ordering::Release,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
            hp_head.reset_protection();
            hp1.reset_protection();
            hp2.reset_protection();
        }
        let last_item = unsafe { Some(std::ptr::read(last_deq_item_ptr).assume_init()) };

        DeqBatchIterator::new(successful_deqs, last_item, old_head_ptr)
    }

    fn execute_enqs_batch(&self, batch_req: InternalBatchRequest<T>) {
        let ann_ptr = Box::new(Ann {
            batch_req: batch_req,
            old_head_node: null_mut(),
            old_tail_node: unsafe { AtomicPtr::new(null_mut()) },
        })
        .into_raw();
        let tagged_ann_ptr = tag_ann(ann_ptr);
        let mut hp_head = HazardPointer::new();
        loop {
            let old_head_ptr = self.help_ann_and_get_head(&mut hp_head);
            unsafe { (*ann_ptr).old_head_node = old_head_ptr };
            if self
                .head
                .compare_exchange(
                    old_head_ptr,
                    tagged_ann_ptr,
                    Ordering::Release,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }

        hp_head.reset_protection();
        self.execute_ann(ann_ptr);
        unsafe {
            Domain::global().retire_ann(ann_ptr);
        }
    }

    fn execute_ann(&self, ann_ptr: *mut Ann<T>) {
        let ann = unsafe { &*ann_ptr };
        let mut hp_tail = HazardPointer::new();
        let mut ann_old_tail: *mut Node<T> = std::ptr::null_mut();

        loop {
            let tail = unsafe { self.tail.as_std().load(Ordering::Acquire) };
            let old_tail = ann.old_tail_node.safe_load(&mut hp_tail);
            if old_tail.is_some() {
                ann_old_tail = old_tail.unwrap() as *const _ as *mut _;
                if tag_ann(ann_ptr) != self.head.load(Ordering::Acquire) {
                    return;
                }
                break;
            }
            let tail_node = unsafe { hp_tail.try_protect(tail, self.tail.as_std()) };
            if tail_node.is_err() || tail_node.unwrap().is_none() {
                if tag_ann(ann_ptr) != self.head.load(Ordering::Acquire) {
                    return;
                }
                continue;
            }
            let tail_node = tail_node.unwrap().unwrap();
            unsafe {
                let _ = tail_node
                    .next
                    .compare_exchange_ptr(null_mut(), ann.batch_req.first_enq);
            }
            if tail_node.next.load_ptr() == ann.batch_req.first_enq {
                ann_old_tail = tail_node as *const _ as *mut _;
                unsafe { ann.old_tail_node.store_ptr(tail_node as *const _ as *mut _) };
                break;
            } else {
                let _ = unsafe {
                    self.tail
                        .compare_exchange_ptr(ann_old_tail, tail_node.next.load_ptr())
                };
            }
            hp_tail.reset_protection();
        }
        let _ = unsafe {
            self.tail
                .compare_exchange_ptr(ann_old_tail, ann.batch_req.last_enq)
        };
        hp_tail.reset_protection();

        let target_head_node_ptr = ann.old_head_node;
        let _ = self.head.compare_exchange(
            tag_ann(ann_ptr),
            target_head_node_ptr,
            Ordering::Release,
            Ordering::Relaxed,
        );
    }

    pub fn enqueue_batch<I>(&self, items: I)
    where
        I: IntoIterator<Item = T>,
    {
        let mut items_iter = items.into_iter();
        let enqs_head: *mut Node<T>;
        let mut enqs_tail: *mut Node<T>;

        if let Some(first_item) = items_iter.next() {
            let first_node = Box::new(Node::new(first_item)).into_raw();
            enqs_head = first_node;
            enqs_tail = first_node;

            for item in items_iter {
                let new_node = Box::into_raw(Box::new(Node::new(item)));
                unsafe {
                    (*enqs_tail).next.store_ptr(new_node);
                }
                enqs_tail = new_node;
            }
        } else {
            return;
        }

        let batch_req = InternalBatchRequest {
            first_enq: enqs_head,
            last_enq: enqs_tail,
        };
        self.execute_enqs_batch(batch_req);
    }
}

impl<T> Drop for BQueue<T>
where
    T: Send + Sync,
{
    fn drop(&mut self) {
        while self.dequeue().is_some() {}
        let pointer = self.tail.load_ptr();
        if !pointer.is_null() {
            unsafe { Domain::global().retire_node(pointer) };
        }
    }
}

pub struct DeqBatchIterator<T> {
    num_deqs_remaining: usize,
    last_deq_item: Option<T>,
    current: *mut Node<T>,
}

impl<'a, T> DeqBatchIterator<T>
where
    T: Send + Sync,
{
    fn new(num_deqs_remaining: usize, last_deq_item: Option<T>, current: *mut Node<T>) -> Self {
        Self {
            num_deqs_remaining: num_deqs_remaining,
            last_deq_item: last_deq_item,
            current: current,
        }
    }
}

impl<'a, T> Iterator for DeqBatchIterator<T>
where
    T: Send + Sync,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_deqs_remaining == 0 {
            return None;
        }
        if self.num_deqs_remaining == 1 {
            self.num_deqs_remaining -= 1;
            return self.last_deq_item.take();
        }
        let to_retire = self.current;
        self.current = unsafe { (*self.current).next.load_ptr() };
        unsafe { Domain::global().retire_node(to_retire) };
        self.num_deqs_remaining -= 1;
        return unsafe {
            Some(std::ptr::read(
                (*self.current).item.assume_init_ref() as *const _
            ))
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_enqueue_dequeue_single() {
        let q = BQueue::new();
        assert_eq!(q.dequeue(), None);
        q.enqueue(42);
        assert_eq!(q.dequeue(), Some(42));
        assert_eq!(q.dequeue(), None);
    }
    #[test]
    fn test_batch_deq_zero() {
        let q = BQueue::new();
        for i in 0..5 {
            q.enqueue(i);
        }
        let mut it = q.deq_batch(0);
        assert_eq!(it.next(), None);
    }
    #[test]
    fn test_batch_deq_exact() {
        let q = BQueue::new();
        for i in 0..3 {
            q.enqueue(i);
        }
        let it = q.deq_batch(3);
        assert_eq!(it.collect::<Vec<_>>(), vec![0, 1, 2]);
    }
    #[test]
    fn test_batch_deq_more() {
        let q = BQueue::new();
        for i in 0..2 {
            q.enqueue(i);
        }
        let it = q.deq_batch(5);
        assert_eq!(it.collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn test_multiple_enqueue_dequeue() {
        let q = BQueue::new();
        for i in 0..5 {
            q.enqueue(i);
        }
        for i in 0..5 {
            assert_eq!(q.dequeue(), Some(i));
        }
        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn test_enqueue_batch_empty() {
        let q: BQueue<i32> = BQueue::new();
        q.enqueue_batch(Vec::<i32>::new());
        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn test_enqueue_batch_and_single_dequeue() {
        let q = BQueue::new();
        q.enqueue_batch(vec![10, 20, 30]);
        assert_eq!(q.dequeue(), Some(10));
        assert_eq!(q.dequeue(), Some(20));
        assert_eq!(q.dequeue(), Some(30));
        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn test_execute_deqs_batch_partial() {
        let q = BQueue::new();
        for i in 0..5 {
            q.enqueue(i);
        }
        let it = q.deq_batch(3);
        assert_eq!(it.collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(q.dequeue(), Some(3));
        assert_eq!(q.dequeue(), Some(4));
        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn test_interleave_batch_and_single() {
        let q = BQueue::new();
        q.enqueue(100);
        q.enqueue_batch(vec![200, 300]);
        assert_eq!(q.dequeue(), Some(100));
        let it = q.deq_batch(2);
        assert_eq!(it.collect::<Vec<_>>(), vec![200, 300]);
        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn test_drop_does_not_crash() {
        let q = BQueue::new();
        for i in 0..10 {
            q.enqueue(i);
        }
        drop(q);
    }
}
