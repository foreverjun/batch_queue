// --- Зависимости и импорты ---
extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::iter::IntoIterator;
use core::marker::PhantomData;
use std::mem::MaybeUninit;
use core::ptr;
use core::sync::atomic::{AtomicPtr as StdAtomicPtr, AtomicUsize, Ordering};
use haphazard::raw::Pointer;
use haphazard::{raw, AtomicPtr, Domain, Global, HazardPointer, Singleton};

// --- Вспомогательные константы и функции для тегирования указателей ---
// ... (без изменений) ...
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

// --- Основные структуры данных ---
struct Node<T> {
    item: MaybeUninit<T>,
    next: AtomicPtr<Node<T>>,
}

type PtrOrAnn<T> = StdAtomicPtr<T>;

struct InternalBatchRequest<T> {
    first_enq: *mut Node<T>,
    last_enq: *mut Node<T>,
    enqs_num: usize,
}

struct Ann<T> {
    batch_req: InternalBatchRequest<T>,
    old_head_node: *mut Node<T>,
    old_head_count: usize,
    old_tail_node: AtomicPtr<Node<T>>,
    old_tail_count: AtomicUsize,
}
unsafe impl<T: Send> Send for Ann<T> {}

// --- Очередь BQ ---
pub struct BQueue<T> {
    head: PtrOrAnn<Node<T>>,
    tail: AtomicPtr<Node<T>>,
}

// --- Трейт-расширение для Domain ---
// Теперь методы можно вызывать прямо на Domain::global()
trait RetireHelpers {
    unsafe fn retire_node<T: Send>(&self, node: *mut Node<T>);
    unsafe fn retire_ann<T: Send>(&self, ann: *mut Ann<T>);
}

impl<F: 'static> RetireHelpers for Domain<F> {
    unsafe fn retire_node<T: Send>(&self, node: *mut Node<T>) {
        // Safety: Передаем гарантии безопасности от вызывающего кода
        unsafe {
            self.retire_ptr::<Node<T>, Box<Node<T>>>(node);
        }
    }
    unsafe fn retire_ann<T: Send>(&self, ann: *mut Ann<T>) {
        // Safety: Передаем гарантии безопасности от вызывающего кода
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

    fn enqueue_to_shared(&self, item: T) {
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

    fn dequeue_from_shared(&self) -> Option<T> {
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
            // let current_head_raw = self.head.load(Ordering::Acquire);
            // if !is_ann(current_head_raw) && current_head_raw != head_node_ptr {
            //     hp_head.reset_protection();
            //     hp_next.reset_protection();
            //     continue;
            // }

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
            // match self.head.compare_exchange(
            //     head_node_ptr,
            //     next_node_ptr,
            //     Ordering::Release,
            //     Ordering::Acquire,
            // ) {
            //     Ok(_) => {
            //         let item = unsafe { (*next_node_ptr).item.take() };
            //         // retire_node вызывается на глобальном домене
            //         unsafe {
            //             domain.retire_node(head_node_ptr);
            //         }
            //         hp_head.reset_protection();
            //         hp_next.reset_protection();
            //         return item;
            //     }
            //     Err(_) => {}
            // }

            hp_head.reset_protection();
            hp_next.reset_protection();
        }
    }

    // execute_deqs_batch: использует глобальный домен
    fn execute_deqs_batch(&self, num_deqs_requested: usize) -> (*mut Node<T>, usize) {
        let mut hp_head = HazardPointer::new();
        let mut hp_traverse = HazardPointer::new();

        loop {
            let (start_head_ptr, start_head_count) = self.help_ann_and_get_head(&mut hp_head); // Использует глоб. домен
                                                                                               // ... (логика обхода списка и подсчета) ...
            if start_head_ptr.is_null() {
                hp_head.reset_protection();
                return (ptr::null_mut(), 0);
            }
            let mut current_node_ptr = start_head_ptr;
            let mut successful_deqs = 0;
            let mut new_head_ptr = start_head_ptr;
            if unsafe { hp_traverse.protect_raw(current_node_ptr) }.is_null() {
                hp_head.reset_protection();
                continue;
            }
            for _ in 0..num_deqs_requested {
                let current_node_ref = unsafe { &*hp_traverse.deref().unwrap().as_ptr() };
                let next_node_opt = unsafe { current_node_ref.next.load(&mut hp_traverse) };
                if let Some(next_node) = next_node_opt {
                    new_head_ptr = next_node as *const _ as *mut _;
                    successful_deqs += 1;
                } else {
                    break;
                }
            }
            if successful_deqs == 0 {
                hp_head.reset_protection();
                hp_traverse.reset_protection();
                return (start_head_ptr, 0);
            }
            let new_head_count = start_head_count + successful_deqs;
            let new_head_ref = unsafe { &*hp_traverse.deref().unwrap().as_ptr() };
            new_head_ref.count.store(new_head_count, Ordering::Release);

            match self.head.compare_exchange(
                start_head_ptr,
                new_head_ptr,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    /* ... */
                    hp_head.reset_protection();
                    hp_traverse.reset_protection();
                    return (start_head_ptr, successful_deqs);
                }
                Err(_) => { /* ... */ }
            }
            hp_head.reset_protection();
            hp_traverse.reset_protection();
        }
    }

    // execute_enqs_batch: использует глобальный домен
    fn execute_enqs_batch(&self, batch_req: InternalBatchRequest<T>) -> *mut Node<T> {
        let ann_ptr = Box::into_raw(Box::new(Ann {
            batch_req, /* ... */
        }));
        let tagged_ann_ptr = tag_ann(ann_ptr);
        let mut hp_head = HazardPointer::new(); // Использует глоб. домен

        let original_head_ptr;
        loop {
            let head_node_ptr = self.help_ann_and_get_head(&mut hp_head); // Использует глоб. домен
                                                                                        // ... (сохранение head в ann) ...
            if head_node_ptr.is_null() {
                hp_head.reset_protection();
                continue;
            }
            unsafe {
                (*ann_ptr).old_head_node = head_node_ptr;
                (*ann_ptr).old_head_count = head_count;
                original_head_ptr = head_node_ptr;
            }

            match self.head.compare_exchange(
                head_node_ptr,
                tagged_ann_ptr,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    /* ... */
                    hp_head.reset_protection();
                    break;
                }
                Err(_) => { /* ... */ }
            }
            hp_head.reset_protection();
        }

        self.execute_ann(ann_ptr); // Использует глоб. домен

        // retire_ann на глоб. домене
        unsafe {
            domain.retire_ann(ann_ptr);
        } //

        original_head_ptr
    }

    fn execute_ann(&self, ann_ptr: *mut Ann<T>) {
        let ann = unsafe { &*ann_ptr };
        let mut hp_tail = HazardPointer::new();

        // --- Шаг 1: Связать элементы батча ---
        // ... (логика цикла, загрузки tail, CAS next) ...
        loop {
            let recorded_old_tail = ann.old_tail_node.load_ptr();
            if !recorded_old_tail.is_null() {
                break;
            }
            let tail_node_opt = unsafe { self.tail.load(&mut hp_tail) };
            if tail_node_opt.is_none() {
                hp_tail.reset_protection();
                continue;
            }
            let tail_node = tail_node_opt.unwrap();
            let tail_node_ptr = tail_node as *const _ as *mut _;
            let tail_count = tail_node.count.load(Ordering::Acquire);
            let std_next_atomic_ptr = unsafe { tail_node.next.as_std() };
            match std_next_atomic_ptr.compare_exchange(
                ptr::null_mut(),
                ann.batch_req.first_enq,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    /* ... */
                    let _ = ann
                        .old_tail_node
                        .compare_exchange_ptr(ptr::null_mut(), tail_node_ptr);
                    ann.old_tail_count.compare_exchange(
                        0,
                        tail_count,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    break;
                }
                Err(actual_next_ptr) => {
                    if actual_next_ptr == ann.batch_req.first_enq {
                        /* ... */
                        let _ = ann
                            .old_tail_node
                            .compare_exchange_ptr(ptr::null_mut(), tail_node_ptr);
                        ann.old_tail_count.compare_exchange(
                            0,
                            tail_count,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        );
                        break;
                    } else {
                        /* ... */
                        if !actual_next_ptr.is_null() {
                            let _ = unsafe {
                                self.tail
                                    .compare_exchange_ptr(tail_node_ptr, actual_next_ptr)
                            };
                        }
                    }
                }
            }
            hp_tail.reset_protection();
        }
        hp_tail.reset_protection();

        // --- Шаг 2: Обновить SQTail ---
        // ... (логика установки счетчика и CAS tail) ...
        let old_tail_node_ptr = ann.old_tail_node.load_ptr();
        let old_tail_count = ann.old_tail_count.load(Ordering::Acquire);
        let new_tail_node_ptr = ann.batch_req.last_enq;
        let num_enqs = ann.batch_req.enqs_num;
        let new_tail_count = old_tail_count + num_enqs;
        unsafe {
            (*new_tail_node_ptr)
                .count
                .store(new_tail_count, Ordering::Release);
        }
        let _ = unsafe {
            self.tail
                .compare_exchange_ptr(old_tail_node_ptr, new_tail_node_ptr)
        };

        // --- Шаг 3: Обновить Head ---
        // ... (логика CAS head) ...
        let target_head_node_ptr = ann.old_head_node;
        let current_head_raw = self.head.load(Ordering::Acquire);
        let _ = self.head.compare_exchange(
            current_head_raw,
            target_head_node_ptr,
            Ordering::Release,
            Ordering::Acquire,
        );
    }

    // --- Публичный интерфейс (без явного Domain) ---

    /// Одиночная операция Enqueue.
    pub fn enqueue(&self, item: T) {
        // Убрали domain
        self.enqueue_to_shared(item);
    }

    /// Одиночная операция Dequeue.
    pub fn dequeue(&self) -> Option<T> {
        // Убрали domain
        self.dequeue_from_shared()
    }

    /// Пакетная вставка из итератора.
    pub fn enqueue_batch<I>(&self, items: I)
    // Убрали domain
    where
        I: IntoIterator<Item = T>,
    {
        // ... (логика создания локального списка узлов без изменений) ...
        let mut enqs_head: *mut Node<T> = ptr::null_mut();
        let mut enqs_tail: *mut Node<T> = ptr::null_mut();
        let mut enqs_num = 0;
        let mut local_nodes: Vec<Box<Node<T>>> = Vec::new();
        for item in items {
            let new_node_box = Box::new(Node {
                item: Some(item),
                next: AtomicPtr::from(ptr::null_mut::<Node<T>>()),
                count: AtomicUsize::new(0),
            });
            local_nodes.push(new_node_box);
            let new_node_ptr = &**local_nodes.last().unwrap() as *const _ as *mut _;
            if enqs_head.is_null() {
                enqs_head = new_node_ptr;
                enqs_tail = new_node_ptr;
            } else {
                unsafe {
                    let tail_node_ref = &mut *enqs_tail;
                    tail_node_ref
                        .next
                        .as_std()
                        .store(new_node_ptr, Ordering::Relaxed);
                }
                enqs_tail = new_node_ptr;
            }
            enqs_num += 1;
        }
        if enqs_num == 0 {
            return;
        }

        let batch_req = InternalBatchRequest {
            /* ... */ first_enq: enqs_head,
            last_enq: enqs_tail,
            enqs_num,
        };
        // execute_enqs_batch использует глоб. домен
        let _ = self.execute_enqs_batch(batch_req); //

        // "Забываем" Box'ы
        for node_box in local_nodes {
            Box::into_raw(node_box);
        }
    }

    /// Пакетное извлечение до `max_count` элементов.
    pub fn dequeue_batch(&self, max_count: usize) -> Vec<Option<T>> {
        // Убрали domain
        if max_count == 0 {
            return Vec::new();
        }
        let domain = Domain::global(); // Получаем глобальный домен

        // execute_deqs_batch использует глоб. домен
        let (original_head_ptr, success_count) = self.execute_deqs_batch(max_count); //

        if success_count == 0 {
            /* ... */
            return core::iter::repeat(None).take(max_count).collect();
        }

        // Локальное сопоставление результатов и удаление узлов
        let mut results = Vec::with_capacity(max_count);
        let mut hp_traverse = HazardPointer::new(); // Использует глоб. домен

        let mut current_node_ptr = original_head_ptr;
        for i in 0..max_count {
            if i < success_count {
                let maybe_protected_node = unsafe { hp_traverse.protect_raw(current_node_ptr) }; // Использует глоб. домен
                                                                                                 // ... (логика извлечения item и retire_node) ...
                if maybe_protected_node.is_none() {
                    /* ... */
                    results.push(None);
                    current_node_ptr = ptr::null_mut();
                    continue;
                }
                let protected_node_ptr = maybe_protected_node.unwrap().as_ptr();
                let protected_node_ref = unsafe { &*protected_node_ptr };
                let next_node_ptr = protected_node_ref.next.load_ptr();
                let item = if !next_node_ptr.is_null() {
                    unsafe { (*next_node_ptr).item.take() }
                } else {
                    None
                };
                results.push(item);
                // retire_node на глоб. домене
                unsafe {
                    domain.retire_node(current_node_ptr);
                } //
                current_node_ptr = next_node_ptr;
            } else {
                results.push(None); //
            }
        }
        hp_traverse.reset_protection();
        // ... (дополнение results до max_count) ...
        while results.len() < max_count {
            results.push(None);
        }

        results
    }
}
