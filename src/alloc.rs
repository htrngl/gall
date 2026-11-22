// buffers of 1 MB and up are mmapped, so a drop gives their pages back at
// once: macOS malloc keeps large freed blocks, even after pressure relief
use std::alloc::{GlobalAlloc, Layout, System};

struct A;
#[global_allocator]
static G: A = A;

fn big(l: Layout) -> bool {
    l.size() >= 1 << 20 && l.align() <= 4096 // the page size or less
}

unsafe impl GlobalAlloc for A {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if !big(l) {
            return System.alloc(l);
        }
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let p = libc::mmap(0 as _, l.size(), rw,
                           libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0);
        if p == libc::MAP_FAILED { 0 as _ } else { p.cast() }
    }
    // anonymous pages are zero already
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if big(l) { self.alloc(l) } else { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if big(l) {
            libc::munmap(p.cast(), l.size());
        } else {
            System.dealloc(p, l)
        }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let m = Layout::from_size_align_unchecked(n, l.align());
        if !big(l) && !big(m) {
            return System.realloc(p, l, n);
        }
        let q = self.alloc(m);
        if !q.is_null() {
            p.copy_to_nonoverlapping(q, l.size().min(n));
            self.dealloc(p, l);
        }
        q
    }
}

// give free malloc pages back, after an unload
pub fn trim() {
    extern "C" {
        fn malloc_zone_pressure_relief(z: *mut u8, goal: usize) -> usize;
    }
    unsafe { malloc_zone_pressure_relief(0 as _, 0) };
}

#[test]
fn grow_and_shrink_across_1mb() {
    let mut v = vec![7u8; 1000];
    for n in [3 << 20, 5 << 20, 100] {
        v.resize(n, 7);
        v.shrink_to_fit();
        assert!(v.iter().all(|&b| b == 7));
    }
    assert!(vec![0u8; 3 << 20].iter().all(|&b| b == 0));
}
