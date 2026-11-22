// metal host. usage:
//   let g = gguf::open(path);
//   let mut gpu = Gpu::new();
//   gpu.map(&g);                         // wrap the mmap, no copy
//   let w = gpu.w(&g.t["blk.0.attn_q.weight"]);     // (buffer, offset)
//   let x = gpu.buf(4 * 2304);           // zeroed, shared with the cpu
//   slice::<f32>(&x)[0] = 1.0;           // cpu view
//   let p = gpu.pipe(SRC, "#define N 4\n", "kernel_name");  // cached
//   let s = gpu.run(|e| {                // one command buffer, gpu seconds
//       call(e, &p, &args, &[w, (&x, 0)], [groups_x, 1, 1], 32);
//   });
// in the kernel, args is buffer 0 (a #[repr(C)] struct, copied) and
// bufs follow from index 1.
use crate::gguf::{Gguf, Tensor};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::{cell::RefCell, collections::HashMap, ffi::c_void, ptr::NonNull};

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

pub type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Enc = ProtocolObject<dyn MTLComputeCommandEncoder>;
pub type B<'a> = (&'a Buf, usize);

pub struct Gpu {
    pub dev: Retained<ProtocolObject<dyn MTLDevice>>,
    pub q: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    w: Vec<(usize, Buf)>, // file offset of each weight buffer, ascending
    libs: RefCell<HashMap<(usize, String), Retained<ProtocolObject<dyn MTLLibrary>>>>,
    pipes: RefCell<HashMap<(usize, String, String), Pipe>>,
}

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

impl Gpu {
    pub fn new() -> Gpu {
        let dev = MTLCreateSystemDefaultDevice().expect("no metal device");
        let q = dev.newCommandQueue().unwrap();
        let (w, libs, pipes) = Default::default();
        Gpu { dev, q, w, libs, pipes }
    }

    // wrap the mapped file in as few buffers as the device allows. each
    // starts on the page of a tensor start, so no tensor straddles two.
    pub fn map(&mut self, g: &Gguf) {
        let pg = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let max = self.dev.maxBufferLength() / pg * pg;
        let mut t: Vec<_> = g.t.values().map(|t| (t.off, t.len)).collect();
        t.sort();
        let end = g.map.len().div_ceil(pg) * pg;
        let mut cuts = vec![t[0].0 / pg * pg];
        for (o, n) in t {
            if o + n - cuts.last().unwrap() > max {
                cuts.push(o / pg * pg);
            }
        }
        for (i, &a) in cuts.iter().enumerate() {
            let n = cuts.get(i + 1).map_or(end, |b| b + pg).min(end) - a;
            let p = NonNull::new(g.map[a..].as_ptr() as *mut c_void).unwrap();
            let b = unsafe {
                self.dev.newBufferWithBytesNoCopy_length_options_deallocator(
                    p, n, SHARED, None,
                )
            };
            self.w.push((a, b.expect("cannot wrap the weights")));
        }
    }

    pub fn w(&self, t: &Tensor) -> B<'_> {
        let i = self.w.partition_point(|w| w.0 <= t.off) - 1;
        (&self.w[i].1, t.off - self.w[i].0)
    }

    pub fn buf(&self, bytes: usize) -> Buf {
        self.dev.newBufferWithLength_options(bytes.max(16), SHARED).unwrap()
    }

    // kernel `func` of `defs` + `src`. libraries and pipelines are cached;
    // keep the result for hot paths.
    pub fn pipe(&self, src: &'static str, defs: &str, func: &str) -> Pipe {
        let k = (src.as_ptr() as usize, defs.to_string(), func.to_string());
        if let Some(p) = self.pipes.borrow().get(&k) {
            return p.clone();
        }
        let mut libs = self.libs.borrow_mut();
        let lib = libs.entry((k.0, k.1.clone())).or_insert_with(|| {
            let o = MTLCompileOptions::new();
            o.setMathMode(MTLMathMode::Fast);
            self.dev
                .newLibraryWithSource_options_error(
                    &NSString::from_str(&format!("{defs}{src}")),
                    Some(&o),
                )
                .unwrap_or_else(|e| panic!("metal: {e}"))
        });
        let f = lib
            .newFunctionWithName(&NSString::from_str(func))
            .unwrap_or_else(|| panic!("metal: no kernel {func}"));
        let p = self
            .dev
            .newComputePipelineStateWithFunction_error(&f)
            .unwrap_or_else(|e| panic!("metal: {func}: {e}"));
        self.pipes.borrow_mut().insert(k, p.clone());
        p
    }

    // encode with f, commit, wait. returns gpu seconds.
    pub fn run(&self, f: impl FnOnce(&Enc)) -> f64 {
        autoreleasepool(|_| {
            let cb = self.q.commandBuffer().unwrap();
            let e = cb.computeCommandEncoder().unwrap();
            f(&e);
            e.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            if let Some(e) = cb.error() {
                panic!("metal: {e}");
            }
            cb.GPUEndTime() - cb.GPUStartTime()
        })
    }
}

pub fn call<P>(e: &Enc, p: &Pipe, args: &P, bufs: &[B], groups: [usize; 3], threads: usize) {
    let sz = |[width, height, depth]: [usize; 3]| MTLSize { width, height, depth };
    // a larger threadgroup runs nothing, without an error
    assert!(threads <= p.maxTotalThreadsPerThreadgroup(), "threadgroup of {threads}");
    unsafe {
        e.setComputePipelineState(p);
        e.setBytes_length_atIndex(NonNull::from(args).cast(), size_of::<P>(), 0);
        for (i, (b, o)) in bufs.iter().enumerate() {
            e.setBuffer_offset_atIndex(Some(b), *o, i + 1);
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(sz(groups), sz([threads, 1, 1]));
    }
}

// cpu view of a shared buffer
#[allow(clippy::mut_from_ref)]
pub fn slice<T>(b: &Buf) -> &mut [T] {
    let n = b.length() / size_of::<T>();
    unsafe { std::slice::from_raw_parts_mut(b.contents().as_ptr().cast(), n) }
}
