// vision: an image file to rows of the model's embedding width.
//   let v = vis::Vis::open(mmproj);
//   let (w, h, rgb) = vis::decode(&bytes, v.max)?;   // ImageIO
//   let e = v.embed(w, h, &rgb);                      // (w/40)(h/40) x 2304
// the encoder is inkling2v (llama.cpp tools/mtmd/models/inkling2v.cpp): a
// vit of 20-pixel patches with a cls token, bilinear position embeddings
// and 2d rope, then 2x2 patches merged through a 2-layer projector. it runs
// on the cpu through Accelerate's sgemm; weights are read from the map.
use crate::gguf::{open, Gguf};
use std::ffi::c_void;

type P = *const c_void;
#[repr(C)]
struct VBuf(P, usize, usize, usize); // data, height, width, row bytes
#[repr(C)]
struct Rect(f64, f64, f64, f64);

#[link(name = "Accelerate", kind = "framework")]
#[link(name = "ImageIO", kind = "framework")]
#[link(name = "CoreGraphics", kind = "framework")]
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    static kCGColorSpaceSRGB: P;
    fn CFDataCreate(a: P, b: *const u8, n: isize) -> P;
    fn CFRelease(p: P);
    fn CGImageSourceCreateWithData(d: P, o: P) -> P;
    fn CGImageSourceCreateImageAtIndex(s: P, i: usize, o: P) -> P;
    fn CGImageGetWidth(i: P) -> usize;
    fn CGImageGetHeight(i: P) -> usize;
    fn CGColorSpaceCreateWithName(n: P) -> P;
    fn CGBitmapContextCreate(d: *mut u8, w: usize, h: usize, bits: usize,
                             row: usize, cs: P, info: u32) -> P;
    fn CGContextSetInterpolationQuality(c: P, q: i32);
    fn CGContextDrawImage(c: P, r: Rect, i: P);
    fn vImageConvert_Planar16FtoPlanarF(s: &VBuf, d: &VBuf, f: u32) -> isize;
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(o: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32, al: f32,
                   a: *const f32, lda: i32, b: *const f32, ldb: i32, be: f32,
                   c: *mut f32, ldc: i32);
    fn erff(x: f32) -> f32;
}

const A: usize = 40; // pixels per output token side: patch 20, merge 2

// any format ImageIO reads, as srgb bytes over white; scaled down (aspect
// kept) while it would give more than max tokens
pub fn decode(b: &[u8], max: usize) -> Option<(usize, usize, Vec<u8>)> {
    unsafe {
        let d = CFDataCreate(0 as _, b.as_ptr(), b.len() as isize);
        let s = CGImageSourceCreateWithData(d, 0 as _);
        let i = if s.is_null() { s } else {
            CGImageSourceCreateImageAtIndex(s, 0, 0 as _)
        };
        let out = (!i.is_null()).then(|| {
            let (mut w, mut h) = (CGImageGetWidth(i), CGImageGetHeight(i));
            let n = |w: usize, h: usize| w.div_ceil(A) * h.div_ceil(A);
            while n(w, h) > max {
                let f = (max as f64 / n(w, h) as f64).sqrt().min(0.99);
                (w, h) = ((w as f64 * f) as usize, (h as f64 * f) as usize);
            }
            let (w, h) = (w.max(1), h.max(1));
            let mut x = vec![255u8; w * h * 4];
            let cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            // 5: rgbx, alpha skipped
            let c = CGBitmapContextCreate(x.as_mut_ptr(), w, h, 8, w * 4, cs, 5);
            CGContextSetInterpolationQuality(c, 3);
            CGContextDrawImage(c, Rect(0., 0., w as f64, h as f64), i);
            [c, cs, i].into_iter().for_each(|p| CFRelease(p));
            let rgb = x.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            (w, h, rgb)
        });
        if !s.is_null() {
            CFRelease(s);
        }
        CFRelease(d);
        out
    }
}

pub struct Vis {
    g: Gguf,
    pub max: usize, // tokens per image
}

// y (m x n) = a (m x k, row stride la) b^T (n x k, row stride lb), scaled
fn mm(m: usize, n: usize, k: usize, s: f32, a: &[f32], la: usize, b: &[f32],
      lb: usize, tb: bool, y: &mut [f32], ly: usize) {
    let i = |v: usize| v as i32;
    unsafe {
        cblas_sgemm(101, 111, 111 + tb as i32, i(m), i(n), i(k), s, a.as_ptr(),
                    i(la), b.as_ptr(), i(lb), 0., y.as_mut_ptr(), i(ly))
    }
}

impl Vis {
    pub fn open(path: &str) -> Vis {
        let max = std::env::var("GALL_IMAGE_TOKENS").ok();
        Vis { g: open(path), max: max.map_or(1024, |v| v.parse().unwrap()) }
    }

    // a tensor as f32 (f16 or f32 in the file)
    fn t(&self, name: &str) -> Vec<f32> {
        let t = &self.g.t[name];
        let (b, n) = (&self.g.map[t.off..t.off + t.len], t.dims.iter().product());
        let mut y = vec![0f32; n];
        if t.ty == 1 {
            let (s, d) = (VBuf(b.as_ptr() as P, 1, n, n * 2),
                          VBuf(y.as_ptr() as P, 1, n, n * 4));
            unsafe { vImageConvert_Planar16FtoPlanarF(&s, &d, 0) };
        } else {
            let f = |c: &[u8]| f32::from_le_bytes(c.try_into().unwrap());
            y.iter_mut().zip(b.chunks(4)).for_each(|(y, c)| *y = f(c));
        }
        y
    }

    // rows of x (width i) through weight NAME.weight (o x i), + NAME.bias
    fn lin(&self, x: &[f32], name: &str) -> Vec<f32> {
        let d = &self.g.t[format!("{name}.weight").as_str()].dims;
        let (o, i) = (d[d.len() - 1], d[..d.len() - 1].iter().product());
        let (w, m) = (self.t(&format!("{name}.weight")), x.len() / i);
        let mut y = vec![0f32; m * o];
        mm(m, o, i, 1., x, i, &w, i, true, &mut y, o);
        if self.g.t.contains_key(format!("{name}.bias").as_str()) {
            let b = self.t(&format!("{name}.bias"));
            y.chunks_mut(o).for_each(|r| r.iter_mut().zip(&b).for_each(|(y, b)| *y += b));
        }
        y
    }

    // layer norm of rows of 1024, then weight and bias
    fn ln(&self, x: &[f32], name: &str) -> Vec<f32> {
        let (w, b) = (self.t(&format!("{name}.weight")), self.t(&format!("{name}.bias")));
        let row = |r: &[f32]| {
            let n = r.len() as f32;
            let m = r.iter().sum::<f32>() / n;
            let v = r.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n;
            let s = 1. / (v + 1e-5).sqrt();
            let f = move |((x, w), b): ((&f32, &f32), &f32)| (x - m) * s * w + b;
            r.iter().zip(&w).zip(&b).map(f).collect::<Vec<_>>()
        };
        x.chunks(1024).flat_map(row).collect()
    }

    // rgb bytes (w x h) to (w/40 up)(h/40 up) rows of the model width
    pub fn embed(&self, w: usize, h: usize, rgb: &[u8]) -> Vec<f32> {
        const E: usize = 1024;
        let (gw, gh) = (w.div_ceil(A) * 2, h.div_ceil(A) * 2);
        let n = gw * gh + 1;
        // patches as (channel, y, x); pixels past the edge are raw -1
        let mut p = vec![0f32; gw * gh * 1200];
        for (i, v) in p.iter_mut().enumerate() {
            let (q, c, y, x) = (i / 1200, i / 400 % 3, i / 20 % 20, i % 20);
            let (x, y) = (q % gw * 20 + x, q / gw * 20 + y);
            let r = if x < w && y < h { rgb[(y * w + x) * 3 + c] as f32 } else { -1. };
            *v = (r - 127.5) / 127.5;
        }
        let mut x = vec![0f32; E];
        x.extend(self.lin(&p, "v.patch_embd"));
        // position grid (32 x 32), bilinear, half-pixel centres
        let (pe, at) = (self.t("v.position_embd.weight"), |i: usize, n: usize| {
            let f = ((i as f32 + 0.5) * 32. / n as f32 - 0.5).max(0.);
            let a = (f as usize).min(31);
            (a, (a + 1).min(31), f - a as f32)
        });
        for (q, r) in x.chunks_mut(E).skip(1).enumerate() {
            let ((x0, x1, dx), (y0, y1, dy)) = (at(q % gw, gw), at(q / gw, gh));
            for (e, r) in r.iter_mut().enumerate() {
                let g = |x: usize, y: usize| pe[(y * 32 + x) * E + e];
                *r += (g(x0, y0) * (1. - dx) + g(x1, y0) * dx) * (1. - dy)
                    + (g(x0, y1) * (1. - dx) + g(x1, y1) * dx) * dy;
            }
        }
        let cls = self.t("v.class_embd");
        let cp = self.t("v.class_position_embd.weight");
        x[..E].iter_mut().zip(cls.iter().zip(&cp)).for_each(|(x, (a, b))| *x = a + b);
        let mut x = self.ln(&x, "v.pre_ln");
        // rope: dims 0..32 of a head turn with the column, 32..64 with the
        // row; the cls token is at (0, 0), patches count from 1
        let rope = |v: &mut [f32]| {
            for (q, r) in v.chunks_mut(E).enumerate().skip(1) {
                let pos = [(q - 1) % gw + 1, (q - 1) / gw + 1];
                for (j, c) in r.chunks_mut(2).enumerate() {
                    let th = pos[j / 16 % 2] as f32 * 1e4f32.powf(-(j as f32 % 16.) / 16.);
                    let (s, k) = th.sin_cos();
                    (c[0], c[1]) = (c[0] * k - c[1] * s, c[0] * s + c[1] * k);
                }
            }
        };
        let gelu = |v: &mut [f32]| {
            let f = |x: &mut f32| *x *= 0.5 + 0.5 * unsafe { erff(*x * 0.5f32.sqrt()) };
            v.iter_mut().for_each(f)
        };
        let mut s = vec![0f32; n * n];
        for l in 0..self.g.u("clip.vision.block_count") {
            let b = |s: &str| format!("v.blk.{l}.{s}");
            let add = |x: &mut [f32], y: &[f32], ls: &str| {
                let ls = self.t(&b(ls));
                let f = |(i, (x, y)): (usize, (&mut f32, &f32))| *x += y * ls[i % E];
                x.iter_mut().zip(y).enumerate().for_each(f)
            };
            let y = self.ln(&x, &b("ln1"));
            let (mut q, mut k) = (self.lin(&y, &b("attn_q")), self.lin(&y, &b("attn_k")));
            let (v, mut o) = (self.lin(&y, &b("attn_v")), vec![0f32; n * E]);
            rope(&mut q);
            rope(&mut k);
            for hd in (0..E).step_by(64) {
                mm(n, n, 64, 0.125, &q[hd..], E, &k[hd..], E, true, &mut s, n);
                for r in s.chunks_mut(n) {
                    let m = r.iter().fold(f32::MIN, |a, &b| a.max(b));
                    r.iter_mut().for_each(|x| *x = (*x - m).exp());
                    let t = r.iter().sum::<f32>();
                    r.iter_mut().for_each(|x| *x /= t);
                }
                mm(n, 64, n, 1., &s, n, &v[hd..], E, false, &mut o[hd..], E);
            }
            add(&mut x, &self.lin(&o, &b("attn_out")), "ls1.weight");
            let mut y = self.lin(&self.ln(&x, &b("ln2")), &b("ffn_up"));
            gelu(&mut y);
            add(&mut x, &self.lin(&y, &b("ffn_down")), "ls2.weight");
        }
        // 2x2 patches side by side (row-major inside the square), projector
        let (tw, th) = (gw / 2, gh / 2);
        let mut m = Vec::with_capacity(tw * th * 4 * E);
        for i in 0..tw * th * 4 {
            let (t, dy, dx) = (i / 4, i / 2 % 2, i % 2);
            let q = 1 + (t / tw * 2 + dy) * gw + t % tw * 2 + dx;
            m.extend_from_slice(&x[q * E..][..E]);
        }
        let mut y = self.lin(&m, "mm.1");
        gelu(&mut y);
        self.lin(&y, "mm.2")
    }
}

// gall vis MMPROJ IMAGE [REF.f32]: encode, time, compare with tools/imgref
pub fn main(a: &[String]) {
    let v = Vis::open(&a[0]);
    let (w, h, rgb) = decode(&std::fs::read(&a[1]).unwrap(), v.max).expect("image");
    let t = std::time::Instant::now();
    let e = v.embed(w, h, &rgb);
    println!("{w} x {h}: {} tokens in {:.2} s", e.len() / 2304, t.elapsed().as_secs_f32());
    if let Some(r) = a.get(2) {
        let r = crate::f32s(r);
        let d = e.iter().zip(&r).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("rel {:.2e}, max abs {d:.2e} of {} rows", crate::rel(&e, &r), r.len() / 2304);
    }
}
