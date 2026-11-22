// vision: an image file to rows of the model's embedding width.
//   let v = vis::Vis::open(mmproj);
//   let (w, h, rgb) = vis::decode(&bytes, v.max)?;   // ImageIO
//   let e = v.embed(w, h, &rgb);                      // (w/40)(h/40) x 2304
// the encoder is inkling2v (llama.cpp tools/mtmd/models/inkling2v.cpp): a
// vit of 20-pixel patches with a cls token, bilinear position embeddings
// and 2d rope, then 2x2 patches merged through a 2-layer projector. the
// layers run on the gpu: weights from the map by prefill's mm_f16, the
// rest in vis.metal.
use crate::gguf::{open, Gguf};
use crate::gpu::{call, slice, Buf, Enc, Gpu, Pipe};
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
    gpu: Gpu,
    k: [Pipe; 6],
    mm: Pipe,
    s: Buf,         // per layer: ln1 w b, ln2 w b, ls1, ls2, biases
    pub max: usize, // tokens per image
}

const VS: usize = 14 * 1024; // floats per position
const SL: usize = 15 * 1024; // small floats per layer
const QC: usize = 256; // queries per attention pass

// y (m x n) = a (m x k, row stride la) b^T (n x k, row stride lb)
fn mm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], y: &mut [f32]) {
    let i = |v: usize| v as i32;
    unsafe {
        cblas_sgemm(101, 111, 112, i(m), i(n), i(k), 1., a.as_ptr(), i(k),
                    b.as_ptr(), i(k), 0., y.as_mut_ptr(), i(n))
    }
}

impl Vis {
    pub fn open(path: &str) -> Vis {
        let (g, mut gpu) = (open(path), Gpu::new());
        gpu.map(&g);
        let max = std::env::var("GALL_IMAGE_TOKENS").ok();
        let d = format!("#define VS {VS}\n");
        let k = ["vln", "vqkv", "vsc", "vres", "vgelu", "vwv"];
        let k = k.map(|k| gpu.pipe(include_str!("vis.metal"), &d, k));
        let (mm, s) = (gpu.pipe(crate::model::pf::MM, "", "mm_f16"),
            gpu.buf(4 * SL * g.u("clip.vision.block_count")));
        let mut v = Vis { g, gpu, k, mm, s, max: max.map_or(1024, |v| v.parse().unwrap()) };
        let names = ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
            "ls1.weight", "ls2.weight", "attn_q.bias", "attn_k.bias",
            "attn_v.bias", "attn_out.bias", "ffn_down.bias", "ffn_up.bias"];
        for (l, r) in slice::<f32>(&v.s.clone()).chunks_mut(SL).enumerate() {
            for (i, n) in names.iter().enumerate() {
                let t = v.t(&format!("v.blk.{l}.{n}"));
                r[i * 1024..][..t.len()].copy_from_slice(&t);
            }
        }
        v.max = v.max.max(1);
        v
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
        mm(m, o, i, x, &w, &mut y);
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
        let x = self.ln(&x, "v.pre_ln");
        // the layers on the gpu. tiles of 32 positions, then of 32 tokens
        let (np, nt) = (n.div_ceil(8) * 8, gw * gh / 4);
        let tiles = |n: usize| (0..n.div_ceil(32)).flat_map(move |i| {
            [0, 32 * i as u32, (n.div_ceil(8) * 8 - 32 * i).min(32) as u32]
        });
        let tl: Vec<u32> = tiles(n).chain(tiles(nt)).collect();
        let (g, t0) = (&self.gpu, n.div_ceil(32));
        let (a, tb, rb, sb) = (g.buf(4 * VS * (np + 8)), g.buf(4 * tl.len()),
            g.buf(8 * 32 * n), g.buf(4 * QC * 16 * n));
        slice::<u32>(&tb).copy_from_slice(&tl);
        let av = slice::<f32>(&a);
        for (i, r) in x.chunks(E).enumerate() {
            av[i * VS..][..E].copy_from_slice(r);
        }
        // rope: dims 0..32 of a head turn with the column, 32..64 with the
        // row; the cls token is at (0, 0), patches count from 1
        let sc = |p: usize| (0..16).map(move |j| {
            let (s, c) = (p as f32 * 1e4f32.powf(-(j as f32) / 16.)).sin_cos();
            [s, c]
        });
        let sc = (0..gw * gh).flat_map(|q| sc(q % gw + 1).chain(sc(q / gw + 1)));
        let rv = slice::<[f32; 2]>(&rb);
        rv[..32].fill([0., 1.]);
        rv[32..].iter_mut().zip(sc).for_each(|(r, s)| *r = s);
        // y = W x for tiles (t0, nt): fields at x and y
        let mm = |e: &Enc, w: &str, x: usize, y: usize, tl: (usize, usize)| {
            let w = &self.g.t[w];
            let (ni, no) = (w.dims[0], w.dims[1]);
            let p = [ni, no, 2 * ni, 0, x, VS, y, VS].map(|v| v as u32);
            let b = [g.w(w), (&a, 0), (&a, 0), (&tb, 0), (&tb, 12 * tl.0)];
            call(e, &self.mm, &p, &b, [no / 64, tl.1, 1], 128)
        };
        let k = |e: &Enc, k: usize, gr: [usize; 3], p: [usize; 5]| {
            let b = [(&a, 0), (&self.s, 0), (&rb, 0), (&sb, 0)];
            let gr = [gr[0].div_ceil(32), gr[1], gr[2]];
            call(e, &self.k[k], &p.map(|v| v as u32), &b, gr, 32)
        };
        // fields of a position: x, y, q k v, o, two of 4096
        let [y, q, o, f, f2] = [E, 2 * E, 5 * E, 6 * E, 10 * E];
        let all = (0, t0);
        g.run(|e| for l in 0..self.g.u("clip.vision.block_count") {
            let (w, s) = (|s: &str| format!("v.blk.{l}.{s}.weight"), l * SL);
            k(e, 0, [n, 1, 1], [n, 0, y, s, s + E]);
            mm(e, &w("attn_q"), y, q, all);
            mm(e, &w("attn_k"), y, q + E, all);
            mm(e, &w("attn_v"), y, q + 2 * E, all);
            k(e, 1, [n, 16, 1], [n, q, 0, 0, s + 6 * E]);
            for q0 in (0..n).step_by(QC) {
                let nq = QC.min(n - q0);
                k(e, 2, [nq, 16, 1], [n, q, 0, q0, 0]);
                k(e, 5, [nq, 16, 4], [n, q, y, q0, 0]);
            }
            mm(e, &w("attn_out"), y, o, all);
            k(e, 3, [n, 1, 1], [n, 0, o, s + 4 * E, s + 9 * E]);
            k(e, 0, [n, 1, 1], [n, 0, y, s + 2 * E, s + 3 * E]);
            mm(e, &w("ffn_up"), y, f, all);
            k(e, 4, [n, 64, 1], [n, f, 0, 1, s + 11 * E]);
            mm(e, &w("ffn_down"), f, o, all);
            k(e, 3, [n, 1, 1], [n, 0, o, s + 5 * E, s + 10 * E]);
        });
        // 2x2 patches side by side (row-major inside the square), projector
        let (tw, x) = (gw / 2, av[..n * VS].to_vec());
        for i in 0..nt * 4 {
            let (t, dy, dx) = (i / 4, i / 2 % 2, i % 2);
            let q = 1 + (t / tw * 2 + dy) * gw + t % tw * 2 + dx;
            av[t * VS + f + i % 4 * E..][..E].copy_from_slice(&x[q * VS..][..E]);
        }
        g.run(|e| {
            mm(e, "mm.1.weight", f, f2, (t0, nt.div_ceil(32)));
            k(e, 4, [nt, 64, 1], [nt, f2, 0, 0, 0]);
            mm(e, "mm.2.weight", f2, f, (t0, nt.div_ceil(32)));
        });
        (0..nt).flat_map(|t| av[t * VS + f..][..2304].iter().copied()).collect()
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
