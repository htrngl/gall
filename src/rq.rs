// requantized weights: dense tensors of the file again at 4, 5 or 6 bits
// in a buffer of our own, so that a token reads fewer bytes.
//   RQ=5                 every q8_0 attention projection at 5 bits
//   RQ=6:qkvr,4:og:6-17,5:h   groups of BITS[:KINDS[:LO-HI]]: the kinds
//                        q k v o g(ate) r(el) of layers LO-HI, h = head
//   let rq = Rq::open(g, &gpu, &cache_dir);
//   let (ty, w) = rq.w(&gpu, t);        // what to read in place of t
// a block is 32 values: an f16 scale, the low nibbles laid out as iq4_nl,
// then the high 1 or 2 bits (rq5, rq6: value = scale * (code - 16 or 32)).
// 4 bits is iq4_nl itself. every block's scale is searched for the least
// squared error. the buffer is kept in the cache directory.
use crate::gguf::{blk, Gguf, Tensor};
use crate::gpu::*;
use crate::mv::deq;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::{fs, io::Read, path::Path, thread, time::Instant};

#[derive(Default)]
pub struct Rq {
    buf: Option<Buf>,
    at: HashMap<&'static str, (u32, usize)>,
}

// tensor type of a bit width
pub fn ty(bits: usize) -> u32 {
    [20, 105, 106][bits - 4]
}

// bytes of n_out rows of n_in values
pub fn size(bits: usize, n_in: usize, n_out: usize) -> usize {
    n_out * n_in / 32 * blk(ty(bits)).1
}

// f32 to f16, round to nearest
fn h16(x: f32) -> u16 {
    let b = x.to_bits();
    let (s, e) = ((b >> 16 & 0x8000) as u16, (b >> 23 & 255) as i32 - 112);
    let m = b & 0x7fffff | 0x800000;
    match e {
        ..-10 => s,
        ..1 => s | ((m >> (14 - e)) + (m >> (13 - e) & 1)) as u16,
        _ => s | (((e as u32) << 10 | b >> 13 & 1023) + (b >> 12 & 1)) as u16,
    }
}

// scale, codes and squared error of 32 values on the ascending levels t:
// 8 scales (both signs at 4 bits, where the levels are lopsided), each
// refit once by least squares
fn block(x: &[f32], t: &[f32]) -> (f32, [u8; 32], f32) {
    let max = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let xx: f32 = x.iter().map(|v| v * v).sum();
    let mut best = (0., [(t.len() / 2) as u8; 32], xx);
    for c in 0..if t.len() == 16 { 16 } else { 8 } {
        let f = [1., -1.][c / 8] * (0.5375 + 0.075 * (c % 8) as f32);
        let mut s = f * max / -t[0];
        for _ in 0..2 {
            let (mut q, mut v, mut xt, mut tt) = ([0u8; 32], [0f32; 32], 0f32, 0f32);
            (0..32).for_each(|i| v[i] = x[i] / s);
            for m in t.windows(2).map(|w| (w[0] + w[1]) / 2.) {
                (0..32).for_each(|i| q[i] += (v[i] > m) as u8);
            }
            for i in 0..32 {
                let v = t[q[i] as usize];
                (xt, tt) = (xt + x[i] * v, tt + v * v);
            }
            if xt == 0. {
                break;
            }
            s = xt / tt;
            if xx - xt * s < best.2 {
                best = (s, q, xx - xt * s);
            }
        }
    }
    best
}

// rows of type ty and width n_in to `bits` in out, on every core;
// returns the squared error and the squared norm
pub fn quant(ty: u32, src: &[u8], n_in: usize, bits: usize, out: &mut [u8]) -> [f64; 2] {
    const NL: [i8; 16] =
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    let (hb, n) = (bits - 4, 1 << (bits - 1));
    let t: Vec<f32> = match bits {
        4 => NL.map(|v| v as f32).into(),
        _ => (-n..n).map(|v| v as f32).collect(),
    };
    let (ri, ro) = (n_in / blk(ty).0 * blk(ty).1, size(bits, n_in, 1));
    let add = |a: [f64; 2], b: [f64; 2]| [a[0] + b[0], a[1] + b[1]];
    let row = |(o, r): (&mut [u8], &[u8])| {
        let (x, mut e) = (deq(ty, r, n_in), [0.; 2]);
        for (o, x) in o.chunks_mut(ro * 32 / n_in).zip(x.chunks(32)) {
            let (s, q, er) = block(x, &t);
            e = add(e, [er as f64, x.iter().map(|v| (v * v) as f64).sum()]);
            o[..2].copy_from_slice(&h16(s).to_le_bytes());
            (0..16).for_each(|i| o[2 + i] = q[i] & 15 | q[i + 16] << 4);
            // the high bits of the 8 values of nibble bytes 4k..4k + 4
            for k in 0..4 {
                let hi = |j: usize| (q[4 * k + j % 4 + 16 * (j / 4)] >> 4) as u32;
                let h = (0..8).fold(0, |h, j| h | hi(j) << (hb * j));
                o[18 + hb * k..][..hb].copy_from_slice(&h.to_le_bytes()[..hb]);
            }
        }
        e
    };
    let cores = thread::available_parallelism().unwrap().get();
    let per = (src.len() / ri).div_ceil(cores);
    thread::scope(|s| {
        let parts = out.chunks_mut(per * ro).zip(src.chunks(per * ri));
        let rows = parts.map(|(o, i)| o.chunks_mut(ro).zip(i.chunks(ri)));
        let run = rows.map(|r| s.spawn(move || r.map(row).fold([0.; 2], add)));
        let h: Vec<_> = run.collect();
        h.into_iter().fold([0.; 2], |a, h| add(a, h.join().unwrap()))
    })
}

impl Rq {
    pub fn open(g: &'static Gguf, gpu: &Gpu, dir: &Path) -> Rq {
        let spec = std::env::var("RQ").unwrap_or_default();
        let (mut at, mut parts, mut n) = (HashMap::new(), vec![], 0);
        for part in spec.split(',').filter(|p| !p.is_empty()) {
            let mut f = part.split(':');
            let bits = f.next().unwrap().parse().ok().filter(|b| (4..7).contains(b));
            let bits: usize = bits.expect("RQ=BITS[:KINDS[:LO-HI]],.. with 4 to 6 bits");
            let kinds = f.next().unwrap_or("qkvogr");
            let r = f.next().and_then(|r| r.split_once('-')).unwrap_or(("0", "23"));
            let [lo, hi] = [r.0, r.1].map(|v| v.parse::<usize>().expect("RQ: layers"));
            let (mut jobs, n0) = (vec![], n);
            for (c, i) in kinds.chars().flat_map(|c| (lo..=hi).map(move |i| (c, i))) {
                let k = ["q", "k", "v", "output", "gate", "rel", "h"];
                let k = k.iter().find(|k| k.starts_with(c)).expect("RQ: kind");
                let name = format!("blk.{i}.attn_{k}.weight");
                let t = &g.t[if c == 'h' { "output.weight" } else { name.as_str() }];
                if (t.ty == 8 || c == 'h') && !at.contains_key(t.name) {
                    at.insert(t.name, (ty(bits), n));
                    jobs.push((t, n - n0));
                    n += size(bits, t.dims[0], t.dims[1]);
                }
            }
            parts.push((part, bits, jobs, n0..n));
        }
        if n == 0 {
            return Rq::default();
        }
        let buf = gpu.buf(n);
        for (part, bits, jobs, r) in parts {
            // a file per group, named by its tensors
            let out = &mut slice::<u8>(&buf)[r];
            let mut h = DefaultHasher::new();
            (1, bits, g.map.len()).hash(&mut h);
            jobs.iter().for_each(|j| (j.0.off, &g.data(j.0)[..64]).hash(&mut h));
            let p = dir.join(format!("{:016x}.rq", h.finish()));
            let n = out.len();
            let read = |mut f: fs::File| {
                f.metadata().unwrap().len() == n as u64 && f.read_exact(out).is_ok()
            };
            if fs::File::open(&p).is_ok_and(read) {
                continue;
            }
            let (t0, mut e) = (Instant::now(), 0.);
            for &(t, o) in &jobs {
                let o = &mut out[o..][..size(bits, t.dims[0], t.dims[1])];
                let r = quant(t.ty, g.data(t), t.dims[0], bits, o);
                e += r[0] / r[1] / jobs.len() as f64;
            }
            let tmp = p.with_extension(std::process::id().to_string());
            fs::write(&tmp, &*out).and_then(|_| fs::rename(&tmp, &p)).expect("rq: cache");
            eprintln!(
                "gall: RQ={part}: {} tensors, {} MB, relative error {:.4}, {:.1} s",
                jobs.len(),
                n >> 20,
                e.sqrt(),
                t0.elapsed().as_secs_f64()
            );
        }
        Rq { buf: Some(buf), at }
    }

    pub fn w<'a>(&'a self, gpu: &'a Gpu, t: &Tensor) -> (u32, B<'a>) {
        match self.at.get(t.name) {
            Some(&(ty, o)) => (ty, (self.buf.as_ref().unwrap(), o)),
            _ => (t.ty, gpu.w(t)),
        }
    }
}
