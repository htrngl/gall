// gall mvtest [filter]: every weight kernel against the scalar reference,
// with its read rate. env: M (gguf), DEFS (extra #defines, ';' = newline).
// gall mvtest synth: attention in other formats (timing).
use crate::gpu::*;
use crate::{gguf, mv};
use std::collections::BTreeMap;

pub fn main() {
    let g = gguf::open(&std::env::var("M").expect("M=model.gguf"));
    let filt = std::env::args().nth(2).unwrap_or_default();
    let defs = std::env::var("DEFS").unwrap_or_default().replace(';', "\n") + "\n";
    let mut gpu = Gpu::new();
    gpu.map(&g);
    let t0 = std::time::Instant::now();
    let m = mv::Mv::new(&gpu, &defs);
    println!("compile {:.2} s", t0.elapsed().as_secs_f64());
    let mut seed = 1u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.
    };
    let x = gpu.buf(4 * 8 * 4608);
    slice::<f32>(&x).iter_mut().for_each(|v| *v = rnd());
    let y = gpu.buf(4 * 201024);
    let ids = gpu.buf(4 * 128); // 16 sets of 8 experts
    (0..128).for_each(|i| slice::<u32>(&ids)[i] = (i as u32 * 37 + 5) % 128);
    let (xs, ys, is) = (slice::<f32>(&x), slice::<f32>(&y), slice::<u32>(&ids));

    // ceiling: 1GB of weights, summed as 16-byte words
    if filt.is_empty() || filt == "bw" {
        let w = gpu.w(g.t.values().min_by_key(|t| t.off).unwrap());
        for n in [64u32, 1024, 16384] {
            let t = (1usize << 26) / n as usize;
            let p = gpu.pipe(mv::SRC, &defs, "bw");
            let run = || gpu.run(|e| call(e, &p, &[n, 0, 0, 0], &[(w.0, 0), (&y, 0), (&y, 0)], [t / 32, 1, 1], 32));
            let s = run().min(run());
            println!("bw {n:5} words/thread {:5.1} GB/s", (1u64 << 30) as f64 / s / 1e9);
        }
    }

    // one tensor per (type, kind, row width)
    let mut pick = BTreeMap::new();
    for t in g.t.values() {
        let d = &t.dims;
        if d.len() >= 2 && d[0] % 8 == 0 && d[0] >= 64 {
            let k = (d.len(), t.ty, d[0], t.name.contains("output.w"));
            if pick.get(&k).is_none_or(|o: &&gguf::Tensor| t.name < o.name) {
                pick.insert(k, t);
            }
        }
    }
    for t in pick.values() {
        if !t.name.contains(&filt) && !mv::name(t.ty).contains(&filt) {
            continue;
        }
        let (n_in, n_out, rb, w) = (t.dims[0], t.dims[1], t.row(), gpu.w(t));
        let b = g.data(t);
        let dot = |r: usize, x: &[f32]| -> f32 {
            let w = mv::deq(t.ty, &b[r * rb..], n_in);
            w.iter().zip(x).map(|(a, b)| (a * b) as f64).sum::<f64>() as f32
        };
        let (bx, by, bi) = ((&x, 0), (&y, 0), (&ids, 0));
        let up = t.name.contains("gate_up");
        let enc = |e: &Enc| match (t.dims.len(), up) {
            (2, _) => m.dense(e, t.ty, w, bx, by, n_in, n_out),
            (_, true) => m.experts_up(e, t.ty, w, bi, bx, by, n_in, n_out),
            _ => m.experts_down(e, t.ty, w, bi, bx, by, n_in, n_out),
        };
        // rows read per call, and the reference for a sample of outputs
        let rows = if t.dims.len() == 2 { n_out } else { mv::NE * n_out };
        let n_y = if up { rows } else { n_out };
        let step = (n_y / 500).max(1);
        ys.fill(f32::NAN);
        gpu.run(enc);
        let (mut err, mut nrm) = (0f32, 0f32);
        for r in (0..n_y).step_by(step).chain([n_y - 1]) {
            let v = match (t.dims.len(), up) {
                (2, _) => dot(r, xs),
                (_, true) => dot(is[r / n_out] as usize * n_out + r % n_out, xs),
                _ => (0..mv::NE)
                    .map(|e| dot(is[e] as usize * n_out + r, &xs[e * n_in..]))
                    .sum(),
            };
            err = err.max((v - ys[r]).abs());
            nrm = nrm.max(v.abs());
        }
        let nan = ys[..n_y].iter().filter(|v| v.is_nan()).count();
        // time over every tensor of this type and shape, expert sets cycling
        let all: Vec<_> = g.t.values().filter(|o| o.ty == t.ty && o.dims == t.dims).collect();
        let reps = (1.5e9 / (rows * rb) as f64).ceil().max(64.) as usize;
        let run = || {
            gpu.run(|e| {
                for i in 0..reps {
                    let (w, bi) = (gpu.w(all[i % all.len()]), (&ids, 32 * (i % 16)));
                    match (t.dims.len(), up) {
                        (2, _) => m.dense(e, t.ty, w, bx, by, n_in, n_out),
                        (_, true) => m.experts_up(e, t.ty, w, bi, bx, by, n_in, n_out),
                        _ => m.experts_down(e, t.ty, w, bi, bx, by, n_in, n_out),
                    }
                }
            })
        };
        let s = run().min(run()) / reps as f64;
        let kind = ["", "", "mv", if up { "up" } else { "down" }][t.dims.len()];
        println!(
            "{:8} {kind:4} {n_in:5}x{n_out:<6} {:6.1} MB {:7.3} ms {:5.1} GB/s  err {:.1e}{}  {}",
            mv::name(t.ty),
            (rows * rb) as f64 / 1e6,
            s * 1e3,
            (rows * rb) as f64 / s / 1e9,
            err / nrm,
            if nan > 0 || err / nrm > 1e-4 { " FAIL" } else { "" },
            t.name
        );
    }

    // synth: attention in every candidate format. each unit as a dense
    // mat-vec over slabs of expert bytes (timing only), against q8_0 on
    // the real tensors; n = q8_0 dispatches per token (layers 6-17 run 3x)
    if filt == "synth" {
        let nv: Vec<_> = g.t.values().filter(|t| t.ty == 40).collect();
        const TY: [u32; 8] = [8, 40, 20, 12, 13, 14, 105, 106];
        let mut tot = [0f64; 8];
        for (n_in, n_out) in [(2304, 3072), (2304, 1024), (2304, 384), (3072, 2304)] {
            let q8: Vec<_> = g.t.values().filter(|t| t.ty == 8 && t.dims == [n_in, n_out]).collect();
            let n: usize = q8.iter().map(|t| {
                let l: usize = t.name.split('.').nth(1).unwrap().parse().unwrap();
                if (6..18).contains(&l) { 3 } else { 1 }
            }).sum();
            let reps = 1000;
            let time = |ty: u32| {
                let slab = n_out * n_in / gguf::blk(ty).0 * gguf::blk(ty).1;
                let run = || gpu.run(|e| for i in 0..reps {
                    let w = if ty == 8 { gpu.w(q8[i % q8.len()]) } else {
                        let t = nv[i % nv.len()];
                        let (b, o) = gpu.w(t);
                        (b, o + (i * 7 % (t.len / slab - 1)) * slab)
                    };
                    m.dense(e, ty, w, (&x, 0), (&y, 0), n_in, n_out);
                }) / reps as f64 * 1e3;
                run().min(run())
            };
            let a = TY.map(time);
            let b = TY.map(time);
            print!("{n_in}x{n_out} x{n}:");
            for i in 0..8 {
                let v = a[i].min(b[i]);
                tot[i] += v * n as f64;
                print!(" {} {v:.4}", mv::name(TY[i]));
            }
            println!(" ms");
        }
        for i in 0..8 {
            println!("{:8} {:6.2} ms per token, {:5.2} saved", mv::name(TY[i]), tot[i], tot[0] - tot[i]);
        }
        return;
    }

    // row gather; the sums can be checked against gguf-py
    for t in pick.values() {
        if !filt.is_empty() && filt != "row" {
            break;
        }
        let (n, i) = (t.dims[0], 1234 % t.dims[1]);
        gpu.run(|e| m.row(e, t.ty, gpu.w(t), i, (&y, 0), n));
        let r = mv::deq(t.ty, &g.data(t)[i * t.row()..], n);
        let err = r.iter().zip(&*ys).map(|(a, b)| (a - b).abs()).fold(0., f32::max);
        let sum: f64 = r.iter().map(|v| v.abs() as f64).sum();
        println!("row {:8} err {err:.1e} sum {sum:.6} {:?}  {}", mv::name(t.ty), &r[..3], t.name);
    }
}
