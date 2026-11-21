// ink run MODEL --tokens FILE [-n 64] [-c 4096] [--prof]
// ink check MODEL REFDIR [-n 64] [-c 4096]
// ink tok MODEL FILE
// ink serve MODEL [--port 9940] [--ctx 102400] (see serve.rs)
mod dr;
mod gguf;
mod gpu;
mod model;
mod mv;
mod serve;
mod tok;
use std::{collections::BTreeMap, fs, time::Instant};

fn ids(p: &str) -> Vec<u32> {
    let s = fs::read_to_string(p).unwrap_or_else(|_| panic!("no {p}"));
    s.split_whitespace().map(|v| v.parse().unwrap()).collect()
}

fn f32s(p: &str) -> Vec<f32> {
    let b = fs::read(p).unwrap_or_default();
    b.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

// kl(p || q) of two logit rows
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let lse = |l: &[f32]| {
        let m = l.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
        m + l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (a, b) = (lse(p), lse(q));
    p.iter()
        .zip(q)
        .map(|(&x, &y)| (x as f64 - a).exp() * (x as f64 - a - y as f64 + b))
        .sum()
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    (d / b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().max(1e-30)).sqrt()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "dr" => return dr::main(&a[2..]),
        "tok" => return tok::main(&a[2..]),
        "serve" => return serve::main(&a[2..]),
        _ => {}
    }
    let opt = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let n: usize = opt("-n").map_or(64, |v| v.parse().unwrap());
    let ctx = opt("-c").map_or(4096, |v| v.parse().unwrap());
    if a[1] == "kb" {
        // ink kb KERNEL GX,GY THREADS P0,P1,..: time 100 dispatches on zeros
        let g = gpu::Gpu::new();
        let b = [256, 512, 64, 128, 1].map(|n| g.buf(n << 20));
        let pipe = g.pipe(include_str!("ops.metal"), "", &a[2]);
        let v = |s: &str| s.split(',').map(|x| x.parse().unwrap()).collect::<Vec<usize>>();
        let (gr, mut p) = (v(&a[3]), [0u32; 24]);
        v(&a[5]).iter().enumerate().for_each(|(i, &x)| p[i] = x as u32);
        for _ in 0..3 {
            let t = g.run(|e| {
                for _ in 0..10000 {
                    let b = b.each_ref().map(|b| (b, 0));
                    gpu::call(e, &pipe, &p, &b, [gr[0], gr[1], 1], a[4].parse().unwrap());
                }
            });
            println!("{}: {:.1} us per dispatch", a[2], 1e2 * t);
        }
        return;
    }
    let mut m = model::Model::open(&a[2], ctx);
    m.reset();
    match a[1].as_str() {
        "run" => {
            let toks = ids(&opt("--tokens").expect("--tokens FILE"));
            let t0 = Instant::now();
            let mut next = 0;
            for &t in &toks {
                next = m.step(t);
            }
            let pp = t0.elapsed().as_secs_f64();
            eprintln!("prompt: {} tokens, {:.1} t/s", toks.len(), toks.len() as f64 / pp);
            let (t0, mut gpu) = (Instant::now(), 0.);
            for _ in 0..n {
                print!("{next} ");
                next = m.step(next);
                gpu += m.t;
            }
            let w = t0.elapsed().as_secs_f64() / n as f64;
            println!();
            eprintln!(
                "decode: {:.1} ms/token wall, {:.1} gpu, {:.1} t/s",
                1e3 * w,
                1e3 * gpu / n as f64,
                1. / w
            );
            if a.iter().any(|x| x == "--prof") {
                // ablation: the token without one kind of dispatch
                let ops = m.sched(next);
                let small = |k: &str| model::KINDS.contains(&k);
                let run = |skip: &str| {
                    let on = |k: &str| match skip {
                        "small" => !small(k),
                        "weights" => small(k),
                        _ => k != skip,
                    };
                    let f = |_| m.gpu.run(|e| {
                        ops.iter().filter(|o| on(o.0)).for_each(|o| o.2(e))
                    });
                    (0..12).map(f).fold(1f64, f64::min)
                };
                let all = run("");
                let mut by: BTreeMap<&str, usize> = BTreeMap::new();
                ops.iter().for_each(|o| *by.entry(o.0).or_default() += 1);
                eprintln!("{} dispatches, {:.1} ms", ops.len(), 1e3 * all);
                by.extend([("small", 0), ("weights", 0)]);
                for (k, c) in by {
                    eprintln!("{k:8} {c:5} {:7.2} ms", 1e3 * (run("") - run(k)));
                }
            }
        }
        "check" => {
            let d = &a[3];
            let (toks, gen) = (ids(&format!("{d}/tokens.txt")), ids(&format!("{d}/gen.txt")));
            let lg = f32s(&format!("{d}/logits.f32"));
            let nv = 200058;
            let (mut kls, mut flips) = (vec![], 0);
            let mut next = 0;
            for (i, &t) in toks.iter().chain(&gen[..n.min(gen.len() - 1)]).enumerate() {
                let Some(g) = i.checked_sub(toks.len()) else {
                    next = m.step(t);
                    continue;
                };
                // i is the step that feeds gen[g]; next predicts gen[g]
                if let Some(r) = lg.chunks(nv).nth(g) {
                    kls.push(kl(r, &m.logits()));
                }
                if next != t {
                    flips += 1;
                    println!("step {g}: ink {next}, ref {t}");
                }
                let sd = format!("{d}/s{g:04}");
                if !fs::exists(&sd).unwrap() {
                    next = m.step(t);
                    continue;
                }
                // one command buffer per visit; compare the named nodes
                let ops = m.sched(t);
                let vs = model::visits();
                let mut worst: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
                for v in 0..=vs.len() {
                    m.gpu.run(|e| ops.iter().filter(|o| o.1 == v).for_each(|o| o.2(e)));
                    let Some(&(phys, slot, _)) = vs.get(v) else { break };
                    for nm in ["Qcur", "Kcur", "Vcur", "Rcur", "ffn_inp",
                        "ffn_moe_weights_scaled", "ffn_moe_out"] {
                        let mut r = f32s(&format!("{sd}/{nm}-{slot}.f32"));
                        let mut x = m.peek(nm, phys).unwrap().to_vec();
                        if r.len() == 8 {
                            // router weights: compare sorted
                            x.sort_by(|a, b| b.total_cmp(a));
                            r.sort_by(|a, b| b.total_cmp(a));
                        }
                        if let Ok(d) = std::env::var("DUMP") {
                            let b: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
                            fs::write(format!("{d}/{nm}-{slot}.f32"), b).unwrap();
                        }
                        let e = rel(&x, &r);
                        let w = worst.entry(nm).or_insert((0., 0));
                        if e > w.0 {
                            *w = (e, slot);
                        }
                        if e > 1e-2 {
                            println!("step {g} visit {v} slot {slot} {nm}: rel {e:.3e}");
                        }
                    }
                }
                drop(ops);
                m.pos += 1;
                next = m.next();
                println!("step {g}: worst rel error (slot)");
                for (k, (e, s)) in worst {
                    println!("  {k:24} {e:.2e} ({s})");
                }
            }
            let mean = kls.iter().sum::<f64>() / kls.len().max(1) as f64;
            let max = kls.iter().fold(0f64, |a, &b| a.max(b));
            println!(
                "{} steps: {flips} greedy mismatches, kl mean {mean:.2e} max {max:.2e} ({} rows)",
                n.min(gen.len() - 1),
                kls.len()
            );
        }
        c => panic!("unknown command {c}"),
    }
}
