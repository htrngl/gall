// gall run MODEL --tokens FILE [-n 64] [-c 4096] [--prof]
// gall check MODEL REFDIR [-n 64] [-c 4096] [-ub]
// gall pp MODEL TOKENS [-n all] [-ub 512] [-c 4096] [--cmp] [--ref REFDIR]
// gall mmb
// gall tok MODEL FILE
// gall serve MODEL [--port 9940] [--ctx 102400] (see serve.rs)
// gall up [CTX] | down | log | model | test (see up.rs)
mod dr;
mod gguf;
mod gpu;
mod model;
mod mv;
mod serve;
mod tok;
mod up;
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
    let mut a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        return eprintln!("gall up [ctx] | down | log | model | test");
    }
    if a[1] == "test" {
        let d = up::cache().join("test");
        fs::create_dir_all(&d).unwrap();
        let f = [("tokens", include_str!("../tests/short/tokens.txt")),
            ("gen", include_str!("../tests/short/gen.txt"))];
        f.map(|(n, s)| fs::write(d.join(n).with_extension("txt"), s).unwrap());
        a = ["gall", "check", &up::model(), d.to_str().unwrap()]
            .map(String::from).into();
    }
    match a[1].as_str() {
        "up" | "down" | "log" | "model" => return up::main(&a[1..]),
        "dr" => return dr::main(&a[2..]),
        "run" if a.iter().any(|x| x == "--draft") => return dr::run(&a[2..]),
        "mmb" => return model::pf::bench(),
        "tok" => return tok::main(&a[2..]),
        "serve" => return serve::main(&a[2..]),
        _ => {}
    }
    let opt = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let n: usize = opt("-n").map_or(64, |v| v.parse().unwrap());
    let ctx = opt("-c").map_or(4096, |v| v.parse().unwrap());
    if a[1] == "kb" {
        // gall kb KERNEL GX,GY THREADS P0,P1,..: time 100 dispatches on zeros
        let g = gpu::Gpu::new();
        let b = [256, 512, 600, 128, 1].map(|n| g.buf(n << 20));
        let defs = std::env::var("DEFS").unwrap_or_default().replace(';', "\n");
        let pipe = g.pipe(include_str!("ops.metal"), &defs, &a[2]);
        let v = |s: &str| s.split(',').map(|x| x.parse().unwrap()).collect::<Vec<usize>>();
        let (gr, mut p) = (v(&a[3]), [0u32; 24]);
        v(&a[5]).iter().enumerate().for_each(|(i, &x)| p[i] = x as u32);
        let reps = std::env::var("REPS").map_or(10000, |v| v.parse().unwrap());
        for _ in 0..3 {
            let t = g.run(|e| {
                for _ in 0..reps {
                    let b = b.each_ref().map(|b| (b, 0));
                    gpu::call(e, &pipe, &p, &b, [gr[0], gr[1], 1], a[4].parse().unwrap());
                }
            });
            println!("{}: {:.1} us per dispatch", a[2], 1e6 * t / reps as f64);
        }
        return;
    }
    let mut m = model::Model::open(&a[2], ctx);
    m.reset();
    match a[1].as_str() {
        "run" => {
            let toks = ids(&opt("--tokens").expect("--tokens FILE"));
            let t0 = Instant::now();
            let mut next = m.feed(&toks);
            let pp = t0.elapsed().as_secs_f64();
            // --deep N: time decode as if N tokens deep, on a stale cache
            m.pos = opt("--deep").map_or(m.pos, |v| v.parse().unwrap());
            eprintln!("prompt: {} tokens, {:.1} t/s", toks.len(), toks.len() as f64 / pp);
            let (t0, mut gpu) = (Instant::now(), 0.);
            m.save();
            let mut out = vec![];
            for _ in 0..n {
                print!("{next} ");
                out.push(next);
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
            if a.iter().any(|x| x == "--rollback") {
                // 5 steps back by the rings, then back to the checkpoint
                for k in [n - 5, 0] {
                    assert!(m.truncate(toks.len() + k), "truncate {k}");
                    let r = out[k..].iter().fold(0, |_, &t| m.step(t));
                    eprintln!("rollback to +{k}: {}", if r == next { "ok" } else { "differs" });
                }
            }
            if a.iter().any(|x| x == "--prof") {
                // ablation: the token without one kind of dispatch
                let ops = m.sched(next, m.pos, true);
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
                // then the kind alone: ms, us per dispatch
                for (k, c) in by {
                    let f = |_| m.gpu.run(|e| {
                        ops.iter().filter(|o| o.0 == k).for_each(|o| o.2(e))
                    });
                    let t = c.min(1) as f64 * (0..8).map(f).fold(1., f64::min);
                    eprintln!(
                        "{k:8} {c:5} {:7.2} ms, alone {:6.2} ms {:6.1} us",
                        1e3 * (run("") - run(k)),
                        1e3 * t,
                        1e6 * t / c.max(1) as f64
                    );
                }
            }
        }
        "pp" => {
            // batched prefill: speed, and logits against stepping / a ref row
            let toks = ids(&a[3]);
            let toks = &toks[..opt("-n").map_or(toks.len(), |_| n)];
            let ub = opt("-ub").map_or(512, |v| v.parse().unwrap());
            let t0 = Instant::now();
            let rest = m.prefill(toks, ub);
            let g = (toks.len() - rest.len()) as f64 / m.t;
            let next = m.feed(rest);
            let (w, n) = (t0.elapsed().as_secs_f64(), toks.len() as f64);
            eprintln!("pp{n}: {:.1} t/s wall, {g:.1} gpu in passes, next {next}", n / w);
            let lg = m.logits();
            let diff = |r: &[f32]| {
                let d = r.iter().zip(&lg).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
                format!("max abs {d:.3}, kl {:.2e}", kl(r, &lg))
            };
            if let Some(d) = opt("--ref") {
                eprintln!("ref row 0: {}", diff(&f32s(&format!("{d}/logits.f32"))[..lg.len()]));
            }
            if a.iter().any(|x| x == "--cmp") {
                m.reset();
                let mut s = 0;
                toks.iter().for_each(|&t| s = m.step(t));
                eprintln!("stepped: next {s}, {}", diff(&m.logits()));
            }
        }
        "check" => {
            let d = &a[3];
            let (toks, gen) = (ids(&format!("{d}/tokens.txt")), ids(&format!("{d}/gen.txt")));
            let lg = f32s(&format!("{d}/logits.f32"));
            let nv = 200058;
            let (mut kls, mut flips, mut rows, mut gt) = (vec![], 0, vec![], vec![]);
            // -ub: the prompt by feed() (batched), not step by step.
            // LOGITS=file keeps every row
            let (mut next, ub) = (0, opt("-ub").is_some());
            if ub {
                next = m.feed(&toks);
            }
            for (i, &t) in toks.iter().chain(&gen[..n.min(gen.len() - 1)]).enumerate() {
                let Some(g) = i.checked_sub(toks.len()) else {
                    if !ub {
                        next = m.step(t);
                    }
                    continue;
                };
                // i is the step that feeds gen[g]; next predicts gen[g]
                let row = m.logits();
                if let Some(r) = lg.chunks(nv).nth(g) {
                    kls.push(kl(r, &row));
                }
                if std::env::var("LOGITS").is_ok() {
                    rows.extend(row.iter().flat_map(|v| v.to_le_bytes()));
                }
                if next != t {
                    flips += 1;
                    println!("step {g}: gall {next}, ref {t}");
                }
                let sd = format!("{d}/s{g:04}");
                if !fs::exists(&sd).unwrap() {
                    next = m.step(t);
                    gt.push(1e3 * m.t);
                    continue;
                }
                // one command buffer per visit; compare the named nodes
                let ops = m.sched(t, m.pos, true);
                let vs = model::visits();
                let mut worst: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
                for v in 0..=vs.len() {
                    m.gpu.run(|e| ops.iter().filter(|o| o.1 == v).for_each(|o| o.2(e)));
                    let Some(&(phys, slot, _)) = vs.get(v) else { break };
                    // a different expert set, and the reference's margin
                    // between its 8th and 9th
                    let rd = |n: &str| fs::read(format!("{sd}/ffn_moe_{n}-{slot}.i32")).unwrap();
                    let i32s = |b: Vec<u8>| -> Vec<u32> {
                        b.chunks(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
                    };
                    let (mut e, mut r) = (m.experts(), i32s(rd("topk")));
                    e.sort();
                    r.sort();
                    if e != r {
                        let (o, pb) = (i32s(rd("argsort")), f32s(&format!("{sd}/ffn_moe_probs_biased-{slot}.f32")));
                        let d = pb[o[7] as usize] - pb[o[8] as usize];
                        println!("step {g} visit {v} slot {slot}: experts differ, ref margin {d:.2e}");
                    }
                    for nm in ["Qcur", "Kcur", "Vcur", "Rcur", "kqv_out", "gate", "is", "ffn_inp",
                        "ffn_moe_out"] {
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
                        let e = if r.len() == x.len() && nm != "kqv_out" { rel(&x, &r) } else { 0. };
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
            if let Ok(f) = std::env::var("LOGITS") {
                fs::write(f, rows).unwrap();
            }
            println!("gpu ms/token: {:.1}", gt.iter().sum::<f64>() / gt.len().max(1) as f64);
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
