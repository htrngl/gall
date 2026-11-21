// dspark block drafter on the target's hidden states (DRAFT.md, dr.metal).
//   let d = Dr::open(&m, path, n_ctx);
//   let next = d.step(&mut m, tok);     // target step + context inject
//   let (ids, secs) = d.draft(&m, anchor, 7); // tokens after anchor at m.pos
// ink dr MODEL DRAFTER REFDIR [-n 64] [-c 4096]: teacher-forced acceptance
use crate::gguf::{self, Gguf, Tensor};
use crate::gpu::*;
use crate::model::{Model, Op};
use crate::mv::{name, Mv};
use std::collections::HashMap;

const N: usize = 2304;
const SRC: &str = include_str!("dr.metal");
const KINDS: [&str; 11] = [
    "tap", "fuse", "norm", "rope", "put", "attn", "silu", "add", "joint", "corr",
    "amax",
];
// target activations, as in model.rs: residual, normed output, its source
const TA: [usize; 3] = [2 * N, 3 * N, 15 * N];
// drafter activations, in floats; block regions hold 8 rows
const FT: usize = 0; // 4 target features
const CX: usize = 4 * N; // 3 layer contexts, f_rnn
const KC: usize = 8 * N; // context k, v per layer
const X: usize = KC + 3 * 2048; // residual
const HN: usize = X + 8 * N; // normed
const QQ: usize = HN + 8 * N;
const KK: usize = QQ + 8 * 3072; // k then v
const AT: usize = KK + 8 * 2048; // attention out
const OO: usize = AT + 8 * 3072; // wo and down out
const GG: usize = OO + 8 * N;
const UU: usize = GG + 8 * 9216;
const ST: usize = UU + 8 * 9216; // head state
const JJ: usize = ST + 256;
const YY: usize = JJ + 768;
const LG: usize = YY + N;

pub struct Dr {
    g: &'static Gguf,
    w: Gpu, // the drafter file's buffers
    mv: Mv,
    pipes: HashMap<&'static str, Pipe>,
    s: Buf,
    pub b: Buf,
    c: Buf,
    id: Buf,
    o: HashMap<&'static str, usize>, // f32 tensors in s
    co: [usize; 3],                  // cache rows before each layer
}

impl Dr {
    pub fn open(m: &Model, path: &str, n_ctx: usize) -> Dr {
        let g: &'static Gguf = Box::leak(Box::new(gguf::open(path)));
        let mut w = Gpu::new();
        w.map(g);
        let (mut s, mut o) = (vec![0f32], HashMap::new());
        let tn = &m.g.t["token_embd_norm.weight"];
        for (g, t) in g.t.values().map(|t| (g, t)).chain([(m.g, tn)]) {
            if t.ty == 0 {
                o.insert(t.name, s.len());
                s.extend(g.data(t).chunks(4).map(|b| {
                    f32::from_le_bytes(b.try_into().unwrap())
                }));
            }
        }
        let sb = m.gpu.buf(4 * s.len());
        slice::<f32>(&sb)[..s.len()].copy_from_slice(&s);
        Dr {
            mv: Mv::new(&m.gpu, ""),
            pipes: KINDS.map(|k| (k, m.gpu.pipe(SRC, "", k))).into(),
            s: sb,
            b: m.gpu.buf(4 * (LG + 201024)),
            c: m.gpu.buf(2 * 2048 * (2048 + n_ctx)),
            id: m.gpu.buf(4 * 8),
            co: [0, 1024, 1024 + n_ctx],
            g,
            w,
            o,
        }
    }

    fn t(&self, l: usize, n: &str) -> &'static Tensor {
        &self.g.t[format!("blk.{l}.{n}.weight").as_str()]
    }

    fn o(&self, l: usize, n: &str) -> usize {
        self.o[self.t(l, n).name]
    }

    // a dr.metal kernel; x: buffers after b, s, c, id
    fn k<'a>(&'a self, n: &'static str, g: [usize; 3], p: &[usize], x: &[&'static Tensor]) -> Op<'a> {
        let (mut a, x) = ([0u32; 8], x.to_vec());
        p.iter().enumerate().for_each(|(i, &v)| a[i] = v as u32);
        (n, 0, Box::new(move |e: &Enc| {
            let mut b = [&self.b, &self.s, &self.c, &self.id].map(|b| (b, 0)).to_vec();
            b.extend(x.iter().map(|t| self.w.w(t)));
            call(e, &self.pipes[n], &a, &b, g, if n == "amax" { 768 } else { 32 })
        }))
    }

    // rows of y = W x; input rows are n_in apart, output rows ys (0: n_out)
    fn m<'a>(&'a self, l: usize, n: &str, x: usize, y: usize, ys: usize, rows: usize) -> impl Iterator<Item = Op<'a>> {
        let t = self.t(l, n);
        let (ni, no) = (t.dims[0], t.dims[1]);
        (0..rows).map(move |r| -> Op {
            let (x, y) = (4 * (x + r * ni), 4 * (y + r * ys.max(no)));
            (name(t.ty), 0, Box::new(move |e: &Enc| {
                self.mv.dense(e, t.ty, self.w.w(t), (&self.b, x), (&self.b, y), ni, no)
            }))
        })
    }

    // context k, v and f_rnn of the token at pos, from the features
    fn inject(&self, pos: usize) -> Vec<Op<'_>> {
        let nw = [0, 1, 2].map(|l| self.o(l, "dflare_norm"));
        let p = [FT, CX, self.o["dspark_fuse.weight"], nw[0], nw[1], nw[2],
            self.o["dspark_norm.weight"]];
        let mut v = vec![self.k("fuse", [72, 4, 1], &p, &[])];
        for l in 0..3 {
            let (kc, r) = (KC + l * 2048, if l == 1 { pos } else { pos % 1024 });
            v.extend(self.m(l, "attn_k_ctx", CX + l * N, kc, 0, 1));
            v.extend(self.m(l, "attn_v_ctx", CX + l * N, kc + 1024, 0, 1));
            let p = [kc, 2048, self.o(l, "attn_k_ctx_norm"), pos];
            v.push(self.k("rope", [8, 1, 1], &p, &[]));
            v.push(self.k("put", [64, 1, 1], &[kc, (self.co[l] + r) * 2048], &[]));
        }
        v
    }

    // one target token, its features tapped and injected; greedy next id
    pub fn step(&self, m: &mut Model, tok: u32) -> u32 {
        let t = {
            let (m, mut ops) = (&*m, m.sched(tok));
            let tap = |i: usize, src: usize, vr: usize| -> Op {
                let p = [src, FT + i * N, vr].map(|v| v as u32);
                ("tap", 0, Box::new(move |e: &Enc| {
                    call(e, &self.pipes["tap"], &p, &[(&m.a, 0), (&self.b, 0)], [72, 1, 1], 32)
                }))
            };
            for (i, v) in [5, 23, 41].into_iter().enumerate() {
                let j = ops.iter().position(|o| o.0 == "elem" && o.1 == v);
                ops.insert(j.unwrap() + 1, tap(i, TA[0], 0));
            }
            ops.push(tap(3, TA[1], TA[2]));
            ops.extend(self.inject(m.pos));
            m.gpu.run(|e| ops.iter().for_each(|o| o.2(e)))
        };
        (m.t, m.pos) = (t, m.pos + 1);
        m.next()
    }

    // nb tokens after anchor a at position m.pos, and the gpu seconds
    pub fn draft(&self, m: &Model, a: u32, nb: usize) -> (Vec<u32>, f64) {
        let n = m.pos;
        slice::<f32>(&self.b)[ST..ST + 256].fill(0.);
        slice::<u32>(&self.id)[0] = a;
        let (emb, out) = (&m.g.t["token_embd.weight"], &m.g.t["output.weight"]);
        let mut v: Vec<Op> = vec![("row", 0, Box::new(move |e: &Enc| {
            self.mv.row(e, emb.ty, m.gpu.w(emb), a as usize, (&self.b, 4 * OO), N)
        }))];
        let g8 = [72, 8, 1];
        let p = [OO, X, self.o["token_embd_norm.weight"], self.o["mask_embd.weight"]];
        v.push(self.k("norm", g8, &p, &[]));
        for l in 0..3 {
            v.push(self.k("norm", g8, &[X, HN, self.o(l, "attn_norm")], &[]));
            v.extend(self.m(l, "attn_q", HN, QQ, 0, 8));
            v.extend(self.m(l, "attn_k", HN, KK, 2048, 8));
            v.extend(self.m(l, "attn_v", HN, KK + 1024, 2048, 8));
            v.push(self.k("rope", [24, 8, 1], &[QQ, 3072, self.o(l, "attn_q_norm"), n], &[]));
            v.push(self.k("rope", [8, 8, 1], &[KK, 2048, self.o(l, "attn_k_norm"), n], &[]));
            let (nk, ring) = if l == 1 { (n, 0) } else { (n.min(1024), 1024) };
            let p = [QQ, KK, self.co[l] * 2048, nk, n - nk, ring, AT];
            v.push(self.k("attn", [24, 8, 1], &p, &[]));
            v.extend(self.m(l, "attn_output", AT, OO, 0, 8));
            v.push(self.k("add", g8, &[X, OO], &[]));
            v.push(self.k("norm", g8, &[X, HN, self.o(l, "ffn_norm")], &[]));
            v.extend(self.m(l, "ffn_gate", HN, GG, 0, 8));
            v.extend(self.m(l, "ffn_up", HN, UU, 0, 8));
            v.push(self.k("silu", [288, 8, 1], &[GG, UU], &[]));
            v.extend(self.m(l, "ffn_down", GG, OO, 0, 8));
            v.push(self.k("add", g8, &[X, OO], &[]));
        }
        v.push(self.k("norm", g8, &[X, HN, self.o["output_norm.weight"]], &[]));
        let t = |n: &str| &self.g.t[n];
        for i in 1..=nb {
            let (c, jb, hb) = (HN + i * N, self.o["joint_proj.bias"], self.o["dspark_hidden_out.bias"]);
            let x = [t("joint_proj.weight"), t("markov_w1.weight")];
            v.push(self.k("joint", [24, 1, 1], &[ST, c, CX + 3 * N, JJ, jb, i - 1], &x));
            v.push(self.k("corr", [72, 1, 1], &[ST, JJ, c, YY, hb], &[t("dspark_hidden_out.weight")]));
            v.push(("head", 0, Box::new(move |e: &Enc| {
                let (x, y) = ((&self.b, 4 * YY), (&self.b, 4 * LG));
                self.mv.dense(e, out.ty, m.gpu.w(out), x, y, N, 201024)
            })));
            v.push(self.k("amax", [1, 1, 1], &[LG, 200058, i], &[]));
        }
        let t = m.gpu.run(|e| v.iter().for_each(|o| o.2(e)));
        (slice::<u32>(&self.id)[1..=nb].to_vec(), t)
    }
}

pub fn main(a: &[String]) {
    let opt = |k: &str, d: usize| {
        a.iter().position(|x| x == k).map_or(d, |i| a[i + 1].parse().unwrap())
    };
    let (n, d) = (opt("-n", 64), &a[2]);
    let mut m = Model::open(&a[0], opt("-c", 4096));
    m.reset();
    let dr = Dr::open(&m, &a[1], opt("-c", 4096));
    let mut all = crate::ids(&format!("{d}/tokens.txt"));
    let np = all.len();
    all.extend(crate::ids(&format!("{d}/gen.txt")));
    // run[k]: drafts whose first k tokens are the target's; hit: per slot
    let (mut run, mut hit, mut td, mut tt) = ([0usize; 8], [0usize; 7], 0., 0.);
    for i in 0..np + n {
        dr.step(&mut m, all[i]);
        if i == np {
            // the step that fed gen[0] is ref step s0000
            for (j, f) in ["l_out-4", "l_out-28", "l_out-46", "result_norm"].iter().enumerate() {
                let r = crate::f32s(&format!("{d}/s0000/{f}.f32"));
                let x = &slice::<f32>(&dr.b)[FT + j * N..][..N];
                println!("feature {j} ({f}): rel {:.2e}", crate::rel(x, &r));
            }
        }
        if i + 1 < np {
            continue;
        }
        tt += m.t;
        let (ids, t) = dr.draft(&m, all[i + 1], 7);
        td += t;
        let ok: Vec<bool> = (0..7).map(|k| ids[k] == all[i + 2 + k]).collect();
        run[ok.iter().take_while(|&&x| x).count()] += 1;
        (0..7).for_each(|k| hit[k] += ok[k] as usize);
    }
    let c = (n + 1) as f64;
    println!("target {:.1} ms, draft of 7 {:.1} ms", 1e3 * tt / c, 1e3 * td / c);
    println!("slot hit {:.2?}", hit.map(|h| h as f64 / c));
    for b in 1..8 {
        let e: usize = run.iter().enumerate().map(|(k, r)| r * (1 + k.min(b))).sum();
        println!("block {b}: {:.2} tokens per pass", e as f64 / c);
    }
}
