// dspark block drafter on the target's hidden states (DRAFT.md, dr.metal).
//   let d = Dr::open(&m, path, n_ctx);
//   let next = d.step(&mut m, tok);     // target step + context inject
//   let (ids, secs) = d.draft(&m, anchor, 7); // tokens after anchor at m.pos
//   let next = d.pass(&m, &[anchor, ids..]); d.commit(&mut m, rows, kept);
//   d.generate(&mut m, next, &mut Spec::new(), |tok| more);  // all of it
// ink dr MODEL DRAFTER REFDIR [-n 64] [-c 4096]: teacher-forced acceptance
// ink run MODEL --tokens FILE --draft DRAFTER [--block 2] [-n 64] [-c 4096]
use crate::gguf::{self, Gguf, Tensor};
use crate::gpu::*;
use crate::model::{Model, Op};
use objc2_metal::*;
use std::time::Instant;
use crate::mv::{name, Mv};
use std::collections::HashMap;

const N: usize = 2304;
const SRC: &str = include_str!("dr.metal");
const MM: &str = concat!(
    "#include <metal_stdlib>\n",
    include_str!("grids.metal"),
    include_str!("mv.metal"),
    include_str!("drm.metal")
);
const KINDS: [&str; 14] = [
    "tap", "fuse", "norm", "rope", "put", "attn", "silu", "add", "joint", "pl",
    "pset", "amax", "uni", "dsum",
];
const XT: [u32; 10] = [8, 12, 13, 14, 16, 17, 18, 20, 22, 40]; // drm.metal x_*
// target activations, as in model.rs: residual, normed output, its source
const TA: [usize; 3] = [2 * N, 3 * N, 15 * N];
// drafter activations, in floats; block regions hold 8 rows
const CX: usize = 0; // 3 layer contexts, f_rnn
const KC: usize = 4 * N; // context k, v per layer
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
const LG: usize = JJ + 768;
const WB: usize = LG + 201024; // output hidden_out bias
const BL: usize = WB + 201024; // base logits, 8 rows
const NV: usize = 201024;
const FR: usize = BL + 8 * NV; // 4 target features per verify row

pub struct Dr {
    g: &'static Gguf,
    w: Gpu, // the drafter file's buffers
    mv: Mv,
    pipes: HashMap<&'static str, Pipe>,
    s: Buf,
    pub b: Buf,
    c: Buf,
    id: Buf,
    p: Buf, // output hidden_out, 201024 x 256 f16
    xp: HashMap<(u32, usize), Pipe>, // slab kernels by type and rows
    ut: Buf, // expert union of a pass; t0: the one dense slab
    t0: Buf,
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
        let mut pipes: HashMap<_, _> = KINDS.map(|k| (k, m.gpu.pipe(SRC, "", k))).into();
        for k in ["q8_0_8", "q5_k_1", "q5_k_2", "q5_k_3", "q5_k_4", "q5_k_7", "q5_k_8"] {
            pipes.insert(k, m.gpu.pipe(MM, "", &format!("mm_{k}")));
        }
        let xp = XT.iter().flat_map(|&t| [2, 4, 8].map(move |r| (t, r))).map(|(t, r)| {
            ((t, r), m.gpu.pipe(MM, "", &format!("x_{}_{r}", name(t))))
        });
        let d = Dr {
            xp: xp.collect(),
            ut: m.gpu.buf(4 * 64 * 9),
            t0: m.gpu.buf(4 * 9),
            mv: Mv::new(&m.gpu, ""),
            pipes,
            s: sb,
            b: m.gpu.buf(4 * (FR + 32 * N)),
            p: m.gpu.buf(2 * 256 * NV),
            c: m.gpu.buf(2 * 2048 * (2048 + n_ctx)),
            id: m.gpu.buf(4 * 8),
            co: [0, 1024, 1024 + n_ctx],
            g,
            w,
            o,
        };
        // P and wb: the head applied to hidden_out's columns and bias
        let (b, out) = (slice::<f32>(&d.b), &m.g.t["output.weight"]);
        let ho = d.g.data(&d.g.t["dspark_hidden_out.weight"]);
        let hb = d.o["dspark_hidden_out.bias"];
        b[X..X + N].copy_from_slice(&slice::<f32>(&d.s)[hb..hb + N]);
        m.gpu.run(|e| d.mm(out, m.gpu.w(out), X, WB, NV, 1).2(e));
        for k in (0..256).step_by(8) {
            for (i, x) in b[HN..HN + 8 * N].iter_mut().enumerate() {
                let j = 2 * (i % N * 256 + k + i / N);
                *x = f32::from_bits((ho[j] as u32) << 16 | (ho[j + 1] as u32) << 24);
            }
            m.gpu.run(|e| {
                d.mm(out, m.gpu.w(out), HN, BL, NV, 8).2(e);
                d.k("pset", [6252, 1, 1], &[BL, k], vec![(&d.p, 0)]).2(e);
            });
        }
        d
    }

    // rows of y = W x in one pass: nt inputs n_in apart, outputs ys apart
    fn mm<'a>(&'a self, t: &'static Tensor, w: B<'a>, x: usize, y: usize, ys: usize, nt: usize) -> Op<'a> {
        let (k, r) = if t.ty == 8 { ("q8_0_8", 4) } else { (["", "q5_k_1", "q5_k_2",
            "q5_k_3", "q5_k_4", "", "", "q5_k_7", "q5_k_8"][nt], 2) };
        let p = [t.dims[0], t.dims[1], t.row(), nt, ys].map(|v| v as u32);
        (k, 0, Box::new(move |e: &Enc| {
            let b = [w, (&self.b, 4 * x), (&self.b, 4 * y)];
            call(e, &self.pipes[k], &p, &b, [t.dims[1] / r, 1, 1], 32)
        }), None)
    }

    fn t(&self, l: usize, n: &str) -> &'static Tensor {
        &self.g.t[format!("blk.{l}.{n}.weight").as_str()]
    }

    fn o(&self, l: usize, n: &str) -> usize {
        self.o[self.t(l, n).name]
    }

    // a dr.metal kernel; x: buffers after b, s, c, id
    fn k<'a>(&'a self, n: &'static str, g: [usize; 3], p: &[usize], x: Vec<B<'a>>) -> Op<'a> {
        let mut a = [0u32; 8];
        p.iter().enumerate().for_each(|(i, &v)| a[i] = v as u32);
        (n, 0, Box::new(move |e: &Enc| {
            let mut b = [&self.b, &self.s, &self.c, &self.id].map(|b| (b, 0)).to_vec();
            b.extend(&x);
            call(e, &self.pipes[n], &a, &b, g, if n == "amax" { 768 } else { 32 })
        }), None)
    }

    // rows of y = W x; input rows are n_in apart, output rows ys (0: n_out)
    fn m<'a>(&'a self, l: usize, n: &str, x: usize, y: usize, ys: usize, rows: usize) -> impl Iterator<Item = Op<'a>> {
        let t = self.t(l, n);
        let (ni, no) = (t.dims[0], t.dims[1]);
        let one = (0..rows % 8).map(move |r| -> Op {
            let (x, y) = (4 * (x + r * ni), 4 * (y + r * ys.max(no)));
            (name(t.ty), 0, Box::new(move |e: &Enc| {
                self.mv.dense(e, t.ty, self.w.w(t), (&self.b, x), (&self.b, y), ni, no)
            }), None)
        });
        one.chain((rows == 8).then(|| self.mm(t, self.w.w(t), x, y, ys.max(no), 8)))
    }

    // context k, v and f_rnn of the token at pos, from row r's features
    fn inject(&self, pos: usize, r: usize) -> Vec<Op<'_>> {
        let nw = [0, 1, 2].map(|l| self.o(l, "dflare_norm"));
        let p = [FR + 4 * r * N, CX, self.o["dspark_fuse.weight"], nw[0], nw[1], nw[2],
            self.o["dspark_norm.weight"]];
        let mut v = vec![self.k("fuse", [72, 4, 1], &p, vec![])];
        for l in 0..3 {
            let (kc, r) = (KC + l * 2048, if l == 1 { pos } else { pos % 1024 });
            v.extend(self.m(l, "attn_k_ctx", CX + l * N, kc, 0, 1));
            v.extend(self.m(l, "attn_v_ctx", CX + l * N, kc + 1024, 0, 1));
            let p = [kc, 2048, self.o(l, "attn_k_ctx_norm"), pos];
            v.push(self.k("rope", [8, 1, 1], &p, vec![]));
            v.push(self.k("put", [64, 1, 1], &[kc, (self.co[l] + r) * 2048], vec![]));
        }
        v
    }

    // the target on toks at m.pos.. (1, 2, 4 or 8 rows): a weight is read
    // once for all rows, the rest runs row by row between the weights.
    // taps each row's features. the greedy id after each row; m.pos stays
    pub fn pass(&self, m: &Model, toks: &[u32]) -> Vec<u32> {
        let (t, rs) = (toks.len(), m.rs());
        let ls: Vec<_> = (0..t).map(|r| m.sched_row(toks[r], m.pos + r, true, r)).collect();
        let mut q: Vec<(&str, Box<dyn Fn(&Enc) + '_>)> = vec![];
        let tap = |r: usize, i: usize, src: usize, vr: usize| -> (&str, Box<dyn Fn(&Enc) + '_>) {
            let p = [src, FR + (4 * r + i) * N, vr].map(|v| v as u32);
            let b = [(&m.a, 4 * r * rs), (&self.b, 0)];
            ("tap", Box::new(move |e: &Enc| call(e, &self.pipes["tap"], &p, &b, [72, 1, 1], 32)))
        };
        let (mut j, mut gu, mut seen) = (0, 0, vec![0; t]);
        while j < ls[0].len() {
            let Some((mode, w, x, y)) = ls[0][j].3.filter(|_| t > 1) else {
                // small ops up to the next weight, row by row
                let n = ls[0][j..].iter().position(|o| o.3.is_some() && t > 1);
                let n = n.unwrap_or(ls[0].len() - j);
                for (r, l) in ls.iter().enumerate() {
                    for o in &l[j..j + n] {
                        q.push((o.0, Box::new(move |e: &Enc| o.2(e))));
                        let i = [5, 23, 41].iter().position(|&v| v == o.1);
                        if let Some(i) = i.filter(|_| o.0 == "elem" && seen[r] < o.1) {
                            seen[r] = o.1;
                            q.push(tap(r, i, TA[0], 0));
                        }
                    }
                }
                j += n;
                continue;
            };
            let (ni, no, pipe) = (w.dims[0], w.dims[1], &self.xp[&(w.ty, t)]);
            if mode == "up" {
                gu = y;
                q.push(("uni", Box::new(move |e: &Enc| {
                    let b = [(&m.id, 0), (&self.ut, 0)];
                    call(e, &self.pipes["uni"], &[t as u32], &b, [1, 1, 1], 1)
                })));
            }
            // experts go to the 8 gate_up rows, down's are then summed
            let (tab, slots, kx, yy) = match mode {
                "up" => (&self.ut, 8 * t, 0, y),
                "down" => (&self.ut, 8 * t, ni, gu),
                _ => (&self.t0, 1, 0, y),
            };
            let p = [ni, no, w.row(), rs, rs, kx].map(|v| v as u32);
            q.push((name(w.ty), Box::new(move |e: &Enc| {
                let b = [m.gpu.w(w), (&m.a, 4 * x), (&m.a, 4 * yy), (tab, 0)];
                call(e, pipe, &p, &b, [no / 2, slots, 1], 32)
            })));
            if mode == "down" {
                let p = [gu, y, rs].map(|v| v as u32);
                q.push(("dsum", Box::new(move |e: &Enc| {
                    call(e, &self.pipes["dsum"], &p, &[(&m.a, 0)], [72, t, 1], 32)
                })));
            }
            j += 1;
        }
        (0..t).for_each(|r| q.push(tap(r, 3, TA[1], TA[2])));
        if std::env::var("PROF").is_ok() {
            // every dispatch in its own command buffer: read as shares
            let mut by = std::collections::BTreeMap::<_, (usize, f64)>::new();
            for o in &q {
                let e = by.entry(o.0).or_default();
                *e = (e.0 + 1, e.1 + m.gpu.run(|e| o.1(e)));
            }
            by.iter().for_each(|(k, (c, t))| eprintln!("{k:8} {c:5} {:7.2} ms", 1e3 * t));
        }
        // several command buffers: the gpu starts while the rest encodes
        objc2::rc::autoreleasepool(|_| {
            let cb = q.chunks(128).map(|c| {
                let cb = m.gpu.q.commandBuffer().unwrap();
                let e = cb.computeCommandEncoder().unwrap();
                c.iter().for_each(|o| o.1(&e));
                e.endEncoding();
                cb.commit();
                cb
            });
            let l = cb.last().unwrap();
            l.waitUntilCompleted();
            assert!(l.error().is_none(), "metal: {:?}", l.error());
        });
        (0..t).map(|r| slice::<u32>(&m.id)[16 * r + 8]).collect()
    }

    // keep rows 0..=k of a pass of t rows: position, rollback of the rest,
    // drafter context
    pub fn commit(&self, m: &mut Model, t: usize, k: usize) {
        let pos = m.pos;
        m.pos += t;
        assert!(m.truncate(pos + k + 1));
        let ops: Vec<Op> = (0..=k).flat_map(|r| self.inject(pos + r, r)).collect();
        m.gpu.run(|e| ops.iter().for_each(|o| o.2(e)));
    }

    // one target token with its context; greedy next id
    pub fn step(&self, m: &mut Model, tok: u32) -> u32 {
        let r = self.pass(m, &[tok])[0];
        self.commit(m, 1, 0);
        r
    }

    // greedy decode from `next` until emit returns false. the block (0,
    // 1, 3 or 7 drafts) is the one with the best expected tokens per
    // second, from running estimates of acceptance and of the costs
    pub fn generate(&self, m: &mut Model, mut next: u32, st: &mut Spec, mut emit: impl FnMut(u32) -> bool) {
        loop {
            let b = st.pick();
            let t0 = Instant::now();
            // without a block, a draft of 3 every 32 tokens keeps score
            let probe = b == 0 && st.n % 32 == 0;
            let ds = if b > 0 || probe { self.draft(m, next, b.max(3)).0 } else { vec![] };
            let t1 = Instant::now();
            st.td[b] += (b > 0) as u8 as f64 * 0.1 * ((t1 - t0).as_secs_f64() - st.td[b]);
            if b == 0 {
                for i in 0..ds.len().max(1) {
                    let t1 = Instant::now();
                    if !emit(next) {
                        return;
                    }
                    next = self.step(m, next);
                    st.tv[1] += 0.1 * (t1.elapsed().as_secs_f64() - st.tv[1]);
                    st.n += 1;
                    if i < ds.len() && !st.see(ds[i] == next) {
                        break;
                    }
                }
                continue;
            }
            let toks: Vec<u32> = [next].into_iter().chain(ds.iter().copied()).collect();
            let r = self.pass(m, &toks);
            let k = (0..b).take_while(|&i| st.see(ds[i] == r[i])).count();
            self.commit(m, b + 1, k);
            st.tv[b + 1] += 0.1 * (t1.elapsed().as_secs_f64() - st.tv[b + 1]);
            (st.n, st.kept[k], next) = (st.n + k + 1, st.kept[k] + 1, r[k]);
            if !toks[..=k].iter().all(|&x| emit(x)) {
                return;
            }
        }
    }

    // nb tokens after anchor a at position m.pos, and the gpu seconds
    pub fn draft(&self, m: &Model, a: u32, nb: usize) -> (Vec<u32>, f64) {
        let n = m.pos;
        slice::<f32>(&self.b)[ST..ST + 256].fill(0.);
        slice::<u32>(&self.id)[0] = a;
        let (emb, out) = (&m.g.t["token_embd.weight"], &m.g.t["output.weight"]);
        let mut v: Vec<Op> = vec![("row", 0, Box::new(move |e: &Enc| {
            self.mv.row(e, emb.ty, m.gpu.w(emb), a as usize, (&self.b, 4 * OO), N)
        }), None)];
        let g8 = [72, 8, 1];
        let p = [OO, X, self.o["token_embd_norm.weight"], self.o["mask_embd.weight"]];
        v.push(self.k("norm", g8, &p, vec![]));
        for l in 0..3 {
            v.push(self.k("norm", g8, &[X, HN, self.o(l, "attn_norm")], vec![]));
            v.extend(self.m(l, "attn_q", HN, QQ, 0, 8));
            v.extend(self.m(l, "attn_k", HN, KK, 2048, 8));
            v.extend(self.m(l, "attn_v", HN, KK + 1024, 2048, 8));
            v.push(self.k("rope", [24, 8, 1], &[QQ, 3072, self.o(l, "attn_q_norm"), n], vec![]));
            v.push(self.k("rope", [8, 8, 1], &[KK, 2048, self.o(l, "attn_k_norm"), n], vec![]));
            let (nk, ring) = if l == 1 { (n, 0) } else { (n.min(1024), 1024) };
            let p = [QQ, KK, self.co[l] * 2048, nk, n - nk, ring, AT];
            v.push(self.k("attn", [24, 8, 1], &p, vec![]));
            v.extend(self.m(l, "attn_output", AT, OO, 0, 8));
            v.push(self.k("add", g8, &[X, OO], vec![]));
            v.push(self.k("norm", g8, &[X, HN, self.o(l, "ffn_norm")], vec![]));
            v.extend(self.m(l, "ffn_gate", HN, GG, 0, 8));
            v.extend(self.m(l, "ffn_up", HN, UU, 0, 8));
            v.push(self.k("silu", [288, 8, 1], &[GG, UU], vec![]));
            v.extend(self.m(l, "ffn_down", GG, OO, 0, 8));
            v.push(self.k("add", g8, &[X, OO], vec![]));
        }
        v.push(self.k("norm", g8, &[X, HN, self.o["output_norm.weight"]], vec![]));
        v.push(self.mm(out, m.gpu.w(out), HN + N, BL, NV, nb));
        let w = |n: &str| self.w.w(&self.g.t[n]);
        for i in 1..=nb {
            let p = [ST, HN + i * N, CX + 3 * N, JJ, self.o["joint_proj.bias"], i - 1];
            let x = vec![w("joint_proj.weight"), w("markov_w1.weight")];
            v.push(self.k("joint", [24, 1, 1], &p, x));
            let p = [ST, JJ, BL + (i - 1) * NV, LG, WB];
            v.push(self.k("pl", [6252, 1, 1], &p, vec![(&self.p, 0)]));
            v.push(self.k("amax", [1, 1, 1], &[LG, 200058, i], vec![]));
        }
        if std::env::var("PROF").is_ok() {
            // every dispatch in its own command buffer: read as shares
            let mut by = std::collections::BTreeMap::<_, (usize, f64)>::new();
            for o in &v {
                let e = by.entry(o.0).or_default();
                *e = (e.0 + 1, e.1 + m.gpu.run(|e| o.2(e)));
            }
            by.iter().for_each(|(k, (c, t))| eprintln!("{k:8} {c:4} {:7.2} ms", 1e3 * t));
        }
        let t = m.gpu.run(|e| v.iter().for_each(|o| o.2(e)));
        (slice::<u32>(&self.id)[1..=nb].to_vec(), t)
    }
}

// running estimates for generate(): acceptance of a draft token, seconds
// of a pass by rows and of a draft by length (measured here, m2 air)
pub struct Spec {
    pub a: f64,
    pub n: usize,
    pub tv: [f64; 9],
    pub td: [f64; 8],
    pub kept: [usize; 8],
}

impl Spec {
    pub fn new() -> Spec {
        let (mut tv, mut td) = ([0.; 9], [0.; 8]);
        (tv[1], tv[2], tv[4], tv[8]) = (0.052, 0.27, 0.25, 0.74);
        (td[1], td[3], td[7]) = (0.015, 0.022, 0.031);
        Spec { a: 0.6, n: 0, tv, td, kept: [0; 8] }
    }

    fn see(&mut self, hit: bool) -> bool {
        self.a += 0.1 * (hit as u8 as f64 - self.a);
        hit
    }

    // tokens per second of a block: 1 + a + a^2 + .. over draft + pass
    fn pick(&self) -> usize {
        let r = |b: usize| {
            (0..=b).map(|i| self.a.powi(i as i32)).sum::<f64>() / (self.td[b] + self.tv[b + 1])
        };
        let b = std::env::var("BLOCK").ok().and_then(|v| v.parse().ok());
        b.unwrap_or([0, 1, 3, 7].into_iter().max_by(|&x, &y| r(x).total_cmp(&r(y))).unwrap())
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
                let x = &slice::<f32>(&dr.b)[FR + j * N..][..N];
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

// greedy decode with drafts; BLOCK=b fixes the block (0: none)
pub fn run(a: &[String]) {
    let opt = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].as_str());
    let (n, ctx) = (opt("-n").map_or(64, |v| v.parse().unwrap()), opt("-c").map_or(4096, |v| v.parse().unwrap()));
    let mut m = Model::open(&a[0], ctx);
    m.reset();
    let dr = Dr::open(&m, opt("--draft").unwrap(), ctx);
    let mut next = 0;
    for t in crate::ids(opt("--tokens").expect("--tokens FILE")) {
        next = dr.step(&mut m, t);
    }
    let (mut st, mut out, t0) = (Spec::new(), 0, Instant::now());
    dr.generate(&mut m, next, &mut st, |t| {
        print!("{t} ");
        out += 1;
        out < n
    });
    println!();
    let ms = |v: &[f64]| v.iter().map(|x| (1e3 * x).round()).collect::<Vec<_>>();
    eprintln!(
        "{:.1} t/s, kept {:?}, acceptance {:.2}, pass ms {:?}, draft ms {:?}",
        out as f64 / t0.elapsed().as_secs_f64(), st.kept, st.a, ms(&st.tv), ms(&st.td)
    );
}
