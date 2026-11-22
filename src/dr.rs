// dspark block drafter on the target's hidden states (DRAFT.md, dr.metal).
//   let d = Dr::open(&m, path, n_ctx);
//   let next = d.step(&mut m, tok);     // target step + context inject
//   let ds = d.draft(&m, anchor, 7); // (token, probability) after anchor
//   let next = d.pass(&m, &[anchor, ids..]); d.commit(&mut m, rows, kept);
//   let (done, next) = d.round(&mut m, next, &mut spec);  // all of it
//   d.generate(&mut m, next, &mut Spec::new(), |tok| more);  // in a loop
// gall dr MODEL DRAFTER REFDIR [-n 64] [-c 4096] [--ub] [--tv 2,4,8]:
// teacher-forced acceptance; --tv: ms of a pass by rows
// gall run MODEL --tokens FILE --draft DRAFTER [--block B] [-n 64] [-c 4096]
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
const KINDS: [&str; 16] = [
    "tap", "fuse", "norm", "rope", "put", "attn", "silu", "add", "joint", "pl",
    "pset", "amax", "uni", "dsum", "conf", "amrg",
];
// drm.metal x_*: (type, weight rows per threadgroup at 1 use), as XS there
const XT: [(u32, usize); 10] = [(8, 4), (12, 2), (13, 2), (14, 2), (16, 8),
    (17, 4), (18, 8), (20, 4), (22, 8), (40, 8)];
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
const CF: usize = FR + 32 * N; // the drafter's probability of each draft
const PA: usize = CF + 8; // attention partials: 192 queries x chunks x 130

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
    // slab kernels by type and uses: weight rows per threadgroup, name
    xp: HashMap<(u32, usize), (Pipe, usize, &'static str)>,
    ut: Buf, // expert slabs of a pass by use count; ui: their grids
    ui: Buf,
    t0: Buf, // the same for a dense matrix: slab 0, used by every row
    o: HashMap<&'static str, usize>, // f32 tensors in s
    co: [usize; 3],                  // cache rows before each layer
    // positions with context: the last run of inject() in a row. what a
    // snapshot restored or the model stepped alone is not in it
    span: std::cell::Cell<(usize, usize)>,
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
        for k in ["q8_0_8", "q5_k_1", "q5_k_2", "q5_k_3", "q5_k_4", "q5_k_5", "q5_k_6",
            "q5_k_7", "q5_k_8"] {
            pipes.insert(k, m.gpu.pipe(MM, "", &format!("mm_{k}")));
        }
        let xp = XT.iter().flat_map(|&t| (1..9).map(move |c| (t, c))).map(|((t, r), c)| {
            let r = match c {
                1 => r,
                2 => 4,
                3..=6 if t == 8 => 4,
                _ => 2,
            };
            let n = format!("x_{}_{c}", name(t));
            ((t, c), (m.gpu.pipe(MM, "", &n), r, &*n.leak()))
        });
        let t0 = m.gpu.buf(4 * 9 * 65 * 8);
        for c in 0..8 {
            let e = &mut slice::<u32>(&t0)[9 * 65 * c..][..9];
            (0..9).for_each(|i| e[i] = 8 * i.saturating_sub(1) as u32);
        }
        let d = Dr {
            xp: xp.collect(),
            ut: m.gpu.buf(4 * 9 * 65 * 8),
            ui: m.gpu.buf(4 * 9 * 8),
            t0,
            mv: Mv::new(&m.gpu, ""),
            pipes,
            s: sb,
            b: m.gpu.buf(4 * (PA + 192 * 130 * (n_ctx + 8).div_ceil(256))),
            p: m.gpu.buf(2 * 256 * NV),
            c: m.gpu.buf(2 * 2048 * (2048 + n_ctx)),
            id: m.gpu.buf(4 * 8),
            co: [0, 1024, 1024 + n_ctx],
            span: Default::default(),
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
            "q5_k_3", "q5_k_4", "q5_k_5", "q5_k_6", "q5_k_7", "q5_k_8"][nt], 2) };
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
            let th = if g[0] == 1 { 768 } else if n == "attn" { 512 } else { 32 };
            call(e, &self.pipes[n], &a, &b, g, th)
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
        let (lo, hi) = self.span.get();
        self.span.set((if pos < lo || pos > hi { pos } else { lo }, pos + 1));
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

    // the target on toks at m.pos.. (1 to 8 rows): a weight is read once
    // for all rows, an expert slab once for the rows that use it, a small
    // op is one dispatch for all rows. taps each row's features. the
    // greedy id after each row; m.pos stays
    pub fn pass(&self, m: &Model, toks: &[u32]) -> Vec<u32> {
        let (t, rs) = (toks.len(), m.rs());
        // row 0 is kept whatever the others do: its context goes along
        let (ops, inj) = (m.sched_z(toks, m.pos, true, 0), self.inject(m.pos, 0));
        let mut q: Vec<(&str, Box<dyn Fn(&Enc) + '_>)> = vec![];
        let tap = |r: usize, i: usize, src: usize, vr: usize| -> (&str, Box<dyn Fn(&Enc) + '_>) {
            let p = [src, FR + (4 * r + i) * N, vr].map(|v| v as u32);
            let b = [(&m.a, 4 * r * rs), (&self.b, 0)];
            ("tap", Box::new(move |e: &Enc| call(e, &self.pipes["tap"], &p, &b, [72, 1, 1], 32)))
        };
        let (mut gu, mut seen) = (0, 0);
        for o in &ops {
            let Some((mode, w, x, y)) = o.3.filter(|_| t > 1) else {
                q.push((o.0, Box::new(move |e: &Enc| o.2(e))));
                let i = [5, 23, 41].iter().position(|&v| v == o.1);
                if let Some(i) = i.filter(|_| o.0 == "elem" && seen < o.1) {
                    seen = o.1;
                    (0..t).for_each(|r| q.push(tap(r, i, TA[0], 0)));
                }
                continue;
            };
            let (ni, no) = (w.dims[0], w.dims[1]);
            if mode == "up" {
                gu = y;
                q.push(("uni", Box::new(move |e: &Enc| {
                    let b = [(&m.id, 0), (&self.ut, 0), (&self.ui, 0)];
                    call(e, &self.pipes["uni"], &[t as u32], &b, [1, 1, 1], 64)
                })));
            }
            // experts go to the 8 gate_up rows, down's are then summed.
            // a slab used by c rows goes to the kernel of c uses
            let (c0, kx, yy) = match mode {
                "up" => (1, 0, y),
                "down" => (1, ni, gu),
                _ => (t, 0, y),
            };
            let p = [ni, no, w.row(), rs, rs, kx].map(|v| v as u32);
            for c in c0..=t {
                let (pipe, r, n) = &self.xp[&(w.ty, c)];
                q.push((n, Box::new(move |e: &Enc| {
                    let b = |t| [m.gpu.w(w), (&m.a, 4 * x), (&m.a, 4 * yy), (t, 0)];
                    if mode == "mv" {
                        return call(e, pipe, &p, &b(&self.t0), [no / r, 1, 1], 32);
                    }
                    let i = 12 * (3 * (c - 1) + r.trailing_zeros() as usize - 1);
                    calli(e, pipe, &p, &b(&self.ut), (&self.ui, i), 32)
                })));
            }
            if mode == "down" {
                let p = [gu, y, rs].map(|v| v as u32);
                q.push(("dsum", Box::new(move |e: &Enc| {
                    call(e, &self.pipes["dsum"], &p, &[(&m.a, 0)], [72, t, 1], 32)
                })));
            }
        }
        (0..t).for_each(|r| q.push(tap(r, 3, TA[1], TA[2])));
        inj.iter().for_each(|o| q.push((o.0, Box::new(move |e: &Enc| o.2(e)))));
        if std::env::var("PROF").is_ok_and(|_| t > 1) {
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
    // drafter context of the rows after the first
    pub fn commit(&self, m: &mut Model, t: usize, k: usize) {
        let pos = m.pos;
        m.pos += t;
        assert!(m.truncate(pos + k + 1));
        m.top = pos + t;
        let ops: Vec<Op> = (1..=k).flat_map(|r| self.inject(pos + r, r)).collect();
        if k > 0 {
            m.gpu.run(|e| ops.iter().for_each(|o| o.2(e)));
        }
    }

    // a prompt: batched prefill with the context of each token, the tail
    // by step(). greedy next id
    pub fn feed(&self, m: &mut Model, toks: &[u32]) -> u32 {
        let rest = m.prefill_tap(toks, 512, Some(&mut |m: &Model, ft: &[f32]| {
            for (i, f) in ft.chunks(32 * N).enumerate() {
                slice::<f32>(&self.b)[FR..][..f.len()].copy_from_slice(f);
                let n = (0..f.len() / (4 * N)).flat_map(|r| self.inject(m.pos + 8 * i + r, r));
                let ops: Vec<Op> = n.collect();
                m.gpu.run(|e| ops.iter().for_each(|o| o.2(e)));
            }
        }));
        rest.iter().fold(0, |_, &t| self.step(m, t))
    }

    // one target token with its context; greedy next id
    pub fn step(&self, m: &mut Model, tok: u32) -> u32 {
        let r = self.pass(m, &[tok])[0];
        self.commit(m, 1, 0);
        r
    }

    // one round of greedy decode from the pending token `next`: the tokens
    // it put into the model (next first) and the new pending token. with
    // drafting on: a draft, then a pass over the anchor and the drafts
    // that st.plan() takes; the drafts after those stay for the next
    // rounds while they hold. else serial steps, and a draft now and then
    // to keep score
    pub fn round(&self, m: &mut Model, mut next: u32, st: &mut Spec) -> (Vec<u32>, u32) {
        let (t0, on) = (Instant::now(), st.block.map_or(st.on(), |b| b > 0));
        let ema = |x: &mut f64, t: Instant| *x += 0.1 * (t.elapsed().as_secs_f64() - *x);
        let mut d = std::mem::take(&mut st.left);
        if st.at != (m.pos, next) {
            d.clear();
        }
        let nd = st.block.unwrap_or(if on || st.n >= st.due[0] { st.nb() } else { 0 });
        if d.is_empty() && nd > 0 {
            d = self.draft(m, next, nd);
            ema(&mut st.td[nd], t0);
            st.fresh(&d, on);
        }
        let (t1, mut j, mut k) = (Instant::now(), st.plan(&d).0, 0);
        j *= on as usize;
        let mut done = vec![next];
        if j == 0 {
            next = self.step(m, next);
        } else {
            done.extend(d[..j].iter().map(|x| x.0));
            let r = self.pass(m, &done);
            k = (0..j).take_while(|&i| st.see(d[i].1, d[i].0 == r[i])).count();
            self.commit(m, j + 1, k);
            done.truncate(k + 1);
            (st.kept[k], next) = (st.kept[k] + 1, r[k]);
        }
        ema(&mut st.tv[j + 1], t1);
        // the draft after the last one verified is scored by the new
        // pending token; if it holds, so does the rest of the draft
        if k == j && j < d.len() && st.see(d[j].1, d[j].0 == next) {
            (st.left, st.at) = (d.split_off(j + 1), (m.pos, next));
        }
        st.n += done.len();
        (done, next)
    }

    // greedy decode from `next` until emit returns false
    pub fn generate(&self, m: &mut Model, mut next: u32, st: &mut Spec, mut emit: impl FnMut(u32) -> bool) {
        loop {
            let (done, n) = self.round(m, next, st);
            next = n;
            if !done.iter().all(|&x| emit(x)) {
                return;
            }
        }
    }

    // nb tokens after anchor a at position m.pos, each with the drafter's
    // probability of it
    pub fn draft(&self, m: &Model, a: u32, nb: usize) -> Vec<(u32, f32)> {
        let (n, (lo, hi)) = (m.pos, self.span.get());
        let lo = if hi < n { n } else { lo.min(n) };
        slice::<f32>(&self.b)[ST..ST + 256].fill(0.);
        slice::<u32>(&self.id)[0] = a;
        let (emb, out) = (&m.g.t["token_embd.weight"], &m.g.t["output.weight"]);
        let cap = (m.g.f("inkling2.final_logit_softcapping") as f32).to_bits() as usize;
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
            let (nk, ring) = if l == 1 { (n - lo, 0) } else { ((n - lo).min(1024), 1024) };
            let nc = (nk + 8).div_ceil(256);
            let p = [QQ, KK, self.co[l] * 2048, nk, n - nk, ring, PA, nc];
            v.push(self.k("attn", [8, 8, nc], &p, vec![]));
            v.push(self.k("amrg", [24, 8, 1], &[PA, nc, AT], vec![]));
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
            v.push(self.k("conf", [1, 1, 1], &[LG, 200058, i, CF + i, cap], vec![]));
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
        m.gpu.run(|e| v.iter().for_each(|o| o.2(e)));
        let (id, p) = (slice::<u32>(&self.id), slice::<f32>(&self.b));
        (1..=nb).map(|i| (id[i], p[CF + i])).collect()
    }
}

// running estimates for round(), seconds as measured here (m2 air)
pub struct Spec {
    pub block: Option<usize>, // a fixed count of drafts per round; 0: none
    pub q: [f64; 12],         // acceptance of a draft by its probability
    pub e: [f64; 2],          // tokens and seconds to expect of a draft
    pub tv: [f64; 9],         // a pass by rows
    pub td: [f64; 8],         // a draft by length
    pub kept: [usize; 8],     // passes by drafts kept
    pub drafts: usize,
    pub n: usize,
    left: Vec<(u32, f32)>, // drafts not yet used, after the pending
    at: (usize, u32),      // token `at.1` at position `at.0`
    due: [usize; 2],       // off: the next draft is at n = due[0]; the gap
    hits: [f64; 2],        // of this draft, and their mean
}

impl Spec {
    pub fn new() -> Spec {
        let ms = |v: [f64; 8]| v.map(|x| 1e-3 * x);
        let tv = [51., 73., 96., 123., 159., 197., 255., 266.];
        Spec {
            block: None,
            q: std::array::from_fn(|i| 0.9 / (1. + (1.75 - i as f64 / 2.).exp())),
            e: [1., 0.055],
            tv: std::array::from_fn(|i| 1e-3 * tv[i.max(1) - 1]),
            td: ms([0., 15., 18., 21., 24., 26., 28., 31.]),
            kept: [0; 8],
            drafts: 0,
            n: 0,
            left: vec![],
            at: (0, 0),
            due: [0; 2],
            hits: [0., 2.],
        }
    }

    // a probability's place in q: half steps of its logit, 0.12 to 0.97
    fn bin(p: f32) -> usize {
        (4. + 2. * (p / (1. - p)).ln()).clamp(0., 11.) as usize
    }

    fn see(&mut self, p: f32, hit: bool) -> bool {
        let q = &mut self.q[Spec::bin(p)];
        *q += 0.05 * (hit as u8 as f64 - *q);
        self.hits[0] += hit as u8 as f64;
        hit
    }

    // drafting beats serial decode
    pub fn on(&self) -> bool {
        self.e[0] * self.tv[1] > self.e[1]
    }

    // drafts to ask for: 2 more than held of late
    fn nb(&self) -> usize {
        (self.hits[1] + 2.5).clamp(3., 7.) as usize
    }

    // the drafts to verify, and the tokens to expect of the round: a draft
    // is taken while its chance to be kept pays for its row at the rate
    // in hand
    fn plan(&self, d: &[(u32, f32)]) -> (usize, f64) {
        let (mut p, mut e) = (1., 1.);
        let rate = (self.e[0] / self.e[1]).max(1. / self.tv[1]);
        for (j, x) in d.iter().enumerate() {
            p *= self.q[Spec::bin(x.1)];
            let dt = (self.tv[j + 2] - self.tv[j + 1]).max(0.3 * self.tv[1]);
            if self.block.is_none() && p < rate * dt {
                return (j, e);
            }
            e += p;
        }
        (d.len(), e)
    }

    // a new draft: the tokens and seconds to expect of it over its
    // rounds, each reached if the drafts before it hold, go into e. off,
    // the next draft is 8 tokens on if this one would have paid, else 32,
    // 64, 128
    fn fresh(&mut self, d: &[(u32, f32)], on: bool) {
        let (mut i, mut s, mut x) = (0, 1., [0., self.td[d.len()]]);
        while i == 0 || i < d.len() {
            let (j, e) = self.plan(&d[i..]);
            x = [x[0] + s * e, x[1] + s * self.tv[j + 1]];
            let held = &d[i..d.len().min(i + j + 1)];
            s *= held.iter().map(|v| self.q[Spec::bin(v.1)]).product::<f64>();
            i += j + 1;
        }
        let w = if on { 0.1 } else { 0.3 };
        (0..2).for_each(|i| self.e[i] += w * (x[i] - self.e[i]));
        self.hits = [0., self.hits[1] + 0.2 * (self.hits[0] - self.hits[1])];
        let gap = if x[0] * self.tv[1] > x[1] { 8 } else { (2 * self.due[1]).clamp(32, 128) };
        self.due = [self.n + gap, gap];
        self.drafts += 1;
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
    let (mut run, mut hit, mut td) = ([0usize; 8], [0usize; 7], 0.);
    let mut cal = [[0usize; 2]; 10]; // by probability: drafts, hits
    if let Some(i) = a.iter().position(|x| x == "--tv") {
        // ms of a pass by rows (best, mean of 6), on the reference text
        dr.feed(&mut m, &all[..np]);
        let mut i0 = np;
        for r in a[i + 1].split(',').map(|v| v.parse::<usize>().unwrap()) {
            let ts: Vec<f64> = (0..6).map(|_| {
                let t0 = Instant::now();
                dr.pass(&m, &all[i0..i0 + r]);
                dr.commit(&mut m, r, r - 1);
                i0 += r;
                1e3 * t0.elapsed().as_secs_f64()
            }).collect();
            let b = ts.iter().fold(1e9f64, |a, &b| a.min(b));
            println!("t({r}) {b:.1} ms best, {:.1} mean", ts.iter().sum::<f64>() / 6.);
        }
        return;
    }
    // --ub: the prompt by feed() (batched), not step by step
    let ub = a.iter().any(|x| x == "--ub");
    if ub {
        dr.feed(&mut m, &all[..np - 1]);
    }
    for i in if ub { np - 1 } else { 0 }..np + n {
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
        let t0 = Instant::now();
        let ds = dr.draft(&m, all[i + 1], 7);
        td += t0.elapsed().as_secs_f64();
        let ok: Vec<bool> = (0..7).map(|k| ds[k].0 == all[i + 2 + k]).collect();
        let r = ok.iter().take_while(|&&x| x).count();
        // a draft counts while those before it hold
        for k in 0..(r + 1).min(7) {
            let c = &mut cal[(ds[k].1 * 10.).min(9.) as usize];
            *c = [c[0] + 1, c[1] + ok[k] as usize];
        }
        run[r] += 1;
        (0..7).for_each(|k| hit[k] += ok[k] as usize);
    }
    let c = (n + 1) as f64;
    println!("draft of 7 {:.1} ms", 1e3 * td / c);
    println!("slot hit {:.2?}", hit.map(|h| h as f64 / c));
    println!("drafts, hits by probability in tenths: {cal:?}");
    for b in 1..8 {
        let e: usize = run.iter().enumerate().map(|(k, r)| r * (1 + k.min(b))).sum();
        println!("block {b}: {:.2} tokens per pass", e as f64 / c);
    }
}

// greedy decode with drafts; --block B fixes the drafts per round (0: none)
pub fn run(a: &[String]) {
    let opt = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].as_str());
    let (n, ctx) = (opt("-n").map_or(64, |v| v.parse().unwrap()), opt("-c").map_or(4096, |v| v.parse().unwrap()));
    let mut m = Model::open(&a[0], ctx);
    m.reset();
    let dr = Dr::open(&m, opt("--draft").unwrap(), ctx);
    let next = dr.feed(&mut m, &crate::ids(opt("--tokens").expect("--tokens FILE")));
    let (mut st, mut out, t0) = (Spec::new(), 0, Instant::now());
    st.block = opt("--block").map(|v| v.parse().unwrap());
    dr.generate(&mut m, next, &mut st, |t| {
        print!("{t} ");
        out += 1;
        out < n
    });
    println!();
    let ms = |v: &[f64]| v.iter().map(|x| (1e3 * x).round()).collect::<Vec<_>>();
    eprintln!(
        "{:.1} t/s, kept {:?}, {} drafts, on {}, a draft {:.2} tokens {:.0} ms, \
         pass ms {:?}, draft ms {:?}, acceptance {:.2?}",
        out as f64 / t0.elapsed().as_secs_f64(), st.kept, st.drafts, st.on(), st.e[0],
        1e3 * st.e[1],
        ms(&st.tv[1..]), ms(&st.td[1..]), st.q
    );
}
