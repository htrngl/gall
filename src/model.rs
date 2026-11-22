// inkling-2-mini decode, one token per command buffer (ops.metal, mv.rs).
//   let mut m = Model::open(path, n_ctx);
//   let next = m.step(token);     // greedy next id; m.logits() for all
//   m.reset();
use crate::gguf::{self, Gguf, Tensor};
use crate::gpu::*;
use crate::mv::Mv;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use std::cell::Cell;
use std::collections::HashMap;

#[path = "pf.rs"]
pub mod pf;
#[path = "snap.rs"]
pub mod snap;

const N: usize = 2304;
const OPS: &str = include_str!("ops.metal");
pub const KINDS: [&str; 19] = [
    "dots", "elem", "low", "ctl", "rows", "post", "score", "smax", "gather",
    "merge", "top8", "act", "msg", "amax", "best", "pack", "scorew", "gatherw",
    "fix",
];
// regions of the activation buffer, in floats. 0 means "none" in params.
const PRE: usize = N; // prelude output
const H: usize = 2 * N; // residual stream
const X: usize = 3 * N; // q, k, v inputs
const XF: usize = 6 * N; // ffn input
const F: usize = 7 * N; // ffn_inp
const AO: usize = 8 * N; // attention output after wo
const D: usize = 9 * N; // expert output
const BK: usize = 10 * N; // normalized block stack, 5 deep
const VR: usize = 15 * N; // mixed inputs before the norm, 3
const TMP: usize = 18 * N; // control inputs, 3 x 2N
const Q: usize = 24 * N; // q, gate, attention out: 3072 each
const G: usize = Q + 3072;
const O: usize = G + 3072;
const K: usize = O + 3072; // k then v
const R: usize = K + 2048;
const SK: usize = R + 384; // sinks
const DK: usize = SK + 24; // dynamic conv taps, k then v
const GC: usize = DK + 64; // control low rank, 3 x 130
const SC: usize = GC + 390; // mix scalars, 3 x 8
const IS: usize = SC + 24; // rms scales: q input, ffn input
const LGT: usize = IS + 2; // router logits, then the ffn input v . v
const WT: usize = LGT + 132; // router weights, before the norm
const GU: usize = WT + 8; // 8 expert gate_up rows
const HH: usize = GU + 8 * N;
const MSG: usize = HH + 4 * N; // 12 layers x 4 messages
const LG: usize = MSG + 48 * N; // logits
const PT: usize = LG + 201024; // attention scratch, to the end: scores
// (24 per key), their maxima per 8 keys, the 24 maxima, partials per 256
const CH: usize = 6656; // conv channels per slot
const W8: usize = 1024; // full attention reads the int8 copy from this depth
const CS: usize = 16 * CH; // conv state floats per slot, a ring of 16
const SW: usize = 2048; // sliding window ring

type T = &'static Tensor;

struct L {
    swa: bool,
    q: T,
    k: T,
    v: T,
    o: T,
    gate: T,
    rel: T,
    gu: T,
    dn: T,
    nq: u32, // attention mix queries
    aq: u32,
    nw: [u32; 3],
    sink: u32,
    qc: u32,
    rc: u32,
    rb: u32,
    kc: u32,
    vc: u32,
    kd: u32,
    vd: u32,
    apc: u32,
    fpc: u32,
    fq: u32,
    fnw: u32,
    gi: u32,
    pb: u32,
    us: u32,
    ds: u32,
    ctl: u32,
}

// (kind, visit, encode, a weight op's (mode, tensor, x, y) in floats)
pub type Op<'a> =
    (&'static str, usize, Box<dyn Fn(&Enc) + 'a>, Option<(&'static str, T, usize, usize)>);
pub const ROWS: usize = 8; // activation rows, for dr.rs's batched verify

pub struct Model {
    pub g: &'static Gguf,
    pub gpu: Gpu,
    mv: Mv,
    pipes: HashMap<&'static str, Pipe>,
    pub a: Buf,
    s: Buf,
    sh: Buf,
    kv: Buf,
    cs: Buf,
    pub id: Buf,
    l: Vec<L>,
    emb: T,
    out: T,
    tn: u32,
    oq: u32,
    on: u32,
    mq: u32,
    kvo: Vec<usize>, // kv slot offsets, in halfs
    hp: [f32; 4],    // qk clip, rel clip, log floor, log alpha
    n_ctx: usize,
    k8: Buf, // int8 copy of the full-attention slots, 2304 bytes per key
    hi: Cell<usize>, // its rows [0, hi) are current
    // from W8 deep, full attention reads the copy, and f16 for the keys
    // within vt of the greatest weight. 0: f16 only
    pub vt: f32,
    pub pos: usize,
    pub t: f64, // gpu seconds of the last step
    ck: Option<(usize, Vec<f32>, Vec<u16>)>, // checkpoint: pos, conv, swa
    pf: Option<pf::Pf>,
}

// f32 to f16, round to nearest; tiny values to 0
fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    let (s, e, m) = (b >> 16 & 0x8000, (b >> 23 & 255) as i32 - 112, b & 0x7fffff);
    match e {
        ..=0 => s as u16,
        31.. => (s | 0x7c00) as u16,
        _ => (s + ((e as u32) << 10) + (m >> 13) + (m >> 12 & 1)) as u16,
    }
}

// (physical layer, cache slot, pass) in execution order
pub fn visits() -> Vec<(usize, usize, usize)> {
    let core = |p| (6..18).map(move |i| (i, if p > 0 { 12 * p + 6 + i } else { i }, p));
    let lin = |r: std::ops::Range<usize>| r.map(|i| (i, i, 0));
    lin(0..6).chain((0..3).flat_map(core)).chain(lin(18..24)).collect()
}

impl Model {
    pub fn open(path: &str, n_ctx: usize) -> Model {
        // one map per process, kept across unloads (one model per process)
        static G: std::sync::OnceLock<Gguf> = std::sync::OnceLock::new();
        let g: &'static Gguf = G.get_or_init(|| gguf::open(path));
        let mut gpu = Gpu::new();
        gpu.map(g);
        // small tensors, as f32 in one buffer; offset 0 is 128 ones
        let mut s = vec![1f32; 128];
        let mut sm = |n: String| -> u32 {
            let Some(t) = g.t.get(n.as_str()) else { return 0 };
            let (o, d) = (s.len() as u32, g.data(t));
            match t.ty {
                0 => s.extend(d.chunks(4).map(|b| {
                    f32::from_le_bytes(b.try_into().unwrap())
                })),
                30 => s.extend(d.chunks(2).map(|b| {
                    f32::from_bits((b[0] as u32) << 16 | (b[1] as u32) << 24)
                })),
                _ => panic!("{n}: not a small tensor"),
            }
            o
        };
        // the larger ones as f16, offsets in halfs
        let mut sh = vec![0u16; 4];
        let mut hm = |n: String| -> u32 {
            let Some(t) = g.t.get(n.as_str()) else { return 0 };
            let (o, d) = (sh.len() as u32, g.data(t));
            match t.ty {
                0 => sh.extend(d.chunks(4).map(|b| {
                    f16(f32::from_le_bytes(b.try_into().unwrap()))
                })),
                30 => sh.extend(d.chunks(2).map(|b| {
                    f16(f32::from_bits((b[0] as u32) << 16 | (b[1] as u32) << 24))
                })),
                _ => panic!("{n}: not a small tensor"),
            }
            o
        };
        let l = (0..24)
            .map(|i| {
                let t = |n: &str| &g.t[format!("blk.{i}.{n}.weight").as_str()];
                let mut w = |n: &str| sm(format!("blk.{i}.{n}"));
                let mut wh = |n: &str| hm(format!("blk.{i}.{n}"));
                let swa = i % 6 != 5;
                let aq = w("attn_res_query.weight");
                let nw = if swa {
                    [w("attn_norm.weight"); 3]
                } else {
                    ["q", "k", "v"].map(|c| w(&format!("attn_{c}_inp_norm.weight")))
                };
                let ctl = ["attn", "k", "v", "mlp"]
                    .map(|c| {
                        ["down", "gain", "forget"]
                            .map(|k| wh(&format!("loop_{c}_{k}.weight")))[0]
                    })[0];
                L {
                    swa,
                    q: t("attn_q"),
                    k: t("attn_k"),
                    v: t("attn_v"),
                    o: t("attn_output"),
                    gate: t("attn_gate"),
                    rel: t("attn_rel"),
                    gu: t("ffn_gate_up_exps"),
                    dn: t("ffn_down_exps"),
                    nq: if aq == 0 { 0 } else if swa { 1 } else { 3 },
                    aq,
                    nw,
                    ctl,
                    sink: wh("attn_sink_proj.weight"),
                    qc: w("attn_q_clip.weight"),
                    rc: w("attn_rel_clip.weight"),
                    rb: w("attn_rel_bias.weight"),
                    kc: w("attn_k_conv.weight"),
                    vc: w("attn_v_conv.weight"),
                    kd: wh("attn_k_conv_proj.weight"),
                    vd: wh("attn_v_conv_proj.weight"),
                    apc: w("attn_post_conv.weight"),
                    fpc: w("ffn_post_conv.weight"),
                    fq: w("ffn_res_query.weight"),
                    fnw: w("ffn_norm.weight"),
                    gi: wh("ffn_gate_inp.weight"),
                    pb: w("exp_probs_b.bias"),
                    us: w("ffn_gate_up_exps.scale"),
                    ds: w("ffn_down_exps.scale"),
                }
            })
            .collect();
        let tn = sm("token_embd_norm.weight".into());
        let oq = sm("output_res_query.weight".into());
        let on = sm("output_norm.weight".into());
        // message queries of core layer i: res[2i], kv[2i], kv[2i+1], res[2i+1]
        let (res, kvq) = (
            sm("loop_res_query.weight".into()) as usize,
            sm("loop_kv_res_query.weight".into()) as usize,
        );
        let mq = s.len() as u32;
        for i in 0..12 {
            for o in [res + 2 * i * N, kvq + 2 * i * N, kvq + (2 * i + 1) * N, res + (2 * i + 1) * N] {
                s.extend_from_within(o..o + N);
            }
        }
        let (sb, hb) = (gpu.buf(4 * s.len()), gpu.buf(2 * sh.len()));
        slice::<f32>(&sb)[..s.len()].copy_from_slice(&s);
        slice::<u16>(&hb)[..sh.len()].copy_from_slice(&sh);
        let mut kvo = vec![0];
        let mut by_slot = visits();
        by_slot.sort_by_key(|v| v.1);
        for (i, _, _) in by_slot {
            let n = if i % 6 != 5 { SW * 2048 } else { n_ctx * 2048 };
            kvo.push(kvo.last().unwrap() + n);
        }
        let nk = n_ctx.max(1024) + 256;
        let pipes = KINDS.map(|k| (k, gpu.pipe(OPS, "", k))).into();
        Model {
            mv: Mv::new(&gpu, ""),
            pipes,
            a: gpu.buf(4 * ROWS * (PT + nk * 27 + 24 + (nk.div_ceil(256).max(65) + 64) * 24 * 132)),
            s: sb,
            sh: hb,
            kv: gpu.buf(2 * kvo[48]),
            cs: gpu.buf(4 * 48 * CS),
            id: gpu.buf(4 * 16 * ROWS),
            l,
            emb: &g.t["token_embd.weight"],
            out: &g.t["output.weight"],
            tn,
            oq,
            on,
            mq,
            kvo,
            hp: [
                g.f("inkling2.attention.qk_clip") as f32,
                g.f("inkling2.attention.relative_clip") as f32,
                g.f("inkling2.attention.log_scale_floor") as f32,
                g.f("inkling2.attention.log_scale_alpha") as f32,
            ],
            n_ctx,
            k8: gpu.buf(n_ctx * 8 * 2304),
            hi: Cell::new(0),
            vt: std::env::var("VT").map_or(0.01, |v| v.parse().unwrap()),
            pos: 0,
            t: 0.,
            ck: None,
            pf: None,
            g,
            gpu,
        }
    }

    // the raw cache and conv state
    pub fn raw(&self) -> [&mut [u8]; 2] {
        [slice(&self.kv), slice(&self.cs)]
    }

    pub fn reset(&mut self) {
        (self.pos, self.ck) = (0, None);
        self.hi.set(0);
        slice::<f32>(&self.cs).fill(0.);
    }

    // the sliding window cache rows of the n positions before pos
    fn swa(&self, pos: usize) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        let v = visits().into_iter().filter(|v| v.0 % 6 != 5);
        v.flat_map(move |(_, slot, _)| {
            (pos.saturating_sub(1024)..pos).map(move |p| {
                let o = self.kvo[slot] + p % SW * 2048;
                o..o + 2048
            })
        })
    }

    // checkpoint the state at pos, for truncate()
    pub fn save(&mut self) {
        let kv = slice::<u16>(&self.kv);
        let mut w = vec![];
        self.swa(self.pos).for_each(|r| w.extend_from_slice(&kv[r]));
        self.ck = Some((self.pos, slice::<f32>(&self.cs).to_vec(), w));
    }

    // forget all but the first n tokens: a few steps back by the rings,
    // or back to the checkpoint. false: not possible, reset
    pub fn truncate(&mut self, n: usize) -> bool {
        if n <= self.pos && self.pos - n <= 12 {
            self.ck.take_if(|c| c.0 > n);
        } else if let Some((_, cs, w)) = self.ck.as_ref().filter(|c| c.0 == n) {
            slice::<f32>(&self.cs).copy_from_slice(cs);
            let kv = slice::<u16>(&self.kv);
            for (r, w) in self.swa(n).zip(w.chunks(2048)) {
                kv[r].copy_from_slice(w);
            }
        } else {
            return false;
        }
        self.pos = n;
        self.hi.set(self.hi.get().min(n));
        true
    }

    // the dispatches of one token: (kind, visit, encode)
    pub fn sched(&self, tok: u32, pos: usize, head: bool) -> Vec<Op<'_>> {
        self.sched_row(tok, pos, head, 0)
    }

    // floats of an activation row
    pub fn rs(&self) -> usize {
        slice::<f32>(&self.a).len() / ROWS
    }

    // sched() on activation row `row` (and its 16 ids)
    pub fn sched_row(&self, tok: u32, pos: usize, head: bool, row: usize) -> Vec<Op<'_>> {
        let (ro, io) = (4 * row * self.rs(), 64 * row);
        assert!(pos < self.n_ctx, "context full");
        let mut ops: Vec<Op> = vec![];
        let mut vi = 0;
        // an ops.metal kernel
        macro_rules! k {
            ($n:expr, $g:expr, $th:expr, $($p:expr),*) => {{
                let mut p = [0u32; 24];
                let v = [$($p as u32),*];
                p[..v.len()].copy_from_slice(&v);
                let pipe = &self.pipes[$n];
                ops.push(($n, vi, Box::new(move |e: &Enc| {
                    let b = [&self.a, &self.s, &self.kv, &self.cs, &self.id, &self.sh,
                        &self.k8];
                    let mut b = b.map(|b| (b, 0));
                    (b[0].1, b[4].1) = (ro, io);
                    call(e, pipe, &p, &b, $g, $th)
                }), None));
            }};
        }
        // a weight mat-vec from a[x] to a[y]
        macro_rules! mv {
            ($n:expr, $t:expr, $x:expr, $y:expr, $ni:expr, $no:expr) => {{
                let (t, x, y): (T, usize, usize) = ($t, $x, $y);
                ops.push((crate::mv::name(t.ty), vi, Box::new(move |e: &Enc| {
                    let (w, id) = (self.gpu.w(t), (&self.id, io));
                    let (x, y) = ((&self.a, ro + 4 * x), (&self.a, ro + 4 * y));
                    match $n {
                        "up" => self.mv.experts_up(e, t.ty, w, id, x, y, $ni, $no),
                        "down" => self.mv.experts_down(e, t.ty, w, id, x, y, $ni, $no),
                        _ => self.mv.dense(e, t.ty, w, x, y, $ni, $no),
                    }
                }), Some(($n, t, x, y))));
            }};
        }
        let (emb, t32) = (self.emb, [18, 1, 1]);
        ops.push(("row", 0, Box::new(move |e: &Enc| {
            self.mv.row(e, emb.ty, self.gpu.w(emb), tok as usize, (&self.a, ro + 4 * VR), N)
        }), None));
        k!("dots", [2, 1, 1], 576, 0, VR, 0, 0, 0, pos, BK, 0, 0, 0, SC);
        k!("elem", t32, 32, 0, VR, 0, 0, 0, pos, BK, 0, 0, 0, SC, 1, VR, H, self.tn,
            0, 0, 1);
        let (mut nb, mut fin) = (0, [0, H, 0, 0, 0]); // pending ffn conv
        // the int8 copy of the rows that prefill or a snapshot wrote
        let (lt, full) = (self.vt.ln().to_bits(), visits().into_iter().filter(|v| v.0 % 6 == 5));
        let full: Vec<_> = full.map(|v| v.1).collect();
        let k8 = |slot| full.iter().position(|&s| s == slot).unwrap() * self.n_ctx * 1152;
        let (hi, w8) = (self.hi.get().min(pos), self.vt > 0. && pos >= W8);
        if w8 {
            if hi < pos {
                self.gpu.run(|e| for &s in &full {
                    let p = [self.kvo[s] + hi * 2048, k8(s) + hi * 1152].map(|x| x as u32);
                    let mut b = [&self.kv; 7].map(|b| (b, 0));
                    b[6].0 = &self.k8;
                    call(e, &self.pipes["pack"], &p, &b, [pos - hi, 4, 1], 32)
                });
            }
            self.hi.set(pos + 1);
        }
        for (i, slot, pass) in visits() {
            let l = &self.l[i];
            let (first, lp) = (i % 6 == 0, pass > 0);
            let msg = if lp { MSG + (i - 6) * 4 * N } else { 0 };
            let nx = if lp || !l.swa { 3 } else { 1 };
            let x = |j: usize| X + (nx / 3) * j * N;
            if i == 6 {
                nb = 1;
                fin[2] = PRE;
            }
            if i == 6 && lp {
                fin = [0, PRE, 0, 0, 0];
            }
            let ([f0, f1, f2, f3, f4], [w0, w1, w2]) = (fin, l.nw);
            k!("dots", [nb + 2, l.nq.max(1) as usize, 1], 576, f0, f1, f2, f3, f4, pos,
                BK, nb, l.nq, l.aq, SC);
            k!("elem", t32, 32, f0, f1, f2, f3, f4, pos, BK, nb, l.nq, l.aq, SC, nx,
                VR, X, w0, w1, w2);
            if lp {
                k!("low", [65, 3, 1], 576, VR, msg, l.ctl, GC);
                k!("ctl", [72, 3, 1], 32, VR, msg, l.ctl, GC, X, w0, w1, w2);
            }
            mv!("mv", l.q, x(0), Q, N, 3072);
            mv!("mv", l.k, x(1), K, N, 1024);
            mv!("mv", l.v, x(2), K + 1024, N, 1024);
            mv!("mv", l.rel, x(0), R, N, 384);
            mv!("mv", l.gate, x(0), G, N, 3072);
            let (idx, nk, tau) = if l.swa {
                (pos % SW, (pos + 1).min(1024), 1.)
            } else {
                let t = ((pos + 1) as f32 / self.hp[2]).max(1.).ln();
                (pos, pos + 1, 1. + self.hp[3] * t)
            };
            let (cs, kv) = (slot * CS, self.kvo[slot]);
            k!("post", [24, 1, 1], 576, Q, K, R, l.qc, l.rc, l.kc, l.vc, l.kd, l.vd,
                cs, kv + idx * 2048, f32::to_bits(tau), self.hp[0].to_bits(),
                self.hp[1].to_bits(), pos, VR, nx, X, IS, l.sink, SK);
            let ng = nk.div_ceil(128);
            let nm = self.n_ctx.max(1024) + 256;
            let (mx, gm, pa) = (PT + 24 * nm, PT + 27 * nm, PT + 27 * nm + 24);
            let th = (ng * 16).next_multiple_of(32).min(1024);
            let nc = if l.swa || !w8 {
                k!("score", [ng, 8, 1], 512, Q, R, PT, kv, nk, pos, l.swa, l.rb, mx);
                k!("smax", [24, 1, 1], th, mx, ng * 16, SK, gm, th);
                k!("gather", [nk.div_ceil(256), 8, 1], 512, gm, 0, PT, kv, nk, pos, l.swa,
                    0, pa);
                nk.div_ceil(256)
            } else {
                // int8 first, f16 for the keys with weight
                let (k8, cw) = (k8(slot), (nk / 64).clamp(16, 256));
                k!("pack", [1, 4, 1], 32, kv + pos * 2048, k8 + pos * 1152);
                k!("scorew", [ng * 16, 2, 1], 32, Q, R, PT, k8, nk, pos, 0, l.rb, mx);
                k!("smax", [24, 1, 1], th, mx, ng * 16, 0, gm, th);
                let nc = nk.div_ceil(cw);
                k!("gatherw", [nc, 2, 1], 32, gm, 0, PT, 0, nk, 0, 0, 0, pa, 0, 0, k8, cw);
                k!("fix", [64, 8, 1], 32, gm, Q, PT, kv, nk, 0, 0, 0, pa, 0, lt, k8,
                    nk.div_ceil(64), nc);
                nc + 64
            };
            let ns = nc.min(16);
            k!("merge", [24, 1, 1], 32 * ns, pa, nc, SK, G, O, IS, gm, ns);
            mv!("mv", l.o, O, AO, 3072, N);
            nb += first as usize;
            let (r, st) = (if first { 0 } else { H }, cs + 2048);
            k!("dots", [nb + 2, 1, 1], 576, AO, r, F, st, l.apc, pos, BK, nb, 1, l.fq, SC);
            k!("elem", t32, 32, AO, r, F, st, l.apc, pos, BK, nb, 1, l.fq, SC, 1, VR,
                XF, l.fnw);
            if lp {
                let (m, c) = (msg + 3 * N, l.ctl as usize + 768 * N);
                k!("low", [65, 1, 1], 576, VR, m, c, GC);
                k!("ctl", [72, 1, 1], 32, VR, m, c, GC, XF, l.fnw);
            }
            k!("rows", [129, 1, 1], 576, XF, l.gi, LGT, VR);
            k!("top8", [128, 1, 1], 32, LGT, l.pb, WT);
            mv!("up", l.gu, XF, GU, N, N);
            k!("act", [9, 1, 1], 256, GU, HH, WT, l.us, l.ds, LGT + 128);
            mv!("down", l.dn, HH, D, 1152, N);
            fin = [D, F, H, cs + 4352, l.fpc as usize];
            if i == 17 && pass < 2 {
                k!("elem", t32, 32, D, F, H, fin[3], fin[4], pos, BK, nb, 0, 0, SC);
                k!("msg", [72, 12, 1], 32, BK + 2 * N, H, self.mq, MSG);
            }
            vi += 1;
        }
        let [f0, f1, _, f3, f4] = fin;
        if !head {
            // only the conv state of the last ffn
            k!("elem", t32, 32, f0, f1, 0, f3, f4, pos, BK, nb, 0, 0, SC);
            return ops;
        }
        k!("dots", [nb + 2, 1, 1], 576, f0, f1, 0, f3, f4, pos, BK, nb, 1, self.oq, SC);
        k!("elem", t32, 32, f0, f1, 0, f3, f4, pos, BK, nb, 1, self.oq, SC, 1, VR, X,
            self.on);
        mv!("mv", self.out, X, LG, N, 201024);
        k!("amax", [1563, 1, 1], 32, LG, 200058, PT);
        k!("best", [1, 1, 1], 1024, PT, 1563, 8);
        ops
    }

    // decode one token; returns the greedy next id
    pub fn step(&mut self, tok: u32) -> u32 {
        let ops = self.sched(tok, self.pos, true);
        // several command buffers: the gpu starts while the rest encodes
        let t = objc2::rc::autoreleasepool(|_| {
            let cbs: Vec<_> = ops
                .chunks(128)
                .map(|c| {
                    let cb = self.gpu.q.commandBuffer().unwrap();
                    let e = cb.computeCommandEncoder().unwrap();
                    c.iter().for_each(|o| o.2(&e));
                    e.endEncoding();
                    cb.commit();
                    cb
                })
                .collect();
            let l = cbs.last().unwrap();
            l.waitUntilCompleted();
            assert!(l.error().is_none(), "metal: {:?}", l.error());
            l.GPUEndTime() - cbs[0].GPUStartTime()
        });
        drop(ops);
        (self.t, self.pos) = (t, self.pos + 1);
        self.next()
    }

    // decode toks (not empty), logits only for the last: no output head
    // and no wait between the others
    pub fn feed(&mut self, toks: &[u32]) -> u32 {
        let toks = self.prefill(toks, 512);
        let (l, h) = toks.split_last().unwrap();
        for c in h.chunks(4) {
            let ops = c.iter().enumerate();
            let ops: Vec<_> = ops.flat_map(|(i, &t)| self.sched(t, self.pos + i, false)).collect();
            self.gpu.run(|e| ops.iter().for_each(|o| o.2(e)));
            drop(ops);
            self.pos += c.len();
        }
        self.step(*l)
    }

    // the experts of the last visit
    pub fn experts(&self) -> Vec<u32> {
        slice::<u32>(&self.id)[..8].to_vec()
    }

    // greedy id of the last step
    pub fn next(&self) -> u32 {
        slice::<u32>(&self.id)[8]
    }

    // logits of the last step, unpadded vocabulary, after the softcap
    pub fn logits(&self) -> Vec<f32> {
        let (c, a) = (self.g.f("inkling2.final_logit_softcapping") as f32, slice::<f32>(&self.a));
        // the output norm scale is left to the reader
        let n = a[VR..VR + N].iter().map(|x| x * x).sum::<f32>() / N as f32;
        let (n, l) = ((n + 1e-6).sqrt(), &a[LG..LG + self.g.u("inkling2.vocab_size_unpadded")]);
        l.iter().map(|x| c * (x / n / c).tanh()).collect()
    }

    // a region of the activations by ref dump name
    pub fn peek(&self, name: &str, phys: usize) -> Option<&[f32]> {
        let (o, n) = match name {
            "Qcur" => (Q, 3072),
            "Kcur" => (K, 1024),
            "Vcur" => (K + 1024, 1024),
            "Rcur" => (R, 384),
            "kqv_out" => (O, 3072),
            "gate" => (G, 3072),
            "is" => (IS, 1),
            "ffn_inp" => (F, N),
            "ffn_moe_out" => (D, N),
            "ffn_moe_situ" => (HH, 8 * 1152),
            "ffn_moe_weights_scaled" => (WT, 8), // before the norm
            "l_out" => (if phys == 5 { PRE } else { H }, N),
            "result_output" => (LG, 200058),
            _ => return None,
        };
        Some(&slice::<f32>(&self.a)[o..o + n])
    }
}
