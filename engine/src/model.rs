// inkling-2-mini decode, one token per command buffer (ops.metal, mv.rs).
//   let mut m = Model::open(path, n_ctx);
//   let next = m.step(token);     // greedy next id; m.logits() for all
//   m.reset();
use crate::gguf::{self, Gguf, Tensor};
use crate::gpu::*;
use crate::mv::Mv;
use std::collections::HashMap;

const N: usize = 2304;
const OPS: &str = include_str!("ops.metal");
const KINDS: [&str; 9] =
    ["prep", "post", "attn", "merge", "mid", "act", "fin", "msg", "amax"];
// regions of the activation buffer, in floats. 0 means "none" in params.
const PRE: usize = N; // prelude output
const H: usize = 2 * N; // residual stream
const X: usize = 3 * N; // q, k, v inputs
const XF: usize = 6 * N; // ffn input
const F: usize = 7 * N; // ffn_inp
const AO: usize = 8 * N; // attention output after wo
const D: usize = 9 * N; // expert output
const BK: usize = 10 * N; // normalized block stack, 5 deep
const TMP: usize = 15 * N;
const Q: usize = 17 * N; // q, gate, attention out: 3072 each
const G: usize = Q + 3072;
const O: usize = G + 3072;
const K: usize = O + 3072;
const V: usize = K + 1024;
const R: usize = V + 1024;
const SK: usize = R + 384;
const WT: usize = SK + 24;
const GU: usize = WT + 8; // 8 expert gate_up rows
const HH: usize = GU + 8 * N;
const MSG: usize = HH + 4 * N; // 12 layers x 4 messages
const LG: usize = MSG + 48 * N; // logits
const PT: usize = LG + 201024; // attention partials, to the end
const CS: usize = 6656 * 3; // conv state floats per slot

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

pub type Op<'a> = (&'static str, usize, Box<dyn Fn(&Enc) + 'a>);

pub struct Model {
    pub g: &'static Gguf,
    pub gpu: Gpu,
    mv: Mv,
    pipes: HashMap<&'static str, Pipe>,
    pub a: Buf,
    s: Buf,
    kv: Buf,
    cs: Buf,
    id: Buf,
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
    pub pos: usize,
    pub t: f64, // gpu seconds of the last step
}

// (physical layer, cache slot, pass) in execution order
pub fn visits() -> Vec<(usize, usize, usize)> {
    let core = |p| (6..18).map(move |i| (i, if p > 0 { 12 * p + 6 + i } else { i }, p));
    let lin = |r: std::ops::Range<usize>| r.map(|i| (i, i, 0));
    lin(0..6).chain((0..3).flat_map(core)).chain(lin(18..24)).collect()
}

impl Model {
    pub fn open(path: &str, n_ctx: usize) -> Model {
        let g: &'static Gguf = Box::leak(Box::new(gguf::open(path)));
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
        let l = (0..24)
            .map(|i| {
                let t = |n: &str| &g.t[format!("blk.{i}.{n}.weight").as_str()];
                let mut w = |n: &str| sm(format!("blk.{i}.{n}"));
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
                            .map(|k| w(&format!("loop_{c}_{k}.weight")))[0]
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
                    sink: w("attn_sink_proj.weight"),
                    qc: w("attn_q_clip.weight"),
                    rc: w("attn_rel_clip.weight"),
                    rb: w("attn_rel_bias.weight"),
                    kc: w("attn_k_conv.weight"),
                    vc: w("attn_v_conv.weight"),
                    kd: w("attn_k_conv_proj.weight"),
                    vd: w("attn_v_conv_proj.weight"),
                    apc: w("attn_post_conv.weight"),
                    fpc: w("ffn_post_conv.weight"),
                    fq: w("ffn_res_query.weight"),
                    fnw: w("ffn_norm.weight"),
                    gi: w("ffn_gate_inp.weight"),
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
        let sb = gpu.buf(4 * s.len());
        slice::<f32>(&sb)[..s.len()].copy_from_slice(&s);
        let mut kvo = vec![0];
        let mut by_slot = visits();
        by_slot.sort_by_key(|v| v.1);
        for (i, _, _) in by_slot {
            let n = if i % 6 != 5 { 1024 } else { n_ctx };
            kvo.push(kvo.last().unwrap() + n * 2048);
        }
        let nc = n_ctx.div_ceil(256).max(4);
        let pipes = KINDS.map(|k| (k, gpu.pipe(OPS, "", k))).into();
        Model {
            mv: Mv::new(&gpu, ""),
            pipes,
            a: gpu.buf(4 * (PT + nc * 24 * 130)),
            s: sb,
            kv: gpu.buf(2 * kvo[48]),
            cs: gpu.buf(4 * 48 * CS),
            id: gpu.buf(4 * 16),
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
            pos: 0,
            t: 0.,
            g,
            gpu,
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        slice::<f32>(&self.cs).fill(0.);
    }

    // the dispatches of one token: (kind, visit, encode)
    pub fn sched(&self, tok: u32) -> Vec<Op<'_>> {
        let pos = self.pos;
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
                    let b = [&self.a, &self.s, &self.kv, &self.cs, &self.id];
                    call(e, pipe, &p, &b.map(|b| (b, 0)), $g, $th)
                })));
            }};
        }
        // a weight mat-vec from a[x] to a[y]
        macro_rules! mv {
            ($n:expr, $t:expr, $x:expr, $y:expr, $ni:expr, $no:expr) => {{
                let (t, x, y): (T, usize, usize) = ($t, $x, $y);
                ops.push((crate::mv::name(t.ty), vi, Box::new(move |e: &Enc| {
                    let (w, x, y) = (self.gpu.w(t), (&self.a, 4 * x), (&self.a, 4 * y));
                    match $n {
                        "up" => self.mv.experts_up(e, t.ty, w, (&self.id, 0), x, y, $ni, $no),
                        "down" => self.mv.experts_down(e, t.ty, w, (&self.id, 0), x, y, $ni, $no),
                        _ => self.mv.dense(e, t.ty, w, x, y, $ni, $no),
                    }
                })));
            }};
        }
        let emb = self.emb;
        ops.push(("row", 0, Box::new(move |e: &Enc| {
            self.mv.row(e, emb.ty, self.gpu.w(emb), tok as usize, (&self.a, 4 * H), N)
        })));
        k!("prep", [1, 1, 1], 768, H, BK, 0, 0, 0, H, 1, 0, 0, 0, self.tn, 0, 0, TMP);
        let mut nb = 0;
        for (i, slot, pass) in visits() {
            let l = &self.l[i];
            let (first, lp) = (i % 6 == 0, pass > 0);
            let h = if i == 6 { PRE } else { H };
            let msg = if lp { MSG + (i - 6) * 4 * N } else { 0 };
            if i == 6 {
                nb = 1;
            }
            let nx = if lp || !l.swa { 3 } else { 1 };
            let x = |j: usize| X + (nx / 3) * j * N;
            k!("prep", [1, 1, 1], 768, h, BK, nb, l.nq, l.aq, X, nx, msg, l.ctl, 0,
                l.nw[0], l.nw[1], l.nw[2], TMP);
            mv!("mv", l.q, x(0), Q, N, 3072);
            mv!("mv", l.k, x(1), K, N, 1024);
            mv!("mv", l.v, x(2), V, N, 1024);
            mv!("mv", l.rel, x(0), R, N, 384);
            mv!("mv", l.gate, x(0), G, N, 3072);
            let (idx, nk, tau) = if l.swa {
                (pos % 1024, (pos + 1).min(1024), 1.)
            } else {
                let t = ((pos + 1) as f32 / self.hp[2]).max(1.).ln();
                (pos, pos + 1, 1. + self.hp[3] * t)
            };
            let (cs, kv) = (slot * CS, self.kvo[slot]);
            k!("post", [1, 1, 1], 768, X, Q, K, V, R, SK, l.sink, l.qc, l.rc, l.kc,
                l.vc, l.kd, l.vd, cs, kv + idx * 2048, f32::to_bits(tau),
                self.hp[0].to_bits(), self.hp[1].to_bits());
            let nc = nk.div_ceil(256);
            k!("attn", [nc, 8, 1], 96, Q, R, PT, kv, nk, pos, l.swa, l.rb);
            k!("merge", [1, 1, 1], 768, PT, nc, SK, G, O);
            mv!("mv", l.o, O, AO, 3072, N);
            nb += first as usize;
            k!("mid", [1, 1, 1], 768, F, BK, nb, 1, l.fq, XF, 1, msg, l.ctl, 3,
                l.fnw, 0, 0, TMP, AO, if first { 0 } else { h }, cs + 2048 * 3,
                l.apc, l.gi, l.pb, WT);
            mv!("up", l.gu, XF, GU, N, N);
            k!("act", [36, 1, 1], 256, GU, HH, WT, l.us, l.ds);
            mv!("down", l.dn, HH, D, 1152, N);
            k!("fin", [9, 1, 1], 256, D, F, if i == 5 { PRE } else { H },
                cs + 4352 * 3, l.fpc);
            if i == 17 && pass < 2 {
                k!("msg", [12, 1, 1], 768, BK + 2 * N, H, self.mq, MSG);
            }
            vi += 1;
        }
        k!("prep", [1, 1, 1], 768, H, BK, nb, 1, self.oq, X, 1, 0, 0, 0, self.on, 0, 0, TMP);
        mv!("mv", self.out, X, LG, N, 201024);
        k!("amax", [1, 1, 1], 768, LG, 200058, 8);
        ops
    }

    // decode one token; returns the greedy next id
    pub fn step(&mut self, tok: u32) -> u32 {
        let ops = self.sched(tok);
        let t = self.gpu.run(|e| ops.iter().for_each(|o| o.2(e)));
        drop(ops);
        (self.t, self.pos) = (t, self.pos + 1);
        self.next()
    }

    // greedy id of the last step
    pub fn next(&self) -> u32 {
        slice::<u32>(&self.id)[8]
    }

    // logits of the last step, unpadded vocabulary, after the softcap
    pub fn logits(&self) -> Vec<f32> {
        let c = self.g.f("inkling2.final_logit_softcapping") as f32;
        let l = &slice::<f32>(&self.a)[LG..LG + self.g.u("inkling2.vocab_size_unpadded")];
        l.iter().map(|x| c * (x / c).tanh()).collect()
    }

    // a region of the activations by ref dump name
    pub fn peek(&self, name: &str, phys: usize) -> Option<&[f32]> {
        let (o, n) = match name {
            "Qcur" => (Q, 3072),
            "Kcur" => (K, 1024),
            "Vcur" => (V, 1024),
            "Rcur" => (R, 384),
            "kqv_out" => (O, 3072),
            "ffn_inp" => (F, N),
            "ffn_moe_out" => (D, N),
            "ffn_moe_situ" => (HH, 8 * 1152),
            "ffn_moe_weights_scaled" => (WT, 8),
            "l_out" => (if phys == 5 { PRE } else { H }, N),
            "result_output" => (LG, 200058),
            _ => return None,
        };
        Some(&slice::<f32>(&self.a)[o..o + n])
    }
}
