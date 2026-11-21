// weight mat-vec for one token, every tensor type of the file (mv.metal).
//   let mv = Mv::new(&gpu, "");
//   mv.dense(e, ty, w, x, y, n_in, n_out)         y = W x
//   mv.row(e, ty, w, i, y, n_in)                  y = row i of W
//   mv.experts_up(e, ty, w, ids, x, y, n_in, n_out)    y[k] = W[ids[k]] x
//   mv.experts_down(e, ty, w, ids, x, y, n_in, n_out)  y = sum W[ids[k]] x[k]
// w, x, y, ids are (buffer, byte offset); x and y are f32, ids NE u32.
// deq() is the scalar reference.
use crate::gguf::blk;
use crate::gpu::*;
use std::{collections::HashMap, sync::OnceLock};

pub const NE: usize = 8; // experts per token
const GRIDS: &str = include_str!("grids.metal");
pub const SRC: &str = concat!(
    "#include <metal_stdlib>\n",
    include_str!("grids.metal"),
    include_str!("mv.metal")
);

#[repr(C)]
struct P(u32, u32, u32, u32); // n_in, n_out, row bytes, experts

// (type, name, rows per threadgroup as in mv.metal)
const TY: [(u32, &str, usize); 12] = [
    (0, "f32", 2),
    (30, "bf16", 2),
    (8, "q8_0", 4),
    (12, "q4_k", 2),
    (13, "q5_k", 2),
    (14, "q6_k", 2),
    (16, "iq2_xxs", 8),
    (17, "iq2_xs", 4),
    (22, "iq2_s", 8),
    (18, "iq3_xxs", 8),
    (20, "iq4_nl", 4),
    (40, "nvfp4", 8),
];

pub fn name(ty: u32) -> &'static str {
    TY.iter().find(|t| t.0 == ty).expect("mv: tensor type").1
}

pub struct Mv {
    p: HashMap<(&'static str, u32), (Pipe, usize)>,
}

impl Mv {
    // defs: extra #defines; NR n overrides the rows per threadgroup
    pub fn new(g: &Gpu, defs: &str) -> Mv {
        let nr = defs.split("NR ").nth(1).map(|s| {
            s.split_whitespace().next().unwrap().parse().unwrap()
        });
        let mut p = HashMap::new();
        for (ty, name, r) in TY {
            for k in ["mv", "up", "down", "row"] {
                let f = g.pipe(SRC, defs, &format!("{k}_{name}"));
                p.insert((k, ty), (f, nr.unwrap_or(r)));
            }
        }
        Mv { p }
    }

    #[allow(clippy::too_many_arguments)]
    fn go(&self, e: &Enc, k: &'static str, ty: u32, b: &[B], n_in: usize, n_out: usize, rows: usize) {
        let rb = n_in / blk(ty).0 * blk(ty).1;
        let p = P(n_in as u32, n_out as u32, rb as u32, NE as u32);
        let (f, r) = &self.p[&(k, ty)];
        call(e, f, &p, b, [rows / r, 1, 1], 32);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dense(&self, e: &Enc, ty: u32, w: B, x: B, y: B, n_in: usize, n_out: usize) {
        self.go(e, "mv", ty, &[w, x, y], n_in, n_out, n_out);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn experts_up(&self, e: &Enc, ty: u32, w: B, ids: B, x: B, y: B, n_in: usize, n_out: usize) {
        self.go(e, "up", ty, &[w, x, y, ids], n_in, n_out, NE * n_out);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn experts_down(&self, e: &Enc, ty: u32, w: B, ids: B, x: B, y: B, n_in: usize, n_out: usize) {
        self.go(e, "down", ty, &[w, x, y, ids], n_in, n_out, n_out);
    }

    pub fn row(&self, e: &Enc, ty: u32, w: B, i: usize, y: B, n_in: usize) {
        let rb = n_in / blk(ty).0 * blk(ty).1;
        let p = P(n_in as u32, 1, rb as u32, 0);
        let b = [(w.0, w.1 + i * rb), y, y];
        call(e, &self.p[&("row", ty)].0, &p, &b, [n_in.div_ceil(32), 1, 1], 32);
    }
}

// a table of grids.metal
fn tab(name: &str) -> Vec<u64> {
    let s = &GRIDS[GRIDS.find(&format!(" {name}[")).unwrap()..];
    s[s.find('{').unwrap() + 1..s.find('}').unwrap()]
        .split(',')
        .filter_map(|v| {
            let v = v.trim();
            match v.strip_prefix("0x") {
                Some(h) => u64::from_str_radix(h, 16).ok(),
                _ => v.parse().ok(),
            }
        })
        .collect()
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = (h >> 15, (h >> 10) & 31, (h & 1023) as f32);
    let v = match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1. + m / 1024.) * 2f32.powi(e as i32 - 15),
    };
    if s == 1 { -v } else { v }
}

// scalar dequant of n elements of a row, after ggml-quants.c
pub fn deq(ty: u32, row: &[u8], n: usize) -> Vec<f32> {
    static T: OnceLock<[Vec<u64>; 5]> = OnceLock::new();
    let [ks, g2xxs, g2xs, g2s, g3xxs] = T.get_or_init(|| {
        ["ksigns_iq2xs", "iq2xxs_grid", "iq2xs_grid", "iq2s_grid", "iq3xxs_grid"]
            .map(tab)
    });
    const NL: [i8; 16] =
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    const FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let (be, bb) = blk(ty);
    (0..n)
        .map(|i| {
            let (b, e) = (&row[i / be * bb..], i % be);
            let u = |o: usize| b[o] as usize;
            let h = |o| f16(u16::from_le_bytes([b[o], b[o + 1]]));
            let w = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
            // 8 values per grid entry, sign bit j of sg, block scale ls
            let (k, j) = (e / 8, e % 8);
            let grid = |g: u64, sg: u64, d: f32| {
                let v = (g >> (8 * j) & 255) as f32 * d;
                if sg >> j & 1 == 1 { -v } else { v }
            };
            let ls = |o: usize| 0.5 + (u(o + k / 4) >> (4 * (k % 4 / 2)) & 15) as f32;
            // q4_k, q5_k: scale and min of sub-block j
            let k4 = |j: usize| {
                let s = |i: usize| u(4 + i);
                if j < 4 {
                    (s(j) & 63, s(j + 4) & 63)
                } else {
                    (s(j + 4) & 15 | s(j - 4) >> 6 << 4, s(j + 4) >> 4 | s(j) >> 6 << 4)
                }
            };
            match ty {
                0 => f32::from_bits(w(0)),
                30 => f32::from_bits(((u(0) | u(1) << 8) as u32) << 16),
                8 => h(0) * (b[2 + e] as i8) as f32,
                12 | 13 => {
                    let (j, l) = (e / 32, e % 32);
                    let (sc, m) = k4(j);
                    let q = if ty == 12 {
                        u(16 + 32 * (j / 2) + l) >> (4 * (j & 1)) & 15
                    } else {
                        (u(48 + 32 * (j / 2) + l) >> (4 * (j & 1)) & 15)
                            + ((u(16 + l) >> j & 1) << 4)
                    };
                    h(0) * (sc * q) as f32 - h(2) * m as f32
                }
                14 => {
                    let (n, g, l) = (e / 128, e % 128 / 32, e % 32);
                    let q = u(64 * n + 32 * (g & 1) + l) >> (4 * (g / 2)) & 15
                        | (u(128 + 32 * n + l) >> (2 * g) & 3) << 4;
                    let sc = b[192 + 8 * n + 2 * g + l / 16] as i8;
                    h(208) * sc as f32 * (q as f32 - 32.)
                }
                20 => h(0) * NL[u(2 + e % 16) >> (4 * (e / 16)) & 15] as f32,
                40 => {
                    let (s, x) = (e / 16, u(e / 16));
                    let (ex, m) = ((x >> 3 & 15) as i32, (x & 7) as f32);
                    let d = match (x, ex) {
                        (0 | 127, _) => 0.,
                        (_, 0) => m * 2f32.powi(-9),
                        _ => (1. + m / 8.) * 2f32.powi(ex - 7),
                    };
                    let q = u(4 + 8 * s + e % 8) >> (4 * (e % 16 / 8)) & 15;
                    FP4[q] as f32 * d * 0.5
                }
                16 => {
                    let (ib, l) = (k / 4, k % 4);
                    let aux = w(6 + 8 * ib);
                    let sg = ks[(aux >> (7 * l) & 127) as usize];
                    let d = h(0) * (0.5 + (aux >> 28) as f32) * 0.25;
                    grid(g2xxs[u(2 + 8 * ib + l)], sg, d)
                }
                17 => {
                    let q = u(2 + 2 * k) | u(3 + 2 * k) << 8;
                    grid(g2xs[q & 511], ks[q >> 9], h(0) * ls(66) * 0.25)
                }
                22 => {
                    let q = u(2 + k) | (u(66 + k / 4) >> (2 * (k % 4)) & 3) << 8;
                    grid(g2s[q], u(34 + k) as u64, h(0) * ls(74) * 0.25)
                }
                18 => {
                    let aux = w(66 + 4 * (k / 4));
                    let sg = ks[(aux >> (7 * (k % 4)) & 127) as usize];
                    let d = h(0) * (0.5 + (aux >> 28) as f32) * 0.5;
                    grid(g3xxs[u(2 + 2 * k + j / 4)] << (32 * (j / 4)), sg, d)
                }
                _ => panic!("mv: tensor type {ty}"),
            }
        })
        .collect()
}
