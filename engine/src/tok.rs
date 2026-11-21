// the gguf tokenizer (model gpt2, pre inkling2), as llama.cpp runs it:
// special tokens are cut out first, the text between them is split into
// words (the o200k regex, hand-rolled in std::regex's backtracking order)
// and each word is merged by rank. no bos, no eos, no space cleaning.
//   let t = Tok::new(&g); t.enc("text", true) -> ids; t.txt[id] -> bytes
use crate::gguf::Gguf;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

// cpt << 3 | class, sorted, from llama.cpp's unicode-data.cpp
static CAT: &[u8] = include_bytes!("ucat.bin");
const DEAD: u32 = u32::MAX;

pub struct Tok {
    pub txt: Vec<Vec<u8>>, // id -> bytes
    pub ctl: Vec<bool>,    // id -> is a control token
    pub eos: u32,
    ids: HashMap<Vec<u8>, u32>,
    rank: HashMap<(u32, u32), (u32, u32)>, // pair -> rank, merged id
    byte: [u32; 256],
    spec: Vec<u32>, // control ids, longest text first
}

// 0 other, 1 letter, 2 number, 3 mark, 4 whitespace
fn class(c: char) -> u8 {
    let e = |i| u32::from_le_bytes(CAT[4 * i..4 * i + 4].try_into().unwrap());
    match c {
        'a'..='z' | 'A'..='Z' => 1,
        '0'..='9' => 2,
        ' ' | '\t'..='\r' => 4,
        c if c.is_ascii() => 0,
        c => {
            let (mut lo, mut hi) = (0, CAT.len() / 4);
            while hi - lo > 1 {
                let m = (lo + hi) / 2;
                *if e(m) >> 3 <= c as u32 { &mut lo } else { &mut hi } = m;
            }
            e(lo) as u8 & 7
        }
    }
}

// word boundaries (char indices) of
//   [^\r\n\p{L}\p{N}]?X*Y+C?|[^\r\n\p{L}\p{N}]?X+Y*C?|\p{N}{1,3}|
//    ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
// X: letter or mark, not a-z; Y: letter or mark, not A-Z; C: 's 't 're ..
// k is the class and a the ascii byte (0x80 if none) of each char
fn word(k: &[u8], a: &[u8], i: usize) -> usize {
    let n = k.len();
    let run = |mut j: usize, f: &dyn Fn(usize) -> bool| {
        while j < n && f(j) {
            j += 1
        }
        j
    };
    let lm = |j: usize| k[j] == 1 || k[j] == 3;
    let x = |j: usize| lm(j) && !a[j].is_ascii_lowercase();
    let y = |j: usize| lm(j) && !a[j].is_ascii_uppercase();
    let nl = |j: usize| a[j] == b'\r' || a[j] == b'\n';
    let contr = |j: usize| {
        let s: Vec<u8> =
            a[j..n.min(j + 3)].iter().map(u8::to_ascii_lowercase).collect();
        let c = [&b"'re"[..], b"'ve", b"'ll", b"'s", b"'t", b"'m", b"'d"];
        j + c.iter().find(|c| s.starts_with(c)).map_or(0, |c| c.len())
    };
    let pre = !nl(i) && k[i] != 1 && k[i] != 2;
    let starts = [i + 1, i];
    let starts = &starts[!pre as usize..];
    for &p in starts {
        let e = run(p, &x);
        for j in (p..=e).rev() {
            let z = run(j, &y);
            if z > j {
                return contr(z);
            }
        }
    }
    for &p in starts {
        let e = run(p, &x);
        if e > p {
            return contr(run(e, &y));
        }
    }
    if k[i] == 2 {
        return run(i, &|j| k[j] == 2).min(i + 3);
    }
    let q = |j: usize| k[j] != 4 && k[j] != 1 && k[j] != 2;
    let p = i + (a[i] == b' ' && i + 1 < n && q(i + 1)) as usize;
    let e = run(p, &q);
    if e > p {
        return run(e, &|j| nl(j) || a[j] == b'/');
    }
    let e = run(i, &|j| k[j] == 4);
    match (i..e).rev().find(|&j| nl(j)) {
        Some(j) => j + 1,
        _ if e < n && e - i > 1 => e - 1,
        _ => e.max(i + 1),
    }
}

impl Tok {
    pub fn new(g: &Gguf) -> Tok {
        // gpt2 byte encoding: printable bytes are themselves, the rest
        // are 256, 257, .. in order
        let (mut inv, mut n) = ([b'?'; 324], 256);
        for b in 0..256 {
            let p = matches!(b, 33..=126 | 161..=172 | 174..=255);
            inv[if p { b } else { n }] = b as u8;
            n += !p as usize;
        }
        let raw = |s: &[u8]| -> Vec<u8> {
            let s = String::from_utf8_lossy(s);
            s.chars().map(|c| inv[(c as usize).min(323)]).collect()
        };
        let ctl: Vec<bool> = g
            .nums("tokenizer.ggml.token_type")
            .iter()
            .map(|&t| t != 1.0)
            .collect();
        let txt: Vec<Vec<u8>> = g
            .strs("tokenizer.ggml.tokens")
            .zip(&ctl)
            .map(|(s, &c)| if c { s.to_vec() } else { raw(s) })
            .collect();
        let ids: HashMap<_, _> = txt.iter().cloned().zip(0u32..).collect();
        let rank = g
            .strs("tokenizer.ggml.merges")
            .zip(0u32..)
            .filter_map(|(m, k)| {
                let i = 1 + m.get(1..)?.iter().position(|&b| b == b' ')?;
                let (l, r) = (raw(&m[..i]), raw(&m[i + 1..]));
                let lr = (*ids.get(&l)?, *ids.get(&r)?);
                Some((lr, (k, *ids.get(&[l, r].concat())?)))
            })
            .collect();
        let mut byte = [DEAD; 256];
        for b in 0..=255u8 {
            byte[b as usize] = ids.get(&[b][..]).copied().unwrap_or(DEAD);
        }
        let mut spec: Vec<u32> =
            (0..txt.len() as u32).filter(|&i| ctl[i as usize]).collect();
        spec.sort_by_key(|&i| Reverse(txt[i as usize].len()));
        let eos = g.u("tokenizer.ggml.eos_token_id") as u32;
        Tok { txt, ctl, eos, ids, rank, byte, spec }
    }

    pub fn id(&self, s: &str) -> u32 {
        self.ids[s.as_bytes()]
    }

    // special: control token text in s becomes the control token
    pub fn enc(&self, s: &str, special: bool) -> Vec<u32> {
        let (mut out, mut a, mut i, b) = (vec![], 0, 0, s.as_bytes());
        while special && i < b.len() {
            let hit = |&&t: &&u32| b[i..].starts_with(&self.txt[t as usize]);
            match self.spec.iter().find(hit) {
                Some(&t) => {
                    self.text(&s[a..i], &mut out);
                    out.push(t);
                    i += self.txt[t as usize].len();
                    a = i;
                }
                None => i += 1,
            }
        }
        self.text(&s[a..], &mut out);
        out
    }

    fn text(&self, s: &str, out: &mut Vec<u32>) {
        let k: Vec<u8> = s.chars().map(class).collect();
        let asc = |c: char| if c.is_ascii() { c as u8 } else { 0x80 };
        let a: Vec<u8> = s.chars().map(asc).collect();
        let mut o: Vec<usize> = s.char_indices().map(|x| x.0).collect();
        o.push(s.len());
        let mut i = 0;
        while i < k.len() {
            let j = word(&k, &a, i);
            self.bpe(&s.as_bytes()[o[i]..o[j]], out);
            i = j;
        }
    }

    // merge the lowest-ranked pair first, the leftmost on a tie
    fn bpe(&self, w: &[u8], out: &mut Vec<u32>) {
        let n = w.len();
        let mut s: Vec<u32> =
            w.iter().map(|&b| self.byte[b as usize]).collect();
        let mut nx: Vec<usize> = (1..=n).collect();
        let mut pv: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut q = BinaryHeap::new();
        let push = |q: &mut BinaryHeap<_>, s: &[u32], l: usize, r: usize| {
            if l < n && r < n {
                if let Some(&(k, m)) = self.rank.get(&(s[l], s[r])) {
                    q.push(Reverse((k, l, r, s[l], s[r], m)));
                }
            }
        };
        for i in 1..n {
            push(&mut q, &s, i - 1, i);
        }
        while let Some(Reverse((_, l, r, a, b, m))) = q.pop() {
            if s[l] != a || s[r] != b {
                continue; // stale
            }
            (s[l], s[r], nx[l]) = (m, DEAD, nx[r]);
            if nx[l] < n {
                pv[nx[l]] = l;
            }
            push(&mut q, &s, pv[l], l);
            push(&mut q, &s, l, nx[l]);
        }
        out.extend(s.into_iter().filter(|&t| t != DEAD));
    }
}

// ink tok <gguf> <file>: the ids of the file, as llama-tokenize --ids
pub fn main(a: &[String]) {
    let t = Tok::new(&crate::gguf::open(&a[0]));
    let s = std::fs::read_to_string(&a[1]).unwrap();
    println!("{:?}", t.enc(&s, true));
}
