// openai-compatible chat server over one model, one request at a time.
//   gall serve <gguf> [--port 9940] [--ctx 102400] [--effort none]
//             [--alias inkling-2-mini] [--api-key-file F]
//             [--idle-unload SECONDS] [--dummy [answer]]
// POST /v1/chat/completions (stream or not), GET /v1/models, GET /health.
// with a key, all but /health need "authorization: bearer KEY". after
// --idle-unload seconds idle the model goes to disk (a snapshot of the last
// conversation) and out of memory, until the next chat request.
// the chat template of the gguf is written out in prompt(); the model
// answers in blocks, <|message_model|> then <|content_thinking|>,
// <|content_text|> or NAME<|content_invoke_tool_json|>, each closed by
// <|end_message|>. a new prompt that extends the last one is fed from
// where they part.
use crate::tok::Tok;
use std::fmt::{self, Write as _};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub trait Lm {
    fn reset(&mut self);
    // feed one token; logits of the next one over the real vocabulary
    fn step(&mut self, tok: u32) -> &[f32];
    // tokens fed since reset
    fn len(&self) -> usize;
    // forget all but the first n tokens; false if that needs a reset
    fn truncate(&mut self, n: usize) -> bool;
    // step() over toks (not empty); override to batch
    fn feed(&mut self, toks: &[u32]) -> &[f32] {
        let (l, h) = toks.split_last().unwrap();
        for &t in h {
            self.step(t);
        }
        self.step(*l)
    }
    // holds p[..k]: restore or feed and store prefixes of p up to the
    // system region p[..sys]; the new k and the length restored
    fn prime(&mut self, _p: &[u32], k: usize, _sys: usize) -> (usize, usize) {
        (k, 0)
    }
    // store toks, which it holds (or nothing, if empty), and free memory
    fn park(&mut self, _toks: &[u32]) {}
    fn loaded(&self) -> bool {
        true
    }
}

// an Lm that load() makes again on first use after park()
type Load = Box<dyn Fn() -> Box<dyn Lm>>;
struct Lazy(Option<Box<dyn Lm>>, Load);

impl Lazy {
    fn get(&mut self) -> &mut dyn Lm {
        &mut **self.0.get_or_insert_with(|| {
            let t = Instant::now();
            let m = (self.1)();
            eprintln!("gall: loaded in {:.2} s", t.elapsed().as_secs_f64());
            m
        })
    }
}

impl Lm for Lazy {
    fn reset(&mut self) {
        self.get().reset()
    }
    fn step(&mut self, t: u32) -> &[f32] {
        self.get().step(t)
    }
    fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |m| m.len())
    }
    fn truncate(&mut self, n: usize) -> bool {
        self.get().truncate(n)
    }
    fn feed(&mut self, toks: &[u32]) -> &[f32] {
        self.get().feed(toks)
    }
    fn prime(&mut self, p: &[u32], k: usize, sys: usize) -> (usize, usize) {
        self.get().prime(p, k, sys)
    }
    fn park(&mut self, toks: &[u32]) {
        if let Some(mut m) = self.0.take() {
            m.park(toks);
            drop(m);
            crate::up::lock(false);
            eprintln!("gall: unloaded");
        }
    }
    fn loaded(&self) -> bool {
        self.0.is_some()
    }
}

// ---- json

#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

static NULL: J = J::Null;

macro_rules! from {
    ($($t:ty => $f:expr),*) => {$(
        impl From<$t> for J {
            fn from(v: $t) -> J { $f(v) }
        }
    )*};
}
from!(&str => |v: &str| J::Str(v.into()), String => J::Str, f64 => J::Num,
    usize => |v| J::Num(v as f64), bool => J::Bool, Vec<J> => J::Arr);

macro_rules! obj {
    ($($k:expr => $v:expr),* $(,)?) => {
        J::Obj(vec![$(($k.to_string(), J::from($v))),*])
    };
}

impl std::ops::Index<&str> for J {
    type Output = J;
    fn index(&self, k: &str) -> &J {
        let J::Obj(v) = self else { return &NULL };
        v.iter().find(|x| x.0 == k).map_or(&NULL, |x| &x.1)
    }
}

impl fmt::Display for J {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.dump(false))
    }
}

struct P<'a>(&'a [u8], usize);

impl P<'_> {
    fn eat(&mut self, s: &str) -> bool {
        while self.0.get(self.1).is_some_and(u8::is_ascii_whitespace) {
            self.1 += 1;
        }
        let ok = self.0[self.1..].starts_with(s.as_bytes());
        self.1 += if ok { s.len() } else { 0 };
        ok
    }
    fn hex(&mut self) -> Option<u32> {
        let s = std::str::from_utf8(self.0.get(self.1..self.1 + 4)?).ok()?;
        self.1 += 4;
        u32::from_str_radix(s, 16).ok()
    }
    fn string(&mut self) -> Option<String> {
        let mut o = vec![];
        loop {
            let mut c = *self.0.get(self.1)?;
            self.1 += 1;
            match c {
                b'"' => return String::from_utf8(o).ok(),
                b'\\' => {
                    c = *self.0.get(self.1)?;
                    self.1 += 1;
                    let c = match c {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        b'b' => '\x08',
                        b'f' => '\x0c',
                        b'u' => {
                            let mut u = self.hex()?;
                            let hi = (0xd800..0xdc00).contains(&u);
                            if hi && self.0[self.1..].starts_with(b"\\u") {
                                self.1 += 2;
                                let lo = self.hex()?.wrapping_sub(0xdc00);
                                u = 0x10000 + ((u - 0xd800) << 10) + lo;
                            }
                            char::from_u32(u).unwrap_or('\u{fffd}')
                        }
                        c => c as char,
                    };
                    o.extend(c.encode_utf8(&mut [0; 4]).bytes());
                }
                c => o.push(c),
            }
        }
    }
    fn val(&mut self) -> Option<J> {
        Some(if self.eat("null") {
            J::Null
        } else if self.eat("true") {
            J::Bool(true)
        } else if self.eat("false") {
            J::Bool(false)
        } else if self.eat("\"") {
            J::Str(self.string()?)
        } else if self.eat("[") {
            let mut v = vec![];
            while !self.eat("]") {
                if !v.is_empty() && !self.eat(",") {
                    return None;
                }
                v.push(self.val()?);
            }
            J::Arr(v)
        } else if self.eat("{") {
            let mut v = vec![];
            while !self.eat("}") {
                if !v.is_empty() && !self.eat(",") || !self.eat("\"") {
                    return None;
                }
                let k = self.string()?;
                if !self.eat(":") {
                    return None;
                }
                v.push((k, self.val()?));
            }
            J::Obj(v)
        } else {
            let a = self.1;
            while self.0.get(self.1).is_some_and(|c| b"+-.eE".contains(c)
                || c.is_ascii_digit())
            {
                self.1 += 1;
            }
            let s = std::str::from_utf8(&self.0[a..self.1]).ok()?;
            J::Num(s.parse().ok()?)
        })
    }
}

impl J {
    pub fn parse(b: &[u8]) -> Option<J> {
        let mut p = P(b, 0);
        let v = p.val()?;
        (!p.eat("\0") && p.1 == b.len()).then_some(v)
    }
    fn s(&self) -> Option<&str> {
        if let J::Str(s) = self { Some(s) } else { None }
    }
    fn f(&self) -> Option<f64> {
        if let J::Num(n) = self { Some(*n) } else { None }
    }
    fn a(&self) -> &[J] {
        if let J::Arr(v) = self { v } else { &[] }
    }
    // jinja truth
    fn t(&self) -> bool {
        match self {
            J::Null => false,
            J::Bool(b) => *b,
            J::Num(n) => *n != 0.0,
            J::Str(s) => !s.is_empty(),
            J::Arr(v) => !v.is_empty(),
            J::Obj(v) => !v.is_empty(),
        }
    }
    fn or(&self, d: J) -> J {
        if self.t() { self.clone() } else { d }
    }
    fn put(&self, o: &mut String, sort: bool) {
        match self {
            J::Null => o.push_str("null"),
            J::Bool(b) => write!(o, "{b}").unwrap(),
            J::Num(n) => write!(o, "{n}").unwrap(),
            J::Str(s) => {
                o.push('"');
                for c in s.chars() {
                    match c {
                        '"' => o.push_str("\\\""),
                        '\\' => o.push_str("\\\\"),
                        '\n' => o.push_str("\\n"),
                        '\r' => o.push_str("\\r"),
                        '\t' => o.push_str("\\t"),
                        '\x08' => o.push_str("\\b"),
                        '\x0c' => o.push_str("\\f"),
                        c if c < ' ' => {
                            write!(o, "\\u{:04x}", c as u32).unwrap()
                        }
                        c => o.push(c),
                    }
                }
                o.push('"');
            }
            J::Arr(v) => {
                o.push('[');
                for (i, x) in v.iter().enumerate() {
                    o.push_str(if i > 0 { "," } else { "" });
                    x.put(o, sort);
                }
                o.push(']');
            }
            J::Obj(v) => {
                let mut v: Vec<_> = v.iter().collect();
                if sort {
                    v.sort_by(|a, b| a.0.cmp(&b.0));
                }
                o.push('{');
                for (i, (k, x)) in v.into_iter().enumerate() {
                    o.push_str(if i > 0 { "," } else { "" });
                    J::from(k.as_str()).put(o, sort);
                    o.push(':');
                    x.put(o, sort);
                }
                o.push('}');
            }
        }
    }
    fn dump(&self, sort: bool) -> String {
        let mut o = String::new();
        self.put(&mut o, sort);
        o
    }
}

// ---- chat template (tokenizer.chat_template of the gguf, by hand)

const END: &str = "<|end_message|>";

fn effort(e: &J) -> Result<String, String> {
    let map = [("none", 0.0), ("minimal", 0.1), ("low", 0.2), ("medium", 0.7),
        ("high", 0.9), ("xhigh", 0.99), ("max", 0.99)];
    let n = match e {
        J::Str(s) => map
            .iter()
            .find(|m| m.0 == s.trim())
            .ok_or(format!("Unknown reasoning_effort: {s}"))?
            .1,
        J::Num(n) => *n,
        _ => 0.9,
    };
    if !(0.0..=0.99).contains(&n) {
        return Err("reasoning_effort must be in [0.0, 0.99]".into());
    }
    Ok(format!(
        "<|message_system|><|content_text|>Thinking effort level: {n}{END}"
    ))
}

// r: the request; eff: the reasoning effort if the request has none
// the prompt, and the length of its system region
fn prompt(r: &J, eff: &J) -> Result<(String, usize), String> {
    let (mut o, mut sys) = (String::new(), None);
    let tools = r["tools"].a();
    if !tools.is_empty() {
        let spec = |t: &J| {
            let f = if t["function"] == NULL { t } else { &t["function"] };
            obj!["description" => f["description"].or("".into()),
                "name" => f["name"].clone(),
                "parameters" => f["parameters"].or(obj![]),
                "type" => t["type"].or("function".into())]
        };
        let specs = J::Arr(tools.iter().map(spec).collect());
        o += "<|message_system|>tool_declare<|content_xml|>";
        o += &(specs.dump(true) + END);
    }
    let e = [&r["reasoning_effort"], &r["chat_template_kwargs"]["reasoning_effort"]];
    let e = e.into_iter().find(|e| **e != NULL).unwrap_or(eff);
    let mut eff = Some(effort(e)?);
    let msgs = r["messages"].a();
    for m in msgs {
        let role = m["role"].s().unwrap_or("");
        let mut h = match role {
            "user" => "<|message_user|>",
            "assistant" => "<|message_model|>",
            "system" => "<|message_system|>",
            "tool" => "<|message_tool|>",
            _ => return Err(format!("Unknown message role: {role}")),
        }
        .to_string();
        if role != "system" {
            o += &eff.take().unwrap_or_default();
            sys.get_or_insert(o.len());
        }
        let asst = role == "assistant";
        let calls = if asst { m["tool_calls"].a() } else { &[] };
        if role == "tool" && !m["name"].t() && m["tool_call_id"].t() {
            for tc in msgs.iter().flat_map(|p| p["tool_calls"].a()) {
                if tc["id"] == m["tool_call_id"] {
                    h += tc["function"]["name"].s().unwrap_or("");
                }
            }
        } else if asst || role == "tool" {
            h += m["name"].s().unwrap_or("");
        }
        let rc = m["reasoning_content"].s().filter(|_| asst).unwrap_or("");
        if !rc.is_empty() {
            o += &format!("{h}<|content_thinking|>{rc}{END}");
        }
        let mut text = |t: &str| o += &format!("{h}<|content_text|>{t}{END}");
        match &m["content"] {
            J::Str(c) => {
                if !c.is_empty() || calls.is_empty() && rc.is_empty() {
                    text(c)
                }
            }
            c => {
                for p in c.a() {
                    match (p, p["type"].s()) {
                        (J::Str(s), _) => text(s),
                        (_, None | Some("text" | "input_text")) => {
                            text(p["text"].s().unwrap_or(""))
                        }
                        (_, Some(t)) => {
                            return Err(format!(
                                "Unsupported content part type: {t}"
                            ))
                        }
                    }
                }
            }
        }
        for tc in calls {
            let f = &tc["function"];
            let bad = "tool call arguments must be an object";
            let n = f["name"].s().ok_or("tool call needs a function name")?;
            let args = match &f["arguments"] {
                J::Str(s) => J::parse(s.as_bytes()).ok_or(bad)?,
                a => a.clone(),
            };
            let J::Obj(mut kv) = args.or(obj![]) else { return Err(bad.into()) };
            kv.sort_by(|a, b| a.0.cmp(&b.0));
            let kv: Vec<_> = kv
                .iter()
                .map(|(k, v)| format!("{}:{v}", J::from(k.as_str())))
                .collect();
            o += &format!(
                "<|message_model|>{n}<|content_invoke_tool_json|>\
                 {{\"name\":{},\"args\":{{{}}}}}{END}",
                J::from(n),
                kv.join(",")
            );
        }
        if asst {
            o += "<|content_model_end_sampling|>";
        }
    }
    let o = o + &eff.unwrap_or_default();
    let n = sys.unwrap_or(o.len());
    Ok((o, n))
}

// ---- sampling: top_k, top_p, min_p, temperature, in llama.cpp's order

struct Sp {
    temp: f64,
    top_k: f64,
    top_p: f64,
    min_p: f64,
}

fn probs(c: &[(f32, u32)], t: f32) -> Vec<f32> {
    let m = c.iter().fold(f32::MIN, |m, x| m.max(x.0));
    let p: Vec<f32> = c.iter().map(|x| ((x.0 - m) / t).exp()).collect();
    let z: f32 = p.iter().sum();
    p.iter().map(|x| x / z).collect()
}

fn sample(l: &[f32], p: &Sp, rng: &mut u64) -> u32 {
    let mut c: Vec<(f32, u32)> = l.iter().copied().zip(0..).collect();
    let k = p.top_k as usize;
    if k > 0 && k < c.len() {
        c.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        c.truncate(k);
    }
    if p.temp <= 0.0 {
        // the first maximum
        return c.iter().fold(c[0], |m, &x| if x.0 > m.0 { x } else { m }).1;
    }
    if p.top_p < 1.0 {
        let pr = probs(&c, 1.0);
        let by = |a: &usize, b: &usize| pr[*b].total_cmp(&pr[*a]);
        let mut ix: Vec<usize> = (0..c.len()).collect();
        // the head nearly always holds top_p: sort it alone first
        let h = ix.len().min(1024);
        if h < ix.len() {
            ix.select_nth_unstable_by(h - 1, by);
        }
        ix[..h].sort_unstable_by(by);
        if (ix[..h].iter().map(|&i| pr[i]).sum::<f32>() as f64) < p.top_p {
            ix.sort_unstable_by(by);
        }
        let mut s = 0.0;
        let n = ix.iter().position(|&i| {
            s += pr[i] as f64;
            s >= p.top_p
        });
        ix.truncate(n.map_or(ix.len(), |n| n + 1));
        c = ix.iter().map(|&i| c[i]).collect();
    }
    if p.min_p > 0.0 {
        let m = c.iter().fold(f32::MIN, |m, x| m.max(x.0));
        c.retain(|x| x.0 >= m + p.min_p.ln() as f32);
    }
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let mut u = (*rng >> 40) as f32 / (1u64 << 24) as f32;
    let pr = probs(&c, p.temp as f32);
    let i = pr.iter().position(|&x| {
        u -= x;
        u < 0.0
    });
    c[i.unwrap_or(c.len() - 1)].1
}

// ---- http

fn head(w: &mut impl Write, code: u16, kind: &str, more: &str) -> io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {code} OK\r\nContent-Type: {kind}\r\nConnection: close\r\n\
         Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n\
         {more}\r\n"
    )
}

fn reply(w: &mut impl Write, code: u16, j: &J) -> io::Result<()> {
    let b = j.to_string();
    head(w, code, "application/json", &format!("Content-Length: {}\r\n", b.len()))?;
    w.write_all(b.as_bytes())
}

fn sse(w: &mut impl Write, j: &J) -> io::Result<()> {
    write!(w, "data: {j}\n\n")?;
    w.flush()
}

fn err(m: &str) -> J {
    obj!["error" => obj!["message" => m, "type" => "invalid_request_error"]]
}

// method, path, body, authorization
fn request(c: &TcpStream) -> Option<(String, String, Vec<u8>, String)> {
    let (mut r, mut l, mut n) = (BufReader::new(c), String::new(), 0);
    let mut auth = String::new();
    r.read_line(&mut l).ok()?;
    let mut it = l.split_whitespace().map(str::to_string);
    let mp = (it.next()?, it.next()?);
    loop {
        l.clear();
        r.read_line(&mut l).ok()?;
        let h = l.trim().to_ascii_lowercase();
        if h.is_empty() {
            break;
        } else if let Some(v) = h.strip_prefix("content-length:") {
            n = v.trim().parse().ok()?;
        } else if let Some(v) = h.strip_prefix("authorization:") {
            auth = v.trim().into();
        } else if h.starts_with("expect:") {
            (&*c).write_all(b"HTTP/1.1 100 Continue\r\n\r\n").ok()?;
        }
    }
    let mut b = vec![0; n];
    r.read_exact(&mut b).ok()?;
    Some((mp.0, mp.1, b, auth))
}

// the whole utf-8 of b past *sent, less the last hold bytes
fn part(b: &[u8], sent: &mut usize, hold: usize) -> Option<String> {
    let e = b.len().saturating_sub(hold).max(*sent);
    let e = match std::str::from_utf8(&b[*sent..e]) {
        Err(x) if x.error_len().is_none() => *sent + x.valid_up_to(),
        _ => e,
    };
    let s = String::from_utf8_lossy(&b[*sent..e]).into_owned();
    *sent = e;
    (!s.is_empty()).then_some(s)
}

struct Srv {
    lm: Box<dyn Lm>,
    tok: Tok,
    fed: Vec<u32>, // the tokens in lm
    ctx: usize,
    effort: J,
    name: String,
    auth: Option<String>, // "bearer KEY", lowercase as request() reads it
    sp: Sp,
    t: [u32; 5], // model, thinking, text, tool json, end
}

impl Srv {
    fn chat(&mut self, r: &J, w: &mut TcpStream) -> io::Result<()> {
        let (p, sys) = match prompt(r, &self.effort) {
            Ok((p, n)) => (self.tok.enc(&p, true), self.tok.enc(&p[..n], true)),
            Err(e) => return reply(w, 400, &err(&e)),
        };
        let sys = if p.starts_with(&sys) { sys.len() } else { 0 };
        if p.is_empty() || p.len() >= self.ctx {
            let e = format!("{} prompt tokens, context {}", p.len(), self.ctx);
            return reply(w, 400, &err(&e));
        }
        let num = |k: &str, d: f64| r[k].f().unwrap_or(d);
        let sp = Sp {
            temp: num("temperature", self.sp.temp),
            top_k: num("top_k", self.sp.top_k),
            top_p: num("top_p", self.sp.top_p),
            min_p: num("min_p", self.sp.min_p),
        };
        let max = r["max_completion_tokens"].f().or(r["max_tokens"].f());
        let max = max.filter(|&n| n >= 0.0).map_or(usize::MAX, |n| n as usize);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let mut rng = num("seed", now.as_nanos() as f64) as u64 | 1 << 40;
        let stops: Vec<&[u8]> = match &r["stop"] {
            J::Str(s) => vec![s.as_bytes()],
            a => a.a().iter().filter_map(J::s).map(str::as_bytes).collect(),
        };
        let stops: Vec<_> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let hold = stops.iter().map(|s| s.len()).max().unwrap_or(1) - 1;
        let stream = r["stream"].t();
        let (id, name) = (format!("chatcmpl-{}", now.as_nanos()), &self.name);
        let chunk = |delta: J, fin: J| {
            obj!["id" => id.as_str(), "object" => "chat.completion.chunk",
                "created" => now.as_secs() as usize, "model" => name.as_str(),
                "choices" => vec![obj!["index" => 0, "delta" => delta,
                    "finish_reason" => fin]]]
        };
        if stream {
            head(w, 200, "text/event-stream", "Cache-Control: no-cache\r\n")?;
            sse(w, &chunk(obj!["role" => "assistant"], J::Null))?;
        }

        let t0 = Instant::now();
        if self.lm.len() != self.fed.len() {
            self.lm.reset();
            self.fed.clear();
        }
        let same = self.fed.iter().zip(&p).take_while(|(a, b)| a == b).count();
        let mut k = same.min(p.len() - 1);
        if k < self.fed.len() && !self.lm.truncate(k) {
            self.lm.reset();
            k = 0;
        }
        let (at, snap) = self.lm.prime(&p, k, sys);
        let k = k.max(snap); // not fed for this request
        self.fed.clear();
        self.fed.extend(&p);
        let mut lg = self.lm.feed(&p[at..]);
        let t1 = Instant::now();

        let [model, think, text, tool, end] = self.t;
        let (mut st, mut n, mut sent) = (0, 0, [0; 2]);
        let mut b = [vec![], vec![], vec![], vec![]]; // thinking, text, name, tool
        let mut calls = vec![];
        let fin = loop {
            if n == max || self.fed.len() >= self.ctx {
                break "length";
            }
            let t = sample(lg, &sp, &mut rng);
            n += 1;
            if t == self.tok.eos {
                break "stop";
            }
            if t == end && st == tool {
                let j = J::parse(&b[3]).unwrap_or(J::Null);
                let s = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
                let (name, raw) = (s(&b[2]), s(&b[3]));
                let args = &j["args"];
                calls.push(obj!["index" => calls.len(),
                    "id" => format!("call_{}{}", now.as_micros(), calls.len()),
                    "type" => "function",
                    "function" => obj![
                        "name" => j["name"].s().unwrap_or(name.trim()),
                        "arguments" => if *args == NULL { raw }
                            else { args.to_string() }]]);
                b[3].clear();
                if stream {
                    let c = calls[calls.len() - 1..].to_vec();
                    sse(w, &chunk(obj!["tool_calls" => c], J::Null))?;
                }
            }
            if self.tok.ctl[t as usize] {
                st = if [model, think, text, tool].contains(&t) { t } else { 0 };
                if t == model {
                    b[2].clear();
                }
            } else {
                let i = [think, text, model, tool].iter().position(|&x| x == st);
                b[i.unwrap_or(1)].extend(&self.tok.txt[t as usize]);
            }
            let find = |s: &&[u8]| b[1][sent[1]..].windows(s.len()).position(|w| w == *s);
            if let Some(i) = stops.iter().filter_map(find).min() {
                b[1].truncate(sent[1] + i);
                break "stop";
            }
            for (i, key) in ["reasoning_content", "content"].iter().enumerate() {
                if !stream {
                    break;
                }
                if let Some(s) = part(&b[i], &mut sent[i], hold * i) {
                    sse(w, &chunk(obj![key => s], J::Null))?;
                }
            }
            self.fed.push(t);
            lg = self.lm.step(t);
        };
        let fin = if calls.is_empty() { fin } else { "tool_calls" };

        let (dp, dg) = (t1 - t0, t1.elapsed());
        let (np, ms) = (p.len() - k, |d: Duration| d.as_secs_f64() * 1e3);
        let rate = |n: usize, d: Duration| n as f64 / d.as_secs_f64().max(1e-9);
        eprintln!(
            "gall: prompt {} ({k} cached, snap {snap}) {:.1} t/s, first token \
             {:.2} s, predicted {n} {:.1} t/s",
            p.len(),
            rate(np, dp),
            dp.as_secs_f64(),
            rate(n, dg)
        );
        let usage = obj!["prompt_tokens" => p.len(), "completion_tokens" => n,
            "total_tokens" => p.len() + n];
        let timings = obj!["cache_n" => k, "prompt_n" => np,
            "prompt_ms" => ms(dp), "prompt_per_second" => rate(np, dp),
            "predicted_n" => n, "predicted_ms" => ms(dg),
            "predicted_per_second" => rate(n, dg)];
        if stream {
            for (i, key) in ["reasoning_content", "content"].iter().enumerate() {
                if let Some(s) = part(&b[i], &mut sent[i], 0) {
                    sse(w, &chunk(obj![key => s], J::Null))?;
                }
            }
            let J::Obj(mut c) = chunk(obj![], fin.into()) else { unreachable!() };
            c.push(("timings".into(), timings));
            sse(w, &J::Obj(c))?;
            if r["stream_options"]["include_usage"].t() {
                let J::Obj(mut c) = chunk(obj![], J::Null) else { unreachable!() };
                c[4].1 = vec![].into();
                c.push(("usage".into(), usage));
                sse(w, &J::Obj(c))?;
            }
            return w.write_all(b"data: [DONE]\n\n");
        }
        let s = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
        let mut m = vec![("role".to_string(), "assistant".into())];
        m.push(("content".into(), s(&b[1]).into()));
        if !b[0].is_empty() {
            m.push(("reasoning_content".into(), s(&b[0]).into()));
        }
        if !calls.is_empty() {
            m.push(("tool_calls".into(), calls.into()));
        }
        let c = obj!["index" => 0, "message" => J::Obj(m), "finish_reason" => fin];
        reply(w, 200, &obj!["id" => id.as_str(), "object" => "chat.completion",
            "created" => now.as_secs() as usize, "model" => name.as_str(),
            "choices" => vec![c], "usage" => usage, "timings" => timings])
    }

    fn conn(&mut self, mut c: TcpStream) -> io::Result<()> {
        c.set_nodelay(true)?;
        c.set_read_timeout(Some(Duration::from_secs(30)))?;
        let Some((m, p, body, a)) = request(&c) else { return Ok(()) };
        let p = p.trim_start_matches("/v1");
        match (m.as_str(), p) {
            ("OPTIONS", _) => head(&mut c, 204, "text/plain", ""),
            ("GET", "/health") => reply(&mut c, 200, &obj!["status" => "ok",
                "loaded" => self.lm.loaded()]),
            _ if self.auth.as_ref().is_some_and(|k| *k != a) => {
                reply(&mut c, 401, &err("invalid api key"))
            }
            ("GET", "/models") => {
                let m = obj!["id" => self.name.as_str(), "object" => "model",
                    "owned_by" => "gall"];
                reply(&mut c, 200, &obj!["object" => "list", "data" => vec![m]])
            }
            ("POST", "/chat/completions") => match J::parse(&body) {
                Some(r) => self.chat(&r, &mut c),
                None => reply(&mut c, 400, &err("the body is not json")),
            },
            _ => reply(&mut c, 404, &err("not found")),
        }
    }
}

// answers every prompt with the same line: for tests without a model
struct Dummy(Vec<u32>, usize, Vec<f32>, usize);

impl Lm for Dummy {
    fn reset(&mut self) {
        self.1 = 0;
    }
    fn step(&mut self, t: u32) -> &[f32] {
        self.1 += 1;
        let hit = t == self.0[self.3];
        self.3 = if hit { (self.3 + 1) % self.0.len() } else { 0 };
        self.2.fill(0.0);
        self.2[self.0[self.3] as usize] = 30.0;
        &self.2
    }
    fn len(&self) -> usize {
        self.1
    }
    fn truncate(&mut self, n: usize) -> bool {
        self.1 = n;
        true
    }
}

// the engine as an Lm; a checkpoint after each feed (the prompt)
struct Ink(crate::model::Model, Vec<f32>, crate::model::snap::Snaps);

impl Lm for Ink {
    fn reset(&mut self) {
        self.0.reset()
    }
    fn step(&mut self, t: u32) -> &[f32] {
        self.0.step(t);
        self.1 = self.0.logits();
        &self.1
    }
    fn len(&self) -> usize {
        self.0.pos
    }
    fn truncate(&mut self, n: usize) -> bool {
        self.0.truncate(n)
    }
    fn feed(&mut self, toks: &[u32]) -> &[f32] {
        self.0.feed(toks);
        self.0.save();
        self.1 = self.0.logits();
        &self.1
    }
    fn prime(&mut self, p: &[u32], k: usize, sys: usize) -> (usize, usize) {
        self.2.prime(&mut self.0, p, k, sys)
    }
    fn park(&mut self, toks: &[u32]) {
        let t = Instant::now();
        if toks.len() < 512 {
            return;
        } else if let Err(e) = self.2.write(&self.0, toks) {
            return eprintln!("gall: snap {}: {e}", toks.len());
        }
        let s = t.elapsed().as_secs_f64();
        eprintln!("gall: snap {} written in {s:.2} s", toks.len());
    }
}

pub fn main(a: &[String]) {
    let opt = |k: &str, d: &str| {
        let i = a.iter().position(|x| x == k);
        let v = i.and_then(|i| a.get(i + 1)).filter(|v| !v.starts_with("--"));
        v.map_or(d, |s| s).to_string()
    };
    let g = crate::gguf::open(&a[0]);
    let tok = Tok::new(&g);
    if a.get(1).is_some_and(|x| x == "--prompt") {
        // gall serve <gguf> --prompt <request.json>: the rendered prompt
        let r = J::parse(&std::fs::read(&a[2]).unwrap()).unwrap();
        return print!("{}", prompt(&r, &"none".into()).unwrap().0);
    }
    let ctx: usize = opt("--ctx", "102400").parse().unwrap();
    let say = "<|message_model|><|content_text|>hello from gall\
               <|end_message|><|content_model_end_sampling|>";
    let load: Load = if a.iter().any(|x| x == "--dummy") {
        let (say, n) = (tok.enc(&opt("--dummy", say), true), tok.txt.len());
        Box::new(move || Box::new(Dummy(say.clone(), 0, vec![0.0; n], 0)))
    } else {
        crate::up::guard();
        let p = a[0].clone();
        Box::new(move || {
            crate::up::lock(true);
            let m = crate::model::Model::open(&p, ctx);
            let s = crate::model::snap::Snaps::new(&p, &m);
            Box::new(Ink(m, vec![], s))
        })
    };
    let lm = Box::new(Lazy(Some(load()), load));
    let kv = |k: &str, d: f64| {
        let k = format!("general.sampling.{k}");
        if g.kv.contains_key(k.as_str()) { g.f(&k) } else { d }
    };
    let sp = Sp {
        temp: kv("temp", 1.0),
        top_k: kv("top_k", 0.0),
        top_p: kv("top_p", 1.0),
        min_p: kv("min_p", 0.0),
    };
    let t = ["message_model", "content_thinking", "content_text",
        "content_invoke_tool_json", "end_message"]
        .map(|s| tok.id(&format!("<|{s}|>")));
    let port: u16 = opt("--port", "9940").parse().unwrap();
    let l = TcpListener::bind(("127.0.0.1", port)).unwrap();
    let mut s = Srv {
        lm,
        tok,
        fed: vec![],
        ctx,
        effort: opt("--effort", "none").into(),
        name: opt("--alias", "inkling-2-mini"),
        auth: a.iter().position(|x| x == "--api-key-file").map(|_| {
            let k = std::fs::read_to_string(opt("--api-key-file", ""));
            format!("bearer {}", k.unwrap().trim().to_ascii_lowercase())
        }),
        sp,
        t,
    };
    eprintln!("gall: http://127.0.0.1:{port}/v1, context {ctx}");
    let idle: i32 = opt("--idle-unload", "0").parse().unwrap();
    let mut pf = libc::pollfd { fd: l.as_raw_fd(), events: libc::POLLIN,
        revents: 0 };
    loop {
        let t = if idle > 0 && s.lm.loaded() { idle * 1000 } else { -1 };
        if unsafe { libc::poll(&mut pf, 1, t) } == 0 {
            let n = if s.lm.len() == s.fed.len() { s.fed.len() } else { 0 };
            s.lm.park(&s.fed[..n]);
            s.fed.clear();
        } else if let Err(e) = l.accept().and_then(|c| s.conn(c.0)) {
            eprintln!("gall: {e}");
        }
    }
}
