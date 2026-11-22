// prefix snapshots on disk, so a known prompt prefix is not fed again.
//   let mut s = Snaps::new(path, &m);      // index of ~/.cache/gall/*.snap
//   let (k, r) = s.prime(&mut m, &p, k, sys); // m holds p[..k], r restored
// a file: tag line, n, n token ids, the ring rows past n that a batched
// pass took (Model::top), the conv rings, the last ROWS rows of each
// sliding window ring, rows [0, n) of each full-attention slot.
use super::*;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::{fs, ops::Range, time::SystemTime};

const ROWS: usize = 1024 + 12; // the window, and truncate()'s 12 steps
const MIN: usize = 512; // shortest prefix worth a file
const KEEP: usize = 8; // files, least recently used goes first

pub struct Snaps {
    dir: PathBuf,
    tag: String,
    ix: Vec<(PathBuf, Vec<u32>)>,
}

fn u32s(r: &mut impl Read, n: usize) -> Option<Vec<u32>> {
    let mut b = vec![0; 4 * n];
    r.read_exact(&mut b).ok()?;
    let le = |c: &[u8]| u32::from_le_bytes(c.try_into().unwrap());
    Some(b.chunks(4).map(le).collect())
}

// the tag and tokens of a file, with the reader at the state
fn head(p: &PathBuf, tag: &str) -> Option<(BufReader<fs::File>, Vec<u32>)> {
    let mut r = BufReader::with_capacity(1 << 20, fs::File::open(p).ok()?);
    let mut t = vec![0; tag.len()];
    r.read_exact(&mut t).ok().filter(|_| t == tag.as_bytes())?;
    let n = u32s(&mut r, 1)?[0] as usize;
    let toks = u32s(&mut r, n)?;
    Some((r, toks))
}

fn common(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

impl Model {
    // halfs per key of a full-attention slot
    fn snap_row(&self) -> usize {
        (self.kvo[6] - self.kvo[5]) / self.n_ctx
    }

    // the cache rows a snapshot at n holds, in halfs
    fn snap_rows(&self, n: usize) -> Vec<Range<usize>> {
        let mut by_slot = visits();
        by_slot.sort_by_key(|v| (v.0 % 6 == 5, v.1));
        let rows = |(i, slot, _): (usize, usize, usize)| {
            let o = self.kvo[slot];
            if i % 6 == 5 {
                return vec![o..o + n * self.snap_row()];
            }
            let r = n.saturating_sub(ROWS)..n;
            r.map(|p| o + p % SW * 2048..o + (p % SW + 1) * 2048).collect()
        };
        by_slot.into_iter().flat_map(rows).collect()
    }
}

impl Snaps {
    // path: the model file, part of the tag with the cache layout
    pub fn new(path: &str, m: &Model) -> Snaps {
        let (md, home) = (fs::metadata(path).unwrap(), std::env::var("HOME"));
        let at = md.modified().unwrap().duration_since(SystemTime::UNIX_EPOCH);
        let tag = format!("gall snap 2 row {} sw {SW} cs {CS} model {} {}\n",
            m.snap_row(), md.len(), at.unwrap().as_secs());
        let dir = PathBuf::from(home.unwrap()).join(".cache/gall");
        fs::create_dir_all(&dir).unwrap();
        let ls = fs::read_dir(&dir).unwrap().flatten().map(|e| e.path());
        let ls = ls.filter(|p| p.extension().is_some_and(|x| x == "snap"));
        let ix: Vec<_> = ls
            .filter_map(|p| head(&p, &tag).map(|h| (p, h.1)))
            .collect();
        eprintln!("gall: {} snapshots in {}", ix.len(), dir.display());
        Snaps { dir, tag, ix }
    }

    // write the state of m, which holds toks
    pub fn write(&mut self, m: &Model, toks: &[u32]) -> std::io::Result<()> {
        let mut h = DefaultHasher::new();
        (&self.tag, toks).hash(&mut h);
        let p = self.dir.join(format!("{:016x}.snap", h.finish()));
        let tmp = p.with_extension("tmp");
        let mut w = BufWriter::with_capacity(1 << 20, fs::File::create(&tmp)?);
        w.write_all(self.tag.as_bytes())?;
        let top = Some(m.top.wrapping_sub(m.pos) as u32).filter(|a| *a <= 8);
        for t in [toks.len() as u32].iter().chain(toks).chain(&[top.unwrap_or(0)]) {
            w.write_all(&t.to_le_bytes())?;
        }
        w.write_all(slice::<u8>(&m.cs))?;
        let kv = slice::<u8>(&m.kv);
        for r in m.snap_rows(toks.len()) {
            w.write_all(&kv[2 * r.start..2 * r.end])?;
        }
        w.into_inner()?.sync_data()?;
        fs::rename(&tmp, &p)?;
        self.ix.retain(|e| e.0 != p);
        self.ix.push((p, toks.to_vec()));
        // drop the least recently used
        let age = |e: &(PathBuf, _)| fs::metadata(&e.0).and_then(|m| m.modified()).ok();
        self.ix.sort_by_key(|e| std::cmp::Reverse(age(e)));
        for e in self.ix.drain(KEEP.min(self.ix.len())..) {
            fs::remove_file(e.0)?;
        }
        Ok(())
    }

    // read snapshot i into m
    fn read(&self, m: &mut Model, i: usize) -> Option<()> {
        let (p, toks) = &self.ix[i];
        let mut r = head(p, &self.tag)?.0;
        m.top = toks.len() + u32s(&mut r, 1)?[0] as usize;
        r.read_exact(slice::<u8>(&m.cs)).ok()?;
        let kv = slice::<u8>(&m.kv);
        for x in m.snap_rows(toks.len()) {
            r.read_exact(&mut kv[2 * x.start..2 * x.end]).ok()?;
        }
        let f = fs::OpenOptions::new().append(true).open(p).ok()?;
        f.set_modified(SystemTime::now()).ok();
        (m.pos, m.ck) = (toks.len(), None);
        Some(())
    }

    // m holds p[..k]. restore the longest stored prefix of p past k (or a
    // file that parts from p in its last 12 tokens, truncated), then feed
    // up to and store: the longest prefix p shares with a stored
    // snapshot, and p[..sys] (the system region). returns the new k and
    // the length restored. p[k..] is never empty
    pub fn prime(&mut self, m: &mut Model, p: &[u32], mut k: usize, sys: usize) -> (usize, usize) {
        let c: Vec<_> = self.ix.iter().map(|e| common(&e.1, p)).collect();
        let n = |i: usize| self.ix[i].1.len();
        let pre = |i: &usize| c[*i] + 12 >= n(*i) && c[*i] < p.len();
        let best = (0..c.len()).filter(pre).max_by_key(|&i| c[i]);
        let mut got = 0;
        if let Some(i) = best.filter(|&i| c[i] > k) {
            let t = std::time::Instant::now();
            if self.read(m, i).is_some() && m.truncate(c[i]) {
                (k, got) = (c[i], c[i]);
                eprintln!("gall: snap {k} read in {:.2} s", t.elapsed().as_secs_f64());
            } else {
                // a short or foreign file: start again
                m.reset();
                k = 0;
            }
        }
        let have: Vec<_> = (0..c.len()).filter(pre).map(|i| c[i]).collect();
        let mut cuts = vec![c.iter().copied().max().unwrap_or(0), sys];
        cuts.sort();
        for n in cuts {
            // not for under half a second of feeding
            if n < MIN || n < k + 64 || n >= p.len() || have.contains(&n) {
                continue;
            }
            m.feed(&p[k..n]);
            k = n;
            let t = std::time::Instant::now();
            match self.write(m, &p[..n]) {
                Ok(()) => eprintln!("gall: snap {n} written in {:.2} s",
                    t.elapsed().as_secs_f64()),
                Err(e) => eprintln!("gall: snap {n}: {e}"),
            }
        }
        (k, got)
    }
}
