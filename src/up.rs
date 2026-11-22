// gall up [ctx] [serve flags] | down | log | model | test: the server as a background
// process (port 9940; log and pid in ~/.cache/gall), the model path, and a
// greedy self-test against 64 reference tokens from llama.cpp
use std::os::{fd::IntoRawFd, unix::process::CommandExt};
use std::sync::atomic::{AtomicI32, Ordering::Relaxed};
use std::process::{exit, Command, Stdio};
use std::{env, fs, net::TcpStream, path::PathBuf, thread, time::Duration};

const REPO: &str = "unsloth/Inkling-2-Mini-GGUF";
const FILE: &str = "Inkling-2-Mini-UD-IQ3_XXS.gguf";
pub const MMPROJ: &str = "mmproj-F16.gguf"; // the vision encoder
pub const DRAFT: &str = "dspark-Inkling-2-Mini-Q8_0.gguf"; // the drafter

pub fn cache() -> PathBuf {
    let d = PathBuf::from(env::var("HOME").unwrap()).join(".cache/gall");
    fs::create_dir_all(&d).unwrap();
    d
}

// $GALL_MODEL, the hugging face cache, ~/.cache/gall, else a download
pub fn model() -> String {
    if let Ok(m) = env::var("GALL_MODEL") {
        return m;
    }
    let hub = format!("huggingface/hub/models--{}/snapshots",
                      REPO.replace('/', "--"));
    let hub = cache().parent().unwrap().join(hub);
    let hit = fs::read_dir(hub).ok().into_iter().flatten()
        .flatten().map(|d| d.path().join(FILE)).find(|p| p.exists());
    let p = hit.unwrap_or(cache().join(FILE));
    for (f, gb) in [(FILE, 11.7), (MMPROJ, 0.6), (DRAFT, 0.4)] {
        let p = p.with_file_name(f);
        if p.exists() {
            continue;
        }
        eprintln!("gall: fetching {f} ({gb}GB) to {}", p.display());
        let part = p.with_extension("part");
        let url = format!("https://huggingface.co/{REPO}/resolve/main/{f}");
        let c = Command::new("curl").args(["-fL", "-C", "-", "-o"])
            .arg(&part).arg(url).status();
        if !c.is_ok_and(|s| s.success()) {
            eprintln!("gall: download failed; run again to resume");
            exit(1)
        }
        fs::rename(part, &p).unwrap();
    }
    p.to_str().unwrap().into()
}

fn sys(k: &str) -> i32 {
    let (mut v, mut n) = (0i32, 4usize);
    let k = std::ffi::CString::new(k).unwrap();
    let p = &mut v as *mut i32 as *mut libc::c_void;
    unsafe { libc::sysctlbyname(k.as_ptr(), p, &mut n, 0 as _, 0) };
    v
}

// one model fits on the gpu: open the machine lock (./wd holds it already
// when WDL is set), park at memory pressure (warn), and leave before memory
// runs out
static FD: AtomicI32 = AtomicI32::new(-1);
pub fn guard(park: impl Fn() + Send + 'static) {
    if env::var("WDL").is_err() {
        let f = fs::File::create("/tmp/gall.lock").unwrap();
        FD.store(f.into_raw_fd(), Relaxed);
    }
    thread::spawn(move || loop {
        let l = sys("kern.memorystatus_vm_pressure_level");
        if l >= 4 || sys("kern.memorystatus_level") < 6 {
            eprintln!("gall: memory is short, leaving");
            exit(1)
        } else if l >= 2 {
            park()
        }
        thread::sleep(Duration::from_secs(1));
    });
}

// take the lock while a model is loaded, waiting for it; false gives it back
pub fn lock(on: bool) {
    let f = FD.load(Relaxed);
    let op = |o| f >= 0 && unsafe { libc::flock(f, o) } != 0;
    if !on {
        op(libc::LOCK_UN);
    } else if op(libc::LOCK_EX | libc::LOCK_NB) {
        // /tmp/gall.want asks ./wd runs to let the server go first
        eprintln!("gall: /tmp/gall.lock is held, waiting");
        fs::write("/tmp/gall.want", std::process::id().to_string()).ok();
        op(libc::LOCK_EX);
        fs::remove_file("/tmp/gall.want").ok();
    }
}

pub fn main(a: &[String]) {
    let (log, pid) = (cache().join("serve.log"), cache().join("serve.pid"));
    let up = || TcpStream::connect(("127.0.0.1", 9940)).is_ok();
    match a[0].as_str() {
        "model" => println!("{}", model()),
        "log" => panic!("{}", Command::new("tail").arg("-f").arg(log).exec()),
        // the server parks, then exits: wait for it, so an up can follow
        "down" => {
            if let Ok(p) = fs::read_to_string(&pid) {
                let p = p.parse().unwrap();
                unsafe { libc::kill(p, libc::SIGTERM) };
                while unsafe { libc::kill(p, 0) } == 0 {
                    thread::sleep(Duration::from_millis(100));
                }
                fs::remove_file(pid).ok();
            }
        }
        _ if up() => {}
        _ => {
            let (m, f) = (model(), fs::File::create(&log).unwrap());
            let ctx = a.get(1).map_or("102400", |s| s);
            let mut c = Command::new(env::current_exe().unwrap())
                .args(["serve", &m, "--ctx", ctx]).args(a.iter().skip(2))
                .stdin(Stdio::null()).stdout(f.try_clone().unwrap())
                .stderr(f).process_group(0).spawn().unwrap();
            fs::write(pid, c.id().to_string()).unwrap();
            while !up() {
                if c.try_wait().unwrap().is_some() {
                    eprint!("{}", fs::read_to_string(&log).unwrap());
                    exit(1)
                }
                thread::sleep(Duration::from_millis(100));
            }
            println!("gall: http://127.0.0.1:9940/v1, context {ctx}");
        }
    }
}
