// gall up [ctx] | down | log | model | test: the server as a background
// process (port 9940; log and pid in ~/.cache/gall), the model path, and a
// greedy self-test against 64 reference tokens from llama.cpp
use std::os::{fd::AsRawFd, unix::process::CommandExt};
use std::process::{exit, Command, Stdio};
use std::{env, fs, net::TcpStream, path::PathBuf, thread, time::Duration};

const REPO: &str = "unsloth/Inkling-2-Mini-GGUF";
const FILE: &str = "Inkling-2-Mini-UD-IQ3_XXS.gguf";

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
    if !p.exists() {
        eprintln!("gall: fetching {FILE} (11.7GB) to {}", p.display());
        let part = p.with_extension("part");
        let url = format!("https://huggingface.co/{REPO}/resolve/main/{FILE}");
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

// one model fits on the gpu: take the machine lock (./wd holds it already
// when WDL is set), and leave before memory runs out
pub fn guard() {
    if env::var("WDL").is_err() {
        let f = fs::File::create("/tmp/gall.lock").unwrap();
        let l = libc::LOCK_EX | libc::LOCK_NB;
        if unsafe { libc::flock(f.as_raw_fd(), l) } != 0 {
            eprintln!("gall: /tmp/gall.lock is held, a model is loaded");
            exit(1)
        }
        std::mem::forget(f);
    }
    thread::spawn(|| loop {
        if sys("kern.memorystatus_vm_pressure_level") >= 4
            || sys("kern.memorystatus_level") < 6
        {
            eprintln!("gall: memory is short, leaving");
            exit(1)
        }
        thread::sleep(Duration::from_secs(1));
    });
}

pub fn main(a: &[String]) {
    let (log, pid) = (cache().join("serve.log"), cache().join("serve.pid"));
    let up = || TcpStream::connect(("127.0.0.1", 9940)).is_ok();
    match a[0].as_str() {
        "model" => println!("{}", model()),
        "log" => panic!("{}", Command::new("tail").arg("-f").arg(log).exec()),
        "down" => {
            if let Ok(p) = fs::read_to_string(&pid) {
                unsafe { libc::kill(p.parse().unwrap(), libc::SIGTERM) };
                fs::remove_file(pid).ok();
            }
        }
        _ if up() => {}
        _ => {
            let (m, f) = (model(), fs::File::create(&log).unwrap());
            let ctx = a.get(1).map_or("32768", |s| s);
            let mut c = Command::new(env::current_exe().unwrap())
                .args(["serve", &m, "--ctx", ctx])
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
