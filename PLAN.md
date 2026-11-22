# plan: gall.app, a menu bar endpoint for low-memory macs

goal: a notarized `Gall.app` that a non-developer drags to Applications,
opens, and gets an openai-compatible endpoint at `127.0.0.1:9940/v1`. the
point of difference is the one gall already has: idle and memory-pressure
unloads keep the conversation on disk, so the next message does not process
the prompt again. ollama and lm studio unload and drop the kv cache.

no imessage, no second user, no daemons, no admin prompt. it runs as the
user. `cargo install` and `gall up` stay as they are for terminal users.

## decision: what stays in rust, what goes to swift

- rust keeps the engine and the http server (`serve.rs`), and `up.rs`
  (`up`, `down`, `log`, `model`, `test`). the app drives the same commands,
  so the download, the model lookup and the process handling exist once,
  for both the app and the terminal. a swift rewrite of `up.rs` would
  duplicate them and add nothing: progress comes from file sizes (below).
- swift owns only what needs appkit: the menu bar, settings, first run,
  login item, and the footprint figure. no xcode project: a swiftpm
  executable and a shell script that assembles the bundle.
- memory-pressure handling stays in rust (`up::guard` already reads the
  sysctls). it parks in place of exiting, so it also works for `gall up`.

## contract (fixed now, so the work runs in parallel)

engine (rust), all in `src/serve.rs` and `src/up.rs`:

| item | behaviour |
| --- | --- |
| `GET /health` | `{"status":"ok","loaded":bool,"busy":bool,"last":T}`. `busy` is true while a chat runs. `T` is the `timings` object of the last chat (`cache_n`, `prompt_n`, `predicted_per_second`, ...), or `null`. no auth, as now. |
| `POST /park` | needs the key if one is set. queues a park behind the current chat: snapshot of the conversation, then unload. answers `{"parked":true}` when done. a no-op when unloaded. |
| `SIGTERM` | park, then exit 0. so `gall down`, a quit and a logout keep the conversation. |
| memory pressure | `kern.memorystatus_vm_pressure_level >= 2` (warn) with the model loaded: park, as `/park`. the current exit at `>= 4` stays. |
| `--host ADDR` | listen address, default `127.0.0.1`. `serve` refuses a host other than loopback without `--api-key-file`. |

the park path is the one the idle timeout uses now (the `else` arm of the
main loop): `s.lm.park(&s.fed[..n]); s.fed.clear()`. `/park`, the signal
and pressure all send one message down the existing queue, so a park never
runs during a chat. `Chat` becomes an enum or an `Option`; keep it small.

app ↔ cli, no new commands:

- the app runs its bundled `Contents/MacOS/gall`: `gall up CTX
  --idle-unload N [--host 0.0.0.0 --api-key-file K]`, `gall down`,
  `gall model`.
- progress of a download: the sum of the sizes of `~/.cache/gall/*.part`
  and of the finished files in that directory, against 12.7 GB. no
  download happens when `gall model` finds the files in the hugging face
  cache, so the bar is skipped when `gall model` returns at once.
- state: poll `/health` once a second while the menu is open, every 10 s
  otherwise. the server pid is in `~/.cache/gall/serve.pid`; the log is
  `~/.cache/gall/serve.log`.
- settings live in `UserDefaults`. a change runs `gall down` then `gall up`
  (the sigterm park keeps the conversation).
- the key, made on the first use of network mode: `~/.cache/gall/key`,
  mode 600, 32 random hex bytes.

## work packages

each package owns its files. no two packages edit one file. a, b and c
start at once. d is last.

### a. engine: park, health, signal, pressure, host (rust)

files: `src/serve.rs`, `src/up.rs`.

1. `busy` and `last` in `/health`: an `AtomicBool` beside `up`, and the
   last `timings` as a string in a `Mutex` (or a static), set in `chat()`.
2. `POST /park` and the queue message for it.
3. `SIGTERM`: block it, `sigwait` on a thread (libc is a dependency
   already), send park and exit down the queue.
4. `guard()`: at pressure `>= 2` send one park (not once a second: only
   when loaded). the thread needs the queue's `Sender`; pass it in.
5. `--host`, with the key check.
6. update the header comment of `serve.rs` and the server flags table in
   `README.md` (the flags table only; d owns the rest of the readme).

check: `gall serve $M --dummy` for the http parts (no gpu). then one gpu
run under the lock: a chat over 512 tokens, `curl -X POST /park`, `/health`
shows `loaded:false`, the next chat logs `snap N` with N close to the
prompt. `kill -TERM` gives a `snap ... written` line and exit 0. no change
to `gall test` (0 greedy mismatches).

### b. menu bar app (swift)

files: `mac/Package.swift`, `mac/Sources/Gall/*.swift`.

one `MenuBarExtra` (`.menu` style), `LSUIElement`. target: under 400
lines.

- status line: `not installed`, `downloading 41%`, `starting`, `unloaded`,
  `ready`, `answering`. the icon dims when unloaded.
- `memory 11.9 GB`: `proc_pid_rusage(pid, RUSAGE_INFO_V4)`,
  `ri_phys_footprint` of the server pid. `unloaded` shows about 0.2 GB.
- `last: 18,204 of 19,011 tokens cached, 20.4 t/s` from `last`.
- `copy endpoint` (`http://127.0.0.1:9940/v1`), `copy key` (network mode
  only), `unload now` (`POST /park`), `open log` (`open -a Console`).
- settings submenu: unload after 5, 15 (default), 60 min, never; context
  32k, 100k (default); start at login (`SMAppService.mainApp`); allow other
  devices on the network (makes the key, restarts with `--host 0.0.0.0`).
- first run: `hw.memsize >= 24 GB` and 14 GB free on the home volume, else
  a plain alert that says why. then `gall model` with the progress bar
  (the menu's status line is enough; no window), then `gall up`.
- quit: `gall down`, then exit.

develop against `gall serve $M --dummy` run by hand: point the app at an
already-running server (`up` is a no-op when the port answers).

### c. bundle, signing, release (shell)

files: `mac/build`, `mac/Info.plist`, `mac/gall.entitlements`, `.gitignore`.

`mac/build` (one script, no make):

1. `cargo build --release`; `swift build -c release --package-path mac`.
2. make `target/Gall.app`: `Contents/MacOS/Gall` (swift),
   `Contents/MacOS/gall` (rust), `Contents/Info.plist`,
   `Contents/Resources/AppIcon.icns` (if present).
3. `codesign --options runtime --timestamp` with the identity in
   `$GALL_SIGN`, inner binary first. without `$GALL_SIGN`, ad hoc (`-`).
4. with `$GALL_NOTARY` (a `notarytool` keychain profile name, never a
   password): `hdiutil create` a dmg, `xcrun notarytool submit --wait`,
   `xcrun stapler staple`.

to verify: that the hardened runtime allows the runtime metal compile in
`gpu.rs` (`newLibraryWithSource`) with no entitlement. if it does not,
name the one entitlement needed in `gall.entitlements`.

check: `mac/build && open target/Gall.app` on a clean user account
(`~/.cache/gall` absent) reaches `ready`; `spctl -a -vv target/Gall.app`
passes when signed.

### d. readme and agent notes (last)

files: `README.md`, `AGENTS.md` (new, short), `PLAN.md` (delete when done).

- an "app" section above "install and run": download, open, copy endpoint.
  keep the cli section as it is.
- the server flags rows from a.
- `AGENTS.md`: the gpu rule, the file ownership above, `mac/build`.

## gpu rule

one model fits on the gpu. any step that loads the model (a's gpu check and
c's last check) takes `/tmp/gall.lock` through `./wd` or runs
`gall` (which takes it). agents for a, b and c use `--dummy` until their
last check. two agents never load the model at once.

## out of scope for now

imessage and a separate `gall` user (the earlier design), sparkle updates,
a homebrew cask, a snapshot store capped by size instead of `KEEP = 8`
files, battery and low power mode rules, an anthropic-compatible endpoint,
client checks and comparisons with ollama and lm studio.
