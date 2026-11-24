# Agent notes

## GPU rule

Only one model fits in GPU memory. A step that loads the model must hold
`/tmp/gall.lock`. Every `gall` command that uses the GPU takes the lock and
waits for it; run other programs, such as `scratch/refdump`, through
`lockf /tmp/gall.lock`. Use `gall serve "$(gall model)" --dummy` (no GPU)
and `tools/smoke.py` for HTTP work, and load the model only for the last
check.

## File ownership

Parallel agents divide the work by area. Two agents do not edit one file.

| Area | Files |
| --- | --- |
| Engine and server | `src/serve.rs`, `src/up.rs` (and the other `src/` files) |
| Menu bar app (Swift) | `mac/Package.swift`, `mac/Sources/Gall/*.swift` |
| Bundle and release | `mac/build`, `mac/bridge`, `mac/Info.plist`, `mac/gall.entitlements`, `.gitignore` |
| Documentation | `README.md`, `AGENTS.md`, `docs/`, `tools/README.md` |

The app runs its bundled `gall` (`up`, `down`, `model`) and reads
`/health`. It adds no commands to the CLI. Server state is in
`~/.cache/gall`: `serve.pid`, `serve.log`, and `key`.

## Build the app

`mac/build` builds the Rust and Swift binaries and makes
`target/Gall.app`, `target/Gall.dmg` and its SHA-256 checksum. It signs
with `$GALL_SIGN` (ad hoc without it). With `$GALL_NOTARY`, a `notarytool`
keychain profile name, and a Developer ID identity in `$GALL_SIGN`, it also
notarizes and staples the dmg. See [releasing](docs/RELEASE.md). The
minimum macOS version is 15 (`MTLCompileOptions.mathMode`). The hardened
runtime needs no entitlement for the Metal runtime compile, so
`mac/gall.entitlements` is empty. The Swift binary is
`Contents/MacOS/GallMenu`, beside `Contents/MacOS/gall`: on a
case-insensitive disk `Gall` and `gall` are one file. The icons are drawn
from one outline: `mac/Sources/Gall/Leaf.swift` (menu bar) and
`mac/leaf.svg` (`mac/AppIcon.icns`, made with `qlmanage`, `sips` and
`iconutil`).

## Scope

Gall is for a person at their own Mac. The app runs as the user, with no
root and no daemons. Server use (always on, no login, no sleep) belongs in
a separate wrapper script that runs `gall up` or `gall serve`.

The Telegram bridge is `mac/bridge`, a shell script in
`Contents/Resources`. The app runs it while the `bridge` setting is on
and kills its process group when the setting goes off or the app quits.
It reads `~/.config/gall/bridge` and pipes each message through
`AGENT` (`sh -c`), by default hax as a llama.cpp client of the server.
While the bridge runs, the app holds a `PreventUserIdleSystemSleep`
assertion (the system drops it if the app dies) and sets the root
domain's clamshell-sleep-disable bit (`kPMSetClamshellSleepState`, no
root needed). The kernel keeps that bit after the app exits, so the app
clears it on each start and stop.

The `wake` setting lets the Mac sleep with the lid closed. The app runs
`sudo -n pmset schedule wake` for 5 minutes later before a sleep from a
full wake (`IORegisterForSystemPower`) and at each dark wake, which only
the `com.apple.powermanagement.systempowerstate` notification reports
(the capability bits are in `State:/IOKit/SystemPowerCapabilities`). The
sudoers rule in `/etc/sudoers.d/gall`, installed with an administrator
prompt, allows only that command. On a dark wake, the app sets the
clamshell bit, declares user activity (a full wake: on battery no
assertion holds a dark wake) and kills the stale long poll. It clears
the bit, and the Mac sleeps, 30 s after a wake or 2 minutes after a
message. The bridge keeps `~/.cache/gall/bridge.run` while a message
runs.

## Model lock

The server holds `/tmp/gall.lock` while a model is loaded and releases it
when it unloads the model. Other commands hold it until they exit. A
second server or command waits for the lock, so `--idle-unload` lets the
server and measurement runs take turns. The server starts unloaded and
loads on the first chat.

## Development

```sh
cargo build --release
cargo test
python3 tools/smoke.py target/release/gall "$(gall model)"
target/release/gall check "$(gall model)" ref/short -n 64
```

`tools/memory.py` measures the server footprint in each phase.
The `ref/` directory holds reference dumps from `tools/refdump`, which uses
the llama.cpp build in `llama/`. See [Reference tools](tools/README.md).
The header of `src/main.rs` lists the other commands, including `run`,
`pp`, `tok`, `vis`, `dr`, and `mvtest`.

## Repository layout

- `src/gguf.rs`, `gpu.rs`, `alloc.rs`: GGUF memory mapping, the Metal
  host, and the allocator that unmaps large buffers when they drop.
- `src/mv.*`, `grids.metal`, `mvtest.rs`: Matrix-vector kernels for the 12
  weight formats, and their test.
- `src/model.rs`, `ops.metal`: Single-token decode, key-value (KV) cache
  rings, and rollback.
- `src/pf*`: Batched prefill.
- `src/tok.rs`, `serve.rs`, `snap.rs`, `up.rs`: Tokenizer, chat template,
  server, prefix snapshots, and server lifecycle commands.
- `src/vis.*`: Image decoding (ImageIO) and the vision encoder.
- `src/dr*`: The dspark drafter and its batched verification.
- [Engine brief](docs/ENGINE.md) and [drafter notes](docs/DRAFT.md).
- `mac/`: The menu bar app (Swift) and `mac/build`, which makes
  `target/Gall.app`.
- `tools/`: Reference dumps from llama.cpp, the HTTP smoke test and the
  memory measurement.
- `llama/`: The patched llama.cpp reference build (`llama/build`).

## Measurements

The [engine notes](docs/ENGINE.md) and [drafter notes](docs/DRAFT.md) hold
the measurements and implementation notes. Keep them current there, and
keep the README tables in agreement with them. [Limits](docs/LIMITS.md)
holds the performance limit model, the outcome of each proposal
and what is still open.
