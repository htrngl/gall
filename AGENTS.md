# Agent notes

## GPU rule

Only one model fits in GPU memory. A step that loads the model must hold
`/tmp/gall.lock`: run it through `./wd`, or run `gall`, which takes the
lock. Two agents must not load the model at the same time. Use
`gall serve $M --dummy` (no GPU) for HTTP work, and load the model only for
the last check.

## File ownership

Parallel agents divide the work by area. Two agents do not edit one file.

| Area | Files |
| --- | --- |
| Engine and server | `src/serve.rs`, `src/up.rs` (and the other `src/` files) |
| Menu bar app (Swift) | `mac/Package.swift`, `mac/Sources/Gall/*.swift` |
| Bundle and release | `mac/build`, `mac/Info.plist`, `mac/gall.entitlements`, `.gitignore` |
| Documentation | `README.md`, `AGENTS.md` |

The app runs its bundled `gall` (`up`, `down`, `model`) and reads
`/health`. It adds no commands to the CLI. Server state is in
`~/.cache/gall`: `serve.pid`, `serve.log`, and `key`.

## Build the app

`mac/build` builds the Rust and Swift binaries and makes
`target/Gall.app`. It signs with `$GALL_SIGN` (ad hoc without it). With
`$GALL_NOTARY`, a `notarytool` keychain profile name, it also makes and
notarizes a dmg. The hardened runtime needs no entitlement for the Metal
runtime compile, so `mac/gall.entitlements` is empty.

## Scope

Gall is for a person at their own Mac. The app runs as the user, with no
root, no daemons and no settings that keep the Mac awake. Server use
(always on, no login, no sleep, a chat bridge) belongs in a separate
wrapper script that runs `gall up` or `gall serve`.
