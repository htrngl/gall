# Releasing Gall

Gall is an arm64 Mac app. It requires macOS 15 or later and at least 24 GB
of memory. Measurements currently cover one M2 MacBook Air running macOS
27.2; the minimum version is based on the APIs used, not a completed test
on macOS 15.

## Local build

```sh
mac/build
```

This builds the Rust server and Swift menu app, signs the app ad hoc,
checks its signature, and produces:

- `target/Gall.app`
- `target/Gall.dmg`, containing the app and an Applications shortcut
- `target/Gall.dmg.sha256`

The app includes `LICENSE` and `NOTICE` in its Resources directory.
An ad hoc build is for local testing; it is not a notarized public release.

## Checks before publishing

Run the allocator and HTTP tests without loading GPU weights:

```sh
cargo test
python3 tools/smoke.py target/release/gall "$(gall model)"
```

Run the engine reference check. It takes the GPU lock:

```sh
target/release/gall check "$(gall model)" ref/short -n 64
```

For a build without local reference dumps, `target/release/gall test`
compares against the bundled greedy token IDs instead.

Check repeated unload and reload with the real model, including a
conversation longer than 512 tokens and an image request. Record the
unloaded footprint after each cycle, snapshot reuse and reload latency.
Compare deterministic responses with the previous build. For prompt and generation memory:

```sh
python3 tools/memory.py target/release/gall "$(gall model)"
```

On a separate Mac or clean user account, check the first download, a
cancelled download followed by a retry, app launch, a chat client
connection, Instant unload, quit, and start at login. Check the oldest
macOS version claimed by the release before describing it as tested.

## Signed distribution

Set the version in `Cargo.toml`; the build uses it for both bundle version
fields. With a Developer ID Application certificate and a configured
`notarytool` keychain profile:

```sh
GALL_SIGN='Developer ID Application: Your Name (TEAMID)' \
GALL_NOTARY='your-notary-profile' mac/build
```

The build rejects notarization without a signing identity, signs the app
and DMG, waits for notarization, staples and validates the ticket, then
writes the checksum of the final DMG.

Upload the DMG and checksum to the release after those checks pass. Test
the downloaded DMG through Finder on another Mac, so Gatekeeper sees the
same quarantined download that users will receive. Building locally does
not publish a release or replace an installed copy of Gall.
