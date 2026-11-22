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
runtime compile, so `mac/gall.entitlements` is empty. The Swift binary is
`Contents/MacOS/GallMenu`, beside `Contents/MacOS/gall`: on a
case-insensitive disk `Gall` and `gall` are one file. The icons are drawn
from one outline: `mac/Sources/Gall/Leaf.swift` (menu bar) and
`mac/leaf.svg` (`mac/AppIcon.icns`, made with `qlmanage`, `sips` and
`iconutil`).

## Scope

Gall is for a person at their own Mac. The app runs as the user, with no
root, no daemons and no settings that keep the Mac awake. Server use
(always on, no login, no sleep, a chat bridge) belongs in a separate
wrapper script that runs `gall up` or `gall serve`.

## Model lock

Gall holds `/tmp/gall.lock` while a model is loaded and releases it when
it unloads the model. A second server or `./wd` waits for the lock, so
`--idle-unload` lets the server and measurement runs take turns.

## Development

```sh
cargo build --release
./wd target/release/gall check $(gall model) ref/short -n 64
```

`./wd` runs a command under the same lock and monitors memory use.
The `ref/` directory holds reference dumps from `tools/refdump`, which uses
the llama.cpp build in `llama/`. See [Reference tools](tools/README.md).
The header of `src/main.rs` lists the other commands, including `run`,
`pp`, `tok`, `vis`, `dr`, and `mvtest`.

## Repository layout

- `src/gguf.rs`, `gpu.rs`: GGUF memory mapping and the Metal host.
- `src/mv.*`, `grids.metal`, `mvtest.rs`: Matrix-vector kernels for the 12
  weight formats, and their test.
- `src/rq.rs`: Attention weights requantized at load (`RQ`, disabled by default).
- `src/model.rs`, `ops.metal`: Single-token decode, key-value (KV) cache rings, and rollback.
- `src/pf*`: Batched prefill.
- `src/tok.rs`, `serve.rs`, `snap.rs`, `up.rs`: Tokenizer, chat template,
  server, prefix snapshots, and server lifecycle commands.
- `src/vis.*`: Image decoding (ImageIO) and the vision encoder.
- `src/dr*`: The dspark drafter and its batched verification.
- [Engine brief](docs/ENGINE.md) and [drafter notes](docs/DRAFT.md).
- `mac/`: The menu bar app (Swift) and `mac/build`, which makes
  `target/Gall.app`.
- `tools/`: Tools for reference dumps from llama.cpp.
- `llama/`: The patched llama.cpp reference build and its
  [notes](llama/PLAN.md). `llama/build` installs it for the llama app.

## Measurements

These results are from one run per measurement on the M2 MacBook Air on
21 and 22 November 2026. The MacBook Air has no fan. Throughput drops by
roughly 10% after about 2.5 minutes of load. Rates are in tokens per
second (t/s).

| Measurement | llama.cpp (`llama/`) | Gall |
| --- | --- | --- |
| Decode, short context | 15 t/s | 20.5 t/s cold |
| Decode, 10.5k context (real prompt) | 12.4 t/s (at 8k) | 18.0 t/s |
| Decode, 29.9k context (real prompt) | No reliable measurement | 16.2–16.5 t/s |
| Prefill, 512 tokens | 121 t/s | 130 t/s |
| Prefill, 10.5k tokens | 89 t/s (measured to 8.3k) | 106 t/s |
| Time to first token with a stored prefix | Not measured | 0.5–1.2 s |

A teacher-forced comparison against llama.cpp reference dumps found no
greedy mismatches across 64 steps. The mean Kullback–Leibler (KL) divergence
was 1.05e-4 (`gall check $M ref/short -n 64`).

A retrieval test recovered a code placed 400 tokens into a 31.5k-token
prompt with a 100k context.

## Implementation notes

- Weight processing accounts for 44.2 of 48.2 ms per token, and small
  kernels for 4.9 ms, in 1,071 dispatches. A token is one command buffer,
  encoded while the previous token runs. Wall time is 0.4 ms above GPU time.
- The expert kernels take 23 ms per token and are bound by arithmetic, not
  by memory reads. An iq3_xxs gate-up visit takes 0.30 ms, of which the
  loads take 0.20 ms. Three tuning passes produced no further improvement.
- The k-quant kernels unpack 4 bytes as one word. The q5_k head reads at
  73 GB/s (4.4 ms), and q6_k at 87 GB/s.
- KV cache reads limit performance at longer contexts, at 85 GB/s. From
  4,096 tokens of context, the 8 full-attention visits read an int8 copy
  of the cache (2,304 bytes per key; f16 uses 4,096). They read the f16
  rows only for keys within 1% of the largest weight.
  On real prompts, GPU time per token is 53.6 ms at 10.5k (54.3 ms with
  f16 only) and 58.6 ms at 29.9k (61.9 ms with f16 only). The two modes
  were measured back to back in one run. No 100k measurement exists.
- Against f16 only, no greedy token differs across 16 steps. The mean KL
  divergence is 4e-9 at 10.5k and 4e-8 at 29.9k.
  `gall check ref/8k.b256 -ub -n 32` finds no greedy mismatches, with a
  mean KL divergence of 1.30e-3 (1.32e-3 with f16 only).
- The int8 copy uses 18 KB of memory per token of context, in addition to
  33 KB for the f16 rows. It uses no memory until it is written. After a
  snapshot restore or a prefill, the first decoded token rebuilds the copy
  from the f16 rows. At 29.9k, the first token after a restore takes
  235 ms (259 ms with f16 only), and the logits equal those without the
  restore. `VT=0` disables the copy. `VT` sets the
  weight threshold (default 0.01).
- With `dspark-Inkling-2-Mini-Q8_0.gguf` beside the model, the server
  verifies drafted tokens ahead in batches of 2 to 8 rows. A pass of two
  or four rows takes 70 or 117 ms, compared with 49 ms per serial token.
  The logits of every step remain those of the model, so sampling is
  unchanged: with one seed, the messages are equal with and without the
  drafter. The server drafts only where the drafter is confident.
  On 120-token answers at temperature 0, it measured 20.5 to 20.7 t/s
  for prose, 20.4 to 23.4 t/s for a proof, and 20.4 to 21.4 t/s for code.
  With the default sampling, the same requests measured 20.2 to 20.5,
  20.0 to 21.8, and 20.2 to 20.4 t/s. `--no-draft` disables the drafter.
  It is off under `response_format`. See the
  [drafter notes](docs/DRAFT.md).
- Prefill uses half-precision tiles, as llama.cpp does. The `HT` setting in
  `pfm.metal` selects float precision at approximately 0.8 times the speed.
- The vision encoder runs on the GPU. A 640 x 488 image (208 tokens) takes
  0.8 s and a 960 x 1280 image (768 tokens) takes 4.2 s. Its output is
  within 2.9e-3 (relative) of llama.cpp's (`gall vis`,
  `tools/imgref.cpp`).
- `RQ=5` requantizes the 100 q8_0 attention projections (234 dispatches
  per token) from their q8_0 values to 5 bits, in a 293 MB buffer. The
  first run takes about 5 s. Later runs read `~/.cache/gall/*.rq` with no
  measurable load time. The attention read takes 8.9 ms instead of 12.7 ms,
  and a token takes 45.1 ms instead of 48.7 ms. `RQ=6` gives 46.5 ms.
  `RQ=4` and `RQ=5,5:h` (the output head at 5 bits) are also accepted.
  Prefill and the drafter read the tensors of the file.
- `RQ` is disabled by default because it changes the weights. The relative
  error is 4.5% at 5 bits and 2.3% at 6 bits. On `ref/short`, the KL mean
  changes from 1.05e-4 to 3.15e-3 at 5 bits and 2.77e-3 at 6 bits, with 0
  greedy mismatches in 64 steps. On `ref/2k`, it changes from 1.1e-3 to
  2.4e-3 and 1.4e-3, with 2 and 1 mismatches. On 500 tokens of each of two
  texts, the negative log-likelihood does not change (-0.001 +- 0.004 at
  4 bits). A selection of projection kinds or layers gave no better
  result than a different bit width.
- At 6 bits, the rq6 format is no faster than the q6_k kernel (10.0 ms
  and 9.9 ms for the attention read). At 5 bits, rq5 takes 8.9 ms and
  q5_k takes 11.0 ms.
