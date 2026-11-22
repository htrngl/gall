# gall

A from-scratch inference engine for Thinking Machines' Inkling-2-Mini (26B
MoE, 3.4B active, 48 layer visits per token) on Apple silicon. A Rust host
and Metal kernels read the unsloth UD-IQ3_XXS GGUF in place. It does not use
ggml or MLX; its four dependencies are Metal bindings and libc.

Iron gall ink is made from oak apples and iron.

## Requirements

- An Apple silicon Mac with 24GB of memory. gall was written and measured on
  one machine, an M2 MacBook Air (10-core GPU). Other machines are untested.
- Rust (`cargo`) and `curl`.
- 12GB of disk for the model.

## Install and run

```sh
cargo install --git https://github.com/ltrgoddard/gall
gall up          # first run fetches the model (11.7GB); server on port 9940
gall test        # 64 greedy tokens against a llama.cpp reference
gall down
```

- `gall up [ctx]` starts the server in the background with a 102,400-token
  context by default. It uses a GGUF in the Hugging Face cache if one is
  there, else it downloads to `~/.cache/gall`. `GALL_MODEL=path` overrides.
- `gall log` follows the request log. `gall model` prints the model path.
- The server speaks the OpenAI chat API at `http://127.0.0.1:9940/v1`:
  `/v1/chat/completions` (streaming, tools, reasoning), `/v1/models`,
  `/health`. Point any OpenAI-compatible client at it.
- `gall serve <gguf> --api-key-file F` asks every request but `/health` for
  the key in `F` (`authorization: Bearer KEY`).
- `gall serve <gguf> --idle-unload N` frees the model's memory after `N`
  seconds without a request. It first stores the conversation as a prefix
  snapshot. The next chat request loads the model again (0.06 s with the
  file in the page cache) and continues from the snapshot. `/health` shows
  `"loaded"`. The lock is given back while the model is unloaded.
- Only one model fits on the GPU. `gall` takes a lock (`/tmp/gall.lock`)
  while a model is loaded. A second server, or `./wd`, waits for it; with
  `--idle-unload`, the server and a measurement take turns. It exits if
  memory runs short. Unload other local models first.
- The prefix of each conversation (system prompt and tools) is stored in
  `~/.cache/gall`, so a new conversation with the same prefix starts in
  about a second.

## Measured

2026-11-21, this machine, one run each. the air has no fan: after about 2.5
minutes of load the figures drop by roughly 10%.

| | llama.cpp (`llama/`) | gall |
| --- | --- | --- |
| decode, short context | 15 t/s | 19.5 t/s cold, ~18 warm |
| decode, 10.5k deep (real prompt) | 12.4 t/s (8k) | 16.4-16.8 t/s |
| decode, 29.9k deep (real prompt) | not measured cleanly | 15.3-15.4 t/s |
| prefill, 512 tokens | 121 t/s | 108-125 t/s |
| prefill, 10.5k tokens | 89 t/s (to 8.3k) | 106 t/s |
| new chat with a stored prefix, first token | | 0.5-1.2 s |

against llama.cpp reference dumps, teacher-forced: 0 greedy mismatches over
64 steps, kl mean 1.05e-4 (`gall check $M ref/short -n 64`). a code planted
400 tokens into a 31.5k-token prompt is retrieved at 100k context.

## Development

```sh
cargo build --release
./wd target/release/gall check $(gall model) ref/short -n 64
```

`./wd` runs a command under the same lock with a memory watchdog. `ref/` holds
reference dumps made with `tools/refdump` against the llama.cpp build in
`llama/` (see `tools/README.md`). Other commands (`run`, `pp`, `tok`, `dr`,
`mvtest`) are listed at the top of `src/main.rs`.

## Layout

- `src/gguf.rs`, `gpu.rs`: GGUF mmap, metal host.
- `src/mv.*`, `grids.metal`: mat-vec kernels for the 12 weight formats.
- `src/model.rs`, `ops.metal`: single-token decode, KV rings, rollback.
- `src/pf*`: batched prefill.
- `src/tok.rs`, `serve.rs`, `snap.rs`, `up.rs`: tokenizer, chat template and
  server, prefix snapshots, start and stop.
- `src/dr*`: the dspark drafter (correct, not yet faster; see below).
- `docs/ENGINE.md`: the original brief. `docs/DRAFT.md`: drafter notes.
- `tools/`: reference dumps from llama.cpp.
- `llama/`: the patched llama.cpp build used as reference and its notes
  (`llama/PLAN.md`); `llama/build` installs it for the llama app.

## State

- weights are 45 of 51 ms per token. every format costs 7-12 ps per element;
  two tuning passes on the expert gate-up kernels found nothing more.
- depth is bound by the cache read (85 GB/s). from 4096 tokens deep the 8
  full-attention visits read an int8 copy of the cache (2304 bytes per key,
  f16 is 4096), and the f16 rows only of the keys within 1% of the greatest
  weight. gpu ms per token, real prompts, f16 only against the default, back
  to back: 56.8 against 56.2 at 10.5k, 64.6 against 61.3 at 29.9k. 100k is
  not measured.
- against f16 only over 16 steps: no greedy token differs; kl mean 4e-9 at
  10.5k and 5e-5 at 29.9k (one step has 8.7e-4, the other 15 average 2e-7).
  `gall check ref/8k.b256 -ub -n 32`: 0 greedy mismatches, kl mean 1.30e-3
  (f16 only: 1.32e-3).
- the copy takes 18KB of memory per token of depth, as well as the 33KB of
  the f16 rows, and none until it is written (footprint at a 102,400-token
  context and a short prompt: 4.63GB with and without it). `VT=0` turns it
  off; `VT` is the weight threshold (0.01).
- the drafter runs and its batched verify gives the same text as greedy, but
  a 4-row verify takes 250 ms against 52 ms per serial token, so the block
  size stays 0. it needs dequant-once expert kernels.
- prefill uses half-precision tiles, as llama.cpp does; `HT` in `pfm.metal`
  switches to float at about 0.8x the speed.
- not done: grammar and `response_format`, images, concurrent requests,
  drafting in the server, a 4-bit attention requant (projected 3.7 ms per
  token, needs the 53GB original and a quality check).

## Licence

MIT. Parts derive from llama.cpp and ggml; see `NOTICE`. The model has its
own licence.
