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
cargo install --git <this repository>
gall up          # first run fetches the model (11.7GB); server on port 9940
gall test        # 64 greedy tokens against a llama.cpp reference
gall down
```

- `gall up [ctx]` starts the server in the background with a 32,768-token
  context by default. It uses a GGUF in the Hugging Face cache if one is
  there, else it downloads to `~/.cache/gall`. `GALL_MODEL=path` overrides.
- `gall log` follows the request log. `gall model` prints the model path.
- The server speaks the OpenAI chat API at `http://127.0.0.1:9940/v1`:
  `/v1/chat/completions` (streaming, tools, reasoning), `/v1/models`,
  `/health`. Point any OpenAI-compatible client at it.
- Only one model fits on the GPU. `gall` takes a lock (`/tmp/gall.lock`) and
  refuses to start beside another instance; it exits if memory runs short.
  Unload other local models first.
- The prefix of each conversation (system prompt and tools) is stored in
  `~/.cache/gall`, so a new conversation with the same prefix starts in
  about a second.

## Measured

2026-11-21, this machine, one run each. the air has no fan: after about 2.5
minutes of load the figures drop by roughly 10%.

| | llama.cpp (`llama/`) | gall |
| --- | --- | --- |
| decode, short context | 15 t/s | 19.5 t/s cold, ~18 warm |
| decode, 8k deep | 12.4 t/s | 17.6 t/s |
| decode, 31.5k deep (real prompt) | not measured cleanly | 13.8-14.6 t/s |
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
- depth is bound by the KV read (81 GB/s): 56.7 ms at 8k, 67 at 32k, 102 at
  100k (the last two synthetic).
- `KV=q8` stores the full-attention caches as int8: half the cache memory at
  100k, no faster, off by default, its prefill path untested.
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
