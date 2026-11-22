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
- `src/rq.rs`: attention weights requantized at load (`RQ`, see below).
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
- `RQ=5` requantizes the 100 q8_0 attention projections (234 dispatches
  per token) from their q8_0 values to 5 bits in a buffer of 293 MB: 4.8 s
  the first time, then read from `~/.cache/gall/*.rq` with no measurable
  load time. the attention read goes from 12.9 to 8.8 ms and a token from
  50.8 to 47.2 ms; `RQ=6` gives 48.7 ms and `RQ=4` 46.0 ms. `RQ=5,5:h` also
  moves the q5_k output head to 5 bits (303 MB more; 5.36 to 3.35 ms):
  45.5-46.2 ms per token. prefill and the drafter read the file's tensors.
- `RQ` is off by default because it changes the weights (relative error
  4.5% at 5 bits, 2.3% at 6, 7.7% at 4). on `ref/short` the kl mean goes
  from 1.05e-4 to 3.2e-3 at 5 bits, 2.8e-3 at 6 and 1.3e-2 at 4, with 0,
  0 and 1 greedy mismatches of 64 (3.8e-3 and 1 with the head); on
  `ref/2k` from 1.1e-3 to 2.4e-3, 1.4e-3 and 2.8e-3, with 2, 1 and 2
  mismatches. one tensor of 0.9M weights at 6 bits (`RQ=6:r:0-0`) already
  gives 1.4e-3 and 1 mismatch on `ref/short`. on 500 tokens of each of two
  texts, teacher-forced, the nll does not move (-0.001 +- 0.004 at 4 bits)
  and the kl to the engine without `RQ` is 1.6-1.8e-3, 2.8-2.9e-3 and
  4.5-4.7e-3 at 6, 5 and 4 bits (2.9e-4 for the one tensor). no choice of
  projection kinds or layers did better than a lower bit width.
- not done: grammar and `response_format`, images, concurrent requests,
  drafting in the server.

## Licence

MIT. Parts derive from llama.cpp and ggml; see `NOTICE`. The model has its
own licence.
