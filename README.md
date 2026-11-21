# gall

a from-scratch inference engine for thinkingmachines' inkling-2-mini (26B MoE,
3.4B active, 48 layer visits per token) on one machine: an m2 macbook air,
24GB. rust host, metal kernels, the unsloth UD-IQ3_XXS GGUF read in place.
no ggml, no mlx; four small crates (metal bindings and libc).

iron gall ink is made from oak apples and iron.

## measured

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

## use

```sh
cargo build --release
. ./env                                  # M=path to the GGUF
./wd target/release/gall serve $M --port 9940 --ctx 32768
```

openai-compatible `/v1/chat/completions` (streaming, tools, reasoning),
`/v1/models`, `/health`. the hax preset `gall` points at port 9940. the
installed copy is `~/.llama-app/gall/gall`.

`./wd` takes a machine-wide lock and kills the command if memory runs out.
two copies of the model do not fit on the GPU: stop the llama app's model
and any other `gall` first.

other commands: `gall run`, `check`, `pp` (prefill timing), `tok`, `dr`
(drafter), `mvtest` (weight kernels alone). usage is at the top of
`src/main.rs`.

## layout

- `src/gguf.rs`, `gpu.rs`: GGUF mmap, metal host.
- `src/mv.*`, `grids.metal`: mat-vec kernels for the 12 weight formats.
- `src/model.rs`, `ops.metal`: single-token decode, KV rings, rollback.
- `src/pf*`: batched prefill.
- `src/tok.rs`, `serve.rs`, `snap.rs`: tokenizer, chat template and server,
  prefix snapshots in `~/.cache/gall`.
- `src/dr*`: the dspark drafter (correct, not yet faster; see below).
- `docs/ENGINE.md`: the original brief. `docs/DRAFT.md`: drafter notes.
- `tools/`: reference dumps from llama.cpp.
- `llama/`: the patched llama.cpp build used as reference and its notes
  (`llama/PLAN.md`); `llama/build` installs it for the llama app.

## state

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
