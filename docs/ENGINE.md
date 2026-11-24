# Engine

Gall is a Rust and Metal inference engine for Inkling-2-Mini on Apple
silicon. It reads the UD-IQ3_XXS GGUF weights directly, without ggml or
MLX. The implementation is specific to this architecture and is measured
on a 24 GB M2 MacBook Air with a 10-core GPU.

The main constraint is sharing that Mac with other applications. Gall
releases its model buffers between conversations and saves enough state
to avoid processing a long, unchanged prompt again.

## Load, answer, unload

The HTTP server owns a `Lazy` model in `src/serve.rs`. It starts unloaded,
loads for the first chat, then unloads on an idle timeout, `POST /park`,
sustained memory pressure, or a graceful shutdown. A new chat loads it
again. The app’s default idle timeout is one second; the CLI requires
`--idle-unload 1` for that behavior. Queued chats run before the idle
timeout can expire.

Before unloading, Gall writes conversation state if it has at least 512
tokens and the model state matches the recorded token sequence. It then
drops the model, vision encoder and drafter, and releases `/tmp/gall.lock`.
The HTTP server and tokenizer remain available. Model, drafter and vision
file mappings remain in the process; their virtual size is not a measure of
physical memory in use. macOS can reclaim file-backed pages.

Large Rust allocations (at least 1 MiB, with alignment up to 4 KiB) use
anonymous mappings through `src/alloc.rs`. Dropping them unmaps their pages
immediately. Smaller or more highly aligned allocations use the system
allocator. This avoids retaining large temporary and checkpoint buffers
in macOS’s deferred-reclaim malloc cache. Checkpoints reserve their full
size up front instead of growing through intermediate allocations.

Load, chat and unload have autorelease pools. After dropping the model,
Gall also asks malloc to release unused pages. That call alone did not
prevent growth across reloads on the test machine.

A sequential comparison on 22 November 2026 used three identical
805-token requests, 24 generated tokens each, a 102,400-token context
limit, the drafter, and a one-second idle timer. Each server had an empty
snapshot directory and a copy of the cached drafter projection.

| Unloaded server footprint | Previous build | Updated build |
| --- | ---: | ---: |
| After first text request | 354.9 MiB | 48.6 MiB |
| After second text request | 542.6 MiB | 49.1 MiB |
| After third text request | 730.1 MiB | 49.1 MiB |
| After an image request | 817.0 MiB | 54.6 MiB |

These are physical footprints reported by `vmmap`, including compressed
memory, not just resident pages. Text and image response messages matched
in every pair. The updated server retained no GGUF file descriptors and
its mapped-file virtual size stayed constant across reloads. Restored
requests reused 791 prompt tokens; model reload took 0.10–0.11 seconds and
time to first token was about 1.18 seconds. These are warm-cache results.

A separate 805-token streaming request with 64 generated tokens measured
32.4 MB before first use, a peak 5.29 GB process footprint during both
prompt processing and generation, and 50.8 MB after unloading. Sampling
used `proc_pid_rusage` every 20 ms. The process footprint does not include
all file-backed model weights; adding the 12.7 GB model files gives the
README’s conservative active budget of about 18 GB. It is not a measured
resident-memory total. The menu app is excluded from these figures.

Reproduce the phase measurement with:

```sh
python3 tools/memory.py target/release/gall "$(gall model)"
```

The tool creates a temporary snapshot cache and removes it afterward.
The first-output boundary includes model loading and prompt processing;
generation begins with the first streamed text or reasoning fragment.

## Weights and working memory

`src/gguf.rs` maps the GGUF file. `src/gpu.rs` wraps its pages in Metal
buffers with `newBufferWithBytesNoCopy`, avoiding a second copy of the
weights. The model, drafter and vision encoder retain one mapping each
across unloads. Each file descriptor is closed immediately after mapping.

The file uses several weight formats: NVFP4, IQ3_XXS, IQ2_S, IQ2_XS,
IQ2_XXS and IQ4_NL for experts; Q8_0, Q6_K, Q5_K and Q4_K for projections,
the output head and embeddings; and F32 for controllers. Gall implements
the corresponding matrix-vector kernels in `src/mv.*`.

Sliding-window attention uses fixed-size cache rings. Full attention
keeps f16 cache rows, using about 33 KB per token across its visits. From
4,096 tokens onward, decode also uses an int8 copy, about 18 KB per token.
The copy consumes physical memory as it is written. The drafter adds
about 4 KB per token for its own context, plus its weights and buffers.
These figures describe cache components, not total process memory.

## Saved prefixes

`src/snap.rs` stores token IDs, convolution state, sliding-window rows and
full-attention rows in `~/.cache/gall/*.snap`. Snapshot tags identify the
model file and cache layout. The cache retains up to eight snapshots,
evicting the least recently used files.

For a new request, Gall looks for a compatible saved prefix. It can also
restore a snapshot whose final tokens differ, within the model’s rollback
window. It processes the remaining prompt normally. System prefixes of
at least 512 tokens can be saved separately.

Snapshots are an inference cache. They do not replace the history sent by
a client, and they are not a permanent conversation archive. A missing or
incompatible snapshot causes prompt processing again. Snapshot write
errors are logged; unloading still releases the model.

The int8 cache is rebuilt from f16 rows after restoration or prefill. At
29.9k context, the first decoded token after a restore took 235 ms,
compared with 259 ms using f16 alone. The restored logits matched those
without a restore. The drafter’s context is not saved; it begins again
with newly processed tokens.

## Decode and prefill

Inkling-2-Mini has 24 physical layers. Layers 6–17 run three times, giving
48 layer visits per token. It uses 24 query heads, 8 key/value heads,
sliding-window and full attention, short convolutions, cross-pass
messages and 128 experts with 8 selected per visit.

Gall encodes one command buffer per decoded token while the previous
token runs. Decode uses a concurrent encoder with a barrier before each
dispatch, except among the q, k, v, rel and gate projections, which read
the same input. On 23–24 November 2026, warm and in interleaved runs, a
short-context token took 39.5–39.8 ms of GPU time (about 25 tokens/s),
down from 45.2 ms.

With greedy decoding, Gall commits the next token before it waits for the
current one, using the GPU’s greedy token ID (`Model.greedy`). The head of
the next token waits on a shared event, so the logits and ID stay readable
until then. This saves 0.4–0.5 ms of wall time a token, so wall time equals
GPU time. `gall run` uses it, and so does the server for temperature 0
requests without `response_format` when no drafter is loaded.

The weight bytes read per token are about 2.9 GB: 1.30 GB of experts,
1.11 GB of Q8_0 attention, 0.32 GB of output head and 0.18 GB of other
attention. At the Q8_0 rate of about 87 GB/s that is 33 ms, or 30
tokens/s. No `bw` access shape and no anonymous copy of a tensor read
faster than the Q8_0 mat-vec. A 7.5 MB tensor read again ran at 117–119
GB/s, against 88 GB/s cold, from the system cache.

| Part of a token | GPU ms | Floor at 87 GB/s |
| --- | ---: | ---: |
| IQ3_XXS experts | 5.95 | 5.4 |
| NVFP4 experts | 5.86 | 6.0 |
| IQ2_S experts | 3.37 | 2.7 |
| IQ4_NL, IQ2_XS and IQ2_XXS experts | 0.92 | 0.82 |
| All experts | about 16.1 | 14.9 |
| Q8_0 attention | 12.9 | 12.8 |
| Q5_K head and attention | 4.26 | 4.2 |
| Q6_K and Q4_K attention | 1.68 | 1.6 |
| Small kernels | about 4.7 | about 1.6 |

IQ3_XXS is limited by arithmetic, not by memory. The other weight kernels
are within 0.7 ms of their floors.

The grid and k-quant kernels read weight bytes as subnormal halves: byte
`b` is `b * 2^-24`, which converts at no cost, and the power of two goes
into the block scale. Grid signs are set by one multiply and shifts. NVFP4
nibbles become half bits 9–11 and 15 (an e2m1 value times 2^-14). Results
are bit-identical to the previous kernels. The IQ3_XXS and IQ2 grids are
in threadgroup memory; IQ4_NL looks up a byte in a threadgroup table of
the halves of both nibbles. IQ2_S and IQ2_XS lanes take two units each,
loaded together, with 8 rows split between the half-simdgroups.

| Format | Before, G weights/s | Now, G weights/s |
| --- | ---: | ---: |
| IQ3_XXS | 142 | 204 |
| IQ2_S | 147 | 194 |
| IQ2_XXS | 166 | 241 |
| IQ2_XS | 141 | 183 |
| NVFP4 | 141 | 157 |
| IQ4_NL | 143 | 158 |
| Q4_K | 110 | 133 |

From a cached 64 KB slab, Q8_0 runs at 195–199 G weights/s. The IQ2_S
change to two units a lane raised it from 62.7 to 68.7 GB/s. The Q5_K
head went from 4.23 to 3.64 ms (75 to 87.5 GB/s). Q6_K reads at 87 GB/s.

The loop controller `ctl` takes 16 lanes per element, each lane 4 of its
64 taps, and sums by shuffles; this saved 0.26–0.36 ms per token. With no
barriers at all (incorrect results), a token is 3.4–4 ms faster. That cost
is the latency of each small kernel, not the barrier: a barrier and a
launch cost 1–2 µs. By kind, small kernels take these ms per token: dots
0.38, elem 0.56, low 0.84, ctl 0.79, rows 0.54, top8 0.04, post 0.51,
score 0.41, gather 0.50, merge 0.05, act 0.21. Fusing dots with elem,
rows with top8, or score with gather was slower each time. Computing act
inside `up` cannot be bit-identical under fast math. The f16 attention
path finds the head maxima in `gather`, so it has no `smax` dispatch.

The model is sensitive to single-ulp changes through near-tied experts,
so kernel changes must keep the output bits. `gall mvtest` checks each
weight kernel against a scalar reference and prints its rate and a hash
of its output bits. `DEFS='#define SLAB 65536'` runs every kernel on one
64 KB slab, so memory is not the limit; `ANON=1` times copies in anonymous
buffers in place of the file mapping; `gall mvtest bw` times bus access
shapes. `gall dr --drift` prints a hash of each verify history’s logits.

Prefill uses half-precision matrix tiles, as the reference llama.cpp build
does. The `HT` setting in `pfm.metal` selects float precision at about 80%
of that speed.

### Prefill floor

`PROF=1 gall pp MODEL TOKENS` runs each prefill dispatch in its own
command buffer. It prints GPU ms, weight bytes, tile multiply-adds (MACs)
and real MACs by kind. The table is from one such run on 24 November
2026: the first 2,048 tokens of `ref/2k`, 4 passes of 512 rows, on a
warm machine shared with other jobs. The dispatches sum to 9.25 ms a
token; plain `pp` runs at that time gave 8–10 ms a token (100–133
tokens/s). The byte floor is the distinct weight bytes of each pass at
87 GB/s. The arithmetic floor is the real MACs at 1.79 T MAC/s (10 cores
x 128 lanes at 1.4 GHz).

| Part of a token | GPU ms | Bytes floor | Arithmetic floor |
| --- | ---: | ---: | ---: |
| IQ3_XXS experts | 1.80 | 0.14 | 0.69 |
| NVFP4 experts | 1.53 | 0.16 | 0.52 |
| IQ2_S experts | 1.05 | 0.08 | 0.40 |
| IQ4_NL, IQ2_XS and IQ2_XXS experts | 0.26 | 0.02 | 0.10 |
| All experts | 4.65 | 0.40 | 1.71 |
| Q8_0 attention | 1.22 | 0.03 | 0.58 |
| Q4_K, Q5_K and Q6_K | 0.49 | 0.004 | 0.14 |
| Small f16 matrices | 0.21 | 0.003 | 0.04 |
| Attention scores (`attn`) | 0.98 | | |
| Relative bias (`relb`) | 0.23 | | |
| Other small kernels | 1.48 | | |

The other small kernels are, in ms a token: prep 0.56, mid 0.28, ctl
0.22, msg 0.16, act 0.08, dsum 0.05, top 0.05, post 0.03, fin 0.03, end
0.01. Prefill is limited by arithmetic, not by memory: the weight
kernels run at 40–55% of the FMA rate, and bytes are under 10% of the
time. Padding adds 10% to the expert MACs.

Expert rows in the same run: 110.6 distinct experts a visit, 37 rows an
expert, 0.91 of the 8-column tiles and 0.66 of the 32-column tiles
filled. The CPU sorts the (token, k) pairs by expert after each visit,
as `uni` does on the GPU for passes.

`gall mmb` times the tile kernels on random bytes, with no model. With
constant weights in place of the decode, a 32-column tile ran Q8_0 at
1,058 against 996 G MAC/s, IQ3_XXS at 1,044 against 922 and NVFP4 at
1,041 against 838. So the decode takes 6–20% of a tile, and the
multiply, at 59% of the FMA rate, takes the rest.

These changes kept the output bits (`gall check` on `ref/2k` and
`ref/8k.b256` with `-ub 1`: no greedy mismatch, kl 1.09e-3 and 1.30e-3,
equal to the present kernels) but were dropped:

- 64-column tiles, so a weight tile is decoded once for 64 rows of the
  batch, not 32. Interleaved in one process, on a 512-column Q8_0
  matrix: 1,129 to 1,197 G MAC/s; IQ3_XXS gate_up on 110 slabs of 40
  columns: 688 to 827; NVFP4 down: 596 to 690; a 128-row f16 matrix:
  723 to 661. In `pp` on 2,048 tokens, interleaved, the present kernels
  gave 133.1, 125.1 and 99.7 tokens/s and the 64-column tiles 118.3,
  126.6 and 105.8 (median 125.1 against 118.3), so no gain was shown.
  128-column tiles leave the registers (14–20 G MAC/s).
- the last visit for only the rows that feed the conv ring (the last 19
  rows, to a tile), without a drafter tap. With 64-column tiles: 115.6,
  120.1 and 107.0 tokens/s in the same series.
- batch size: with 64-column tiles, two interleaved rounds gave 123.6
  and 108.0 tokens/s at `ub` 256, 131.8 and 115.2 at 512, 128.9 and
  130.2 at 1024. This does not separate 512 from 1024, so `ub` stays
  512.

## Long-context attention

At longer contexts, KV reads limit performance at about 85 GB/s. The eight
full-attention visits use int8 rows to estimate attention scores, then
read f16 rows for keys within 1% of the largest weight. `VT=0` disables
this path; the default `VT=0.01` sets the threshold.

Back-to-back measurements on real prompts:

| Context | Int8-assisted GPU time/token | F16 only |
| --- | ---: | ---: |
| 10.5k | 53.6 ms | 54.3 ms |
| 29.9k | 58.6 ms | 61.9 ms |

No greedy token differed over 16 steps. Mean KL divergence against f16
was 4e-9 at 10.5k and 4e-8 at 29.9k. Against `ref/8k.b256`, a 32-step
check found no greedy mismatches and mean KL of 1.30e-3 (1.32e-3 with
f16 only). There is no 100k throughput measurement.

## Vision

`src/vis.*` decodes images through ImageIO and runs the vision encoder on
the GPU. A 640 × 488 image (208 tokens) took 0.8 seconds; a 960 × 1280
image (768 tokens) took 4.2 seconds. Relative output error against the
llama.cpp reference (`gall vis`, `tools/imgref.cpp`) was within 2.9e-3.
The server caches embeddings for
the 32 newest images on disk.

## Measurements and validation

Memory, long-context and vision results here and in the README are
individual runs on 21–22 November 2026; decode and kernel rates are
interleaved medians on a warm machine, 23–24 November 2026. The MacBook
Air has no fan; throughput fell roughly 10% after about 2.5 minutes of
load. Compare changes back to back and record run order.

The comparison baseline is **patched llama.cpp b12600 + Unsloth pins**,
including our local Metal patches, measured on 21–22 November 2026. Its
short-context rate was about 15 tokens/s. Stock upstream Metal at that
date lacked GPU support for these NVFP4 experts and still built a dense
prefill mask; we report no throughput measurement for stock upstream.

The short-context teacher-forced check (`ref/short`) found no greedy
mismatches in 64 steps, with mean KL divergence of 1.05e-4 (maximum
1.92e-3). The `ref/2k` check found none, with mean KL divergence of
9.90e-4 (maximum 2.51e-2). A retrieval test recovered a
code placed 400 tokens into a 31.5k-token prompt with a 100k context
setting. Neither result is a general model-quality benchmark.

```sh
cargo build --release
target/release/gall check "$(gall model)" ref/short -n 64
```

Only one model may hold GPU memory at a time. Every `gall` command that
uses the GPU takes `/tmp/gall.lock`; run other programs through
`lockf /tmp/gall.lock`. For HTTP work use `gall serve "$(gall model)"
--dummy` or `tools/smoke.py`, which do not load the model onto the GPU.

See [reference tools](../tools/README.md) for dump formats and numerical
comparison, and [drafter notes](DRAFT.md) for speculative decoding.
