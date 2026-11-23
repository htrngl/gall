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
token runs. A measured short-context token used 1,071 dispatches and
48.2 ms of GPU time; wall time was about 0.4 ms higher. Weight processing
accounted for 44.2 ms, with small kernels measured at about 4.9 ms.

Expert kernels took about 23 ms per token and were limited by arithmetic.
Three tuning passes produced no further improvement. On 23 November 2026,
two rows per threadgroup raised the NVFP4 and IQ4_NL down kernels from
62–75 GB/s to about 80 GB/s, and one 16-bit load of the two IQ3_XXS grid
indices raised that kernel from 54 to 57 GB/s. Decode uses a concurrent
encoder with a barrier before each dispatch, except among the q, k, v,
rel and gate projections, which read the same input. Back to back, the
short-context rate went from 20.0 to 20.9 tokens/s (49.4 to 47.5 ms of
GPU time per token).

The weight bytes read per token are about 2.9 GB: 1.30 GB of experts,
1.11 GB of Q8_0 attention, 0.32 GB of output head and 0.18 GB of other
attention. At the Q8_0 rate of about 87 GB/s that is 33 ms, or 30
tokens/s, with no time for small kernels. The IQ kernels remain at
39–57 GB/s. Tests that removed parts of the IQ3_XXS unit showed that
the loads alone ran at about 87 GB/s; the scale and sign work took the
rest. Integer sign masks, a sign-mask table, threadgroup copies of the
grids, half-precision dots and other row counts did not help. With no
barriers at all (incorrect results), a token took 42 ms, which bounds
the gain from more concurrency at about 5 ms. The k-quant kernels
unpack four bytes as a word: the Q5_K head read at 73 GB/s (4.4 ms), and
Q6_K at 87 GB/s.

Prefill uses half-precision matrix tiles, as the reference llama.cpp build
does. The `HT` setting in `pfm.metal` selects float precision at about 80%
of that speed.

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

Results here and in the README are individual runs on 21–22 November
2026. The MacBook Air has no fan; throughput fell roughly 10% after about
2.5 minutes of load. Compare changes back to back and record run order.

The comparison baseline is **patched llama.cpp b12600 + Unsloth pins**,
including our local Metal patches, measured on 21–22 November 2026. Its
short-context rate was about 15 tokens/s. Stock upstream Metal at that
date lacked GPU support for these NVFP4 experts and still built a dense
prefill mask; we report no throughput measurement for stock upstream.

The short-context teacher-forced check found no greedy mismatches in 64
steps, with mean KL divergence of 1.05e-4. A retrieval test recovered a
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
