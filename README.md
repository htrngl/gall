# Gall

Gall is an inference engine for Thinking Machines' Inkling-2-Mini on Apple
silicon. The mixture-of-experts (MoE) model has 26 billion parameters,
3.4 billion active parameters, and 48 layer visits per token.

The engine uses a Rust host and Metal kernels to read the Unsloth
UD-IQ3_XXS GGUF file in place. Its four dependencies provide Metal bindings
and libc. It does not use ggml or MLX.

## Requirements

- An Apple silicon Mac with 24 GB of memory. Development and measurements
  used an M2 MacBook Air with a 10-core GPU. Other machines are untested.
- Rust (`cargo`) and `curl`.
- 12 GB of disk space for the model.

## Install and run

```sh
cargo install --git https://github.com/ltrgoddard/gall
gall up          # Downloads the model (12.3 GB) on first use; starts the server on port 9940
gall test        # Checks 64 greedy tokens against a llama.cpp reference
gall down
```

`gall up [ctx] [serve flags]` starts the server in the background.
The default context is 102,400 tokens. Gall uses a GGUF file from the
Hugging Face cache if available. Otherwise, it downloads the model to
`~/.cache/gall`. Set `GALL_MODEL=path` to use a different model file.

Use `gall log` to follow the request log. Use `gall model` to print the
model path.

### API

The server provides an OpenAI-compatible chat API at
`http://127.0.0.1:9940/v1`. It supports these endpoints:

- `/v1/chat/completions`, with streaming, tools, reasoning, images, and
  JSON `response_format`.
- `/v1/models`.
- `/health`.

Configure an OpenAI-compatible client to use the server's base URL.
Send an image as an `image_url` content part in a user or tool message.
The URL can be a `data:` URL or an http URL, in any format that ImageIO
reads. Gall's vision encoder reads `mmproj-F16.gguf` from the model's
directory; `gall up` downloads that file (0.6 GB). An image uses one token
for each 40 x 40 pixels. Images above 1,024 tokens (`GALL_IMAGE_TOKENS`)
are scaled down.

With `response_format` of type `json_object` or `json_schema`, the answer
text is JSON that fits the schema. The server checks `type`, `properties`,
`required`, `additionalProperties: false`, `items`, `enum`, `const`,
`anyOf`, and `oneOf`; it does not check other keywords. It also appends a
system line with the schema to the prompt.

To require authentication, use `gall serve <gguf> --api-key-file F`.
Every request except `/health` must supply the key from `F` in the
`Authorization: Bearer KEY` header.

### Memory and prefix cache

`gall serve <gguf> --idle-unload N` releases model memory after `N` seconds
without a request. Before unloading, the server saves the conversation as
a prefix snapshot. The next chat request reloads the model and continues
from that snapshot. The measured reload time is 0.06 seconds when the file
is in the page cache. The `/health` response includes a `"loaded"` field.

Only one model fits in the tested machine's GPU memory. Gall holds
`/tmp/gall.lock` while a model is loaded and releases the lock when it
unloads the model. A second server or `./wd` waits for this lock.
The `--idle-unload` option lets the server and measurement commands take
turns. Gall exits if memory runs short. Unload other local models first.

Gall stores conversation prefixes, including the system prompt and tools,
in `~/.cache/gall`. A new conversation with the same prefix starts in
about one second.

## Measurements

These results are from one run per measurement on the M2 MacBook Air
on 21 November 2026. The MacBook Air has no fan. Throughput drops by roughly
10% after about 2.5 minutes of load. Rates below are in tokens per second
(t/s).

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

A retrieval test recovered a
code placed 400 tokens into a 31.5k-token prompt with a 100k context.

## Development

```sh
cargo build --release
./wd target/release/gall check $(gall model) ref/short -n 64
```

`./wd` runs a command under the same lock and monitors memory use.
The `ref/` directory holds reference dumps from `tools/refdump`, which uses
the llama.cpp build in `llama/`. See [Reference tools](tools/README.md).
See `src/main.rs` for other commands, including `run`, `pp`, `tok`, and `dr`.

## Repository layout

- `src/gguf.rs`, `gpu.rs`: GGUF memory mapping and the Metal host.
- `src/mv.*`, `grids.metal`: Matrix-vector kernels for the 12 weight formats.
- `src/rq.rs`: Attention weights requantized at load (`RQ`, disabled by default).
- `src/model.rs`, `ops.metal`: Single-token decode, key-value (KV) cache rings, and rollback.
- `src/pf*`: Batched prefill.
- `src/tok.rs`, `serve.rs`, `snap.rs`, `up.rs`: Tokenizer, chat template,
  server, prefix snapshots, and server lifecycle commands.
- `src/vis.*`: Image decoding (ImageIO) and the vision encoder.
- `src/dr*`: The dspark drafter. See the limitations below.
- [Engine brief](docs/ENGINE.md) and [drafter notes](docs/DRAFT.md).
- `tools/`: Tools for reference dumps from llama.cpp.
- `llama/`: The patched llama.cpp reference build and its
  [notes](llama/PLAN.md). `llama/build` installs it for the llama app.

## Implementation status and limitations

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
- The drafter's batched verification produces the same text as greedy
  decoding. Verification of four rows takes 250 ms, compared
  with 52 ms per serial token. The block size remains 0.
  The drafter needs expert kernels that dequantize weights once per batch.
- Prefill uses half-precision tiles, as llama.cpp does. The `HT` setting in
  `pfm.metal` selects float precision at approximately 0.8 times the speed.
- The vision encoder runs on the GPU. A 640 x 488 image (208 tokens) takes
  0.8 s and a 960 x 1280 image (768 tokens) takes 4.2 s. Its output is within 2.9e-3 (relative) of llama.cpp's
  (`gall vis`, `tools/imgref.cpp`). The rows of the 32 newest images are
  kept in `~/.cache/gall/img`.
- The server runs one chat at a time; other chats wait in a queue, and
  `/health` and `/v1/models` answer meanwhile.
- Unsupported features include GBNF grammars and drafting in the server.
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

## License

Gall uses the MIT license. Parts derive from llama.cpp and ggml.
See `NOTICE`. The model has a separate license.
