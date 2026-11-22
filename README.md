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
| Decode, short context | 15 t/s | 19.5 t/s cold, approximately 18 t/s warm |
| Decode, 8k context | 12.4 t/s | 17.6 t/s |
| Decode, 31.5k context (real prompt) | No reliable measurement | 13.8–14.6 t/s |
| Prefill, 512 tokens | 121 t/s | 108–125 t/s |
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

- Weight processing accounts for 45 of 51 ms per token. Each format takes
  7–12 ps per element. Two tuning passes on the expert gate-up kernels
  produced no further improvement.
- KV cache reads limit performance at longer contexts, at 81 GB/s.
  Decode takes 56.7 ms at 8k, 67 ms at 32k, and 102 ms at 100k.
  The 32k and 100k measurements use synthetic inputs.
- `KV=q8` stores full-attention caches as int8 and halves cache memory at
  100k context. It does not improve speed and is disabled by default.
  Its prefill path is untested.
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
- A proposed 4-bit attention requantization has a projected cost of
  3.7 ms per token. It requires the 53 GB original model and a quality check.

## License

Gall uses the MIT license. Parts derive from llama.cpp and ggml.
See `NOTICE`. The model has a separate license.
