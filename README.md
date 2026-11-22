# Gall

> **gall** /ɡɔːl/ *noun*
> 1. A growth that forms on a plant where an insect lays its eggs; the oak
>    gall, or oak apple, is the source of the tannin in iron gall ink.
> 2. Bold and impudent behaviour.

Gall is an inference engine for Thinking Machines' Inkling-2-Mini on Apple
silicon. The mixture-of-experts (MoE) model has 26 billion parameters,
3.4 billion active parameters, and 48 layer visits per token.

The engine is a Rust host with Metal kernels. It reads the Unsloth
UD-IQ3_XXS GGUF file in place and serves an OpenAI-compatible chat API.
Its four dependencies provide Metal bindings and libc. It does not use
ggml or MLX.

## Requirements

- An Apple silicon Mac with 24 GB of memory. Development and measurements
  used an M2 MacBook Air with a 10-core GPU and macOS 27.2. Other machines
  and versions are untested.
- Rust (`cargo`) and `curl`.
- 13 GB of disk space for the model files.

## App

`Gall.app` is a menu bar app for the same server. It needs no terminal.

1. Download `Gall.dmg` from the
   [releases page](https://github.com/ltrgoddard/gall/releases).
2. Drag `Gall.app` to the Applications folder, then open it.
3. On first use, the app checks for 24 GB of memory and 14 GB of free disk
   space. Then it downloads the model files (12.7 GB). The menu shows the
   progress.
4. When the menu shows `ready`, select `copy endpoint` and paste the URL
   (`http://127.0.0.1:9940/v1`) into an OpenAI-compatible client.

The menu shows the server state, its memory use, and the cache use and
speed of the last answer. `unload now` releases the model's memory and
keeps the conversation. The settings submenu sets the idle time before an
unload (instant, 1 to 60 minutes, or never; 15 minutes by default), the
context length, start at login, and
access from other devices on the network. Network access makes a key in
`~/.cache/gall/key`; `copy key` copies it. Quit stops the server.

To build the app from source, run `mac/build`. It makes `target/Gall.app`
and signs it ad hoc. Set `GALL_SIGN` to a signing identity to sign it with
that identity. Also set `GALL_NOTARY` to a `notarytool` keychain profile
name to make a notarized `target/Gall.dmg`.

## Install and run

```sh
cargo install --git https://github.com/ltrgoddard/gall
gall up      # downloads the model files on first use, then starts the server
gall test    # checks 64 greedy tokens against a llama.cpp reference
curl http://127.0.0.1:9940/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
gall down
```

`gall up [ctx] [serve flags]` starts the server in the background on port
9940. The default context is 102,400 tokens. On first use it downloads
three files from `unsloth/Inkling-2-Mini-GGUF`: the model (11.7 GB), the
vision encoder `mmproj-F16.gguf` (0.6 GB), and the drafter
`dspark-Inkling-2-Mini-Q8_0.gguf` (0.4 GB). Gall uses the files in the
Hugging Face cache if the model is there. Otherwise, it downloads them to
`~/.cache/gall`. Set `GALL_MODEL=path` to use a different model file; the
other two files are read from its directory.

| Command | Result |
| --- | --- |
| `gall up [ctx] [serve flags]` | Starts the server in the background. |
| `gall down` | Stops the server. |
| `gall log` | Follows the request log. |
| `gall model` | Prints the model path, after a download if necessary. |
| `gall test` | Compares 64 greedy tokens with a llama.cpp reference. |
| `gall serve <gguf> [flags]` | Runs the server in the foreground. |

### API

The server provides an OpenAI-compatible chat API at
`http://127.0.0.1:9940/v1`. Configure an OpenAI-compatible client to use
that base URL. The server supports these endpoints:

- `/v1/chat/completions`, with streaming, tools, reasoning, images, and
  JSON `response_format`.
- `/v1/models`.
- `/health`, with `loaded`, `busy` (a chat runs), and `last` (the
  `timings` of the last chat, or `null`). It needs no key.
- `POST /park`, which saves the conversation and unloads the model after
  the current chat.

The server runs one chat at a time. Other chats wait in a queue, and
`/health` and `/v1/models` answer during a chat.

Send an image as an `image_url` content part in a user or tool message.
The URL can be a `data:` URL or an http URL, in any format that ImageIO
reads. An image uses one token for each 40 x 40 pixels. Images above
1,024 tokens (`GALL_IMAGE_TOKENS`) are scaled down.

With a `response_format` of type `json_object` or `json_schema`, the answer
text is JSON that fits the schema. The server checks `type`, `properties`,
`required`, `additionalProperties: false`, `items`, `enum`, `const`,
`anyOf`, and `oneOf`. It does not check other keywords. It also appends a
system line with the schema to the prompt. GBNF grammars are not supported.

### Server flags

| Flag | Result |
| --- | --- |
| `--host ADDR` | Listens on address `ADDR` (default `127.0.0.1`). An address other than loopback requires `--api-key-file`. |
| `--port N` | Listens on port `N` (default 9940). |
| `--ctx N` | Sets the context length in tokens (default 102,400). |
| `--alias NAME` | Sets the model name that the API reports. |
| `--effort LEVEL` | Sets the reasoning effort for requests that give none (default `none`). |
| `--api-key-file F` | Requires `Authorization: Bearer KEY` with the key in `F`, on every request except `/health`. |
| `--idle-unload N` | Releases the model's memory after `N` seconds without a request. `POST /park`, memory pressure and `SIGTERM` also release it. |
| `--no-draft` | Disables the drafter. |

### Memory and prefix cache

With `--idle-unload N`, the server saves the conversation as a prefix
snapshot and unloads the model after `N` seconds without a request. The
next chat request reloads the model and continues from that snapshot. The
measured reload time is 0.08 seconds when the files are in the page cache.
The `/health` response includes a `"loaded"` field. `POST /park`, a
`SIGTERM` (from `gall down`, a quit, or a logout), and a memory pressure
warning from macOS also save the conversation and unload the model.

Only one model fits in the tested machine's GPU memory. Gall holds
`/tmp/gall.lock` while a model is loaded and releases the lock when it
unloads the model. A second server or `./wd` waits for this lock.
The `--idle-unload` option lets the server and measurement commands take
turns. Gall exits if memory becomes critically short. Unload other local
models first.

Gall stores conversation prefixes, including the system prompt and tools,
in `~/.cache/gall`. A new conversation with the same prefix starts in
about one second. The same directory holds the rows of the 32 newest
images and the drafter's precomputed table.

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

## License

Gall uses the MIT license. Parts derive from llama.cpp and ggml.
See `NOTICE`. The model has a separate license.
