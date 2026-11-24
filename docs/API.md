# Command line and API

Gall serves one model at `http://127.0.0.1:9940/v1`. It processes one chat
at a time; additional requests wait in a queue.

## Install and run

The command-line install needs Rust (`cargo`), Apple’s command-line
developer tools and `curl`.

```sh
cargo install --git https://github.com/ltrgoddard/gall
gall up 102400 --idle-unload 1
curl http://127.0.0.1:9940/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Hello"}]}'
gall down
```

`gall up [ctx] [serve flags]` starts a background server. Supply the context
length before server flags. With no arguments, it uses 102,400 tokens and
keeps the model loaded after the first chat. The example above uses the
app’s one-second idle timer. The server starts unloaded; `/health` and
`/v1/models` do not load it. If a server is already running, `gall up`
leaves it alone; stop it before changing its flags.

On first use, Gall downloads three files from
`unsloth/Inkling-2-Mini-GGUF`: the UD-IQ3_XXS model (11.7 GB),
`mmproj-F16.gguf` (0.6 GB) and `dspark-Inkling-2-Mini-Q8_0.gguf` (0.4 GB).
It looks in the Hugging Face cache first, then `~/.cache/gall`.
`GALL_MODEL` overrides the model path and skips the download; the vision
encoder and drafter are read from the same directory. Gall supports this
model architecture, not arbitrary GGUF models.

| Command | Purpose |
| --- | --- |
| `gall up [ctx] [serve flags]` | Start the background server. |
| `gall down` | Save eligible conversation state and stop the server. |
| `gall log` | Follow the server log. |
| `gall model` | Find or download the model and print its path. |
| `gall test` | Compare 64 greedy tokens with the bundled llama.cpp reference. |
| `gall serve <gguf> [flags]` | Run the server in the foreground. |

The PID, log and optional network key are in `~/.cache/gall` as
`serve.pid`, `serve.log` and `key`. `serve.old` is the log of the run
before. The app starts the server again after a panic.

### Server flags

| Flag | Result |
| --- | --- |
| `--host ADDR` | Listens on address `ADDR` (default `127.0.0.1`). An address other than loopback requires `--api-key-file`. |
| `--port N` | Listens on port `N` (default 9940). |
| `--ctx N` | Sets the context length in tokens (default 102,400). |
| `--alias NAME` | Sets the model name that the API reports. |
| `--effort LEVEL` | Sets the reasoning effort for requests that give none (default `none`). |
| `--api-key-file F` | Requires `Authorization: Bearer KEY` or `x-api-key: KEY` with the case-sensitive key in `F`, on every request except `/health`. |
| `--idle-unload N` | Unloads after `N` idle seconds. `1` matches the app’s Instant setting; `0` disables the timer (the CLI default). |
| `--no-draft` | Disables the drafter. |

## API

The server provides an OpenAI-compatible chat API at
`http://127.0.0.1:9940/v1`. Configure an OpenAI-compatible client to use
that base URL. The server supports these endpoints:

- `/v1/chat/completions`, with streaming, tools, reasoning, images, and
  JSON `response_format`.
- `/v1/messages`, the Anthropic Messages API, with streaming, tools,
  thinking, images, and `stop_sequences`. See below.
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

With the drafter loaded, the server verifies proposed tokens in batches,
which round differently from one token at a time. A temperature 0 answer
is then a greedy decode as close to the llama.cpp reference as serial
decode, and the same from run to run, but it can differ from the serial
answer once a near-tie occurs, typically after 100–300 tokens. Sampled
answers with the same seed can differ likewise. `--no-draft` gives serial
output. See [drafter notes](DRAFT.md#exactness).


### Anthropic Messages API

`/v1/messages` translates each request to a chat completion and the
answer back, so both APIs share the prompt cache and saved state. To use
it with Bricolage, set `BRIC_URL=http://127.0.0.1:9940/v1/messages` and
`BRIC_MODEL=inkling-2-mini`. The server maps these fields:

- `system`, a string or text blocks.
- `text`, `image` (base64 or url), `tool_use`, `tool_result` and
  `thinking` blocks. Other block types give a 400 error.
- `tools` with an `input_schema`. Server tools, such as `web_search`,
  are left out.
- `thinking` of type `disabled` sets the reasoning effort to `none`.
  `output_config.effort` sets it otherwise.
- `output_config.format` or `output_format` of type `json_schema`, as
  for `response_format`.
- `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` and
  `stream`.

`tool_choice`, `metadata` and `cache_control` are ignored. A thinking
block has an empty `signature`. `usage.input_tokens` counts the whole
prompt, including the cached part. `/v1/messages/count_tokens` is not
supported.

## Unloading and saved state

`POST /park` requests an unload after the current chat. A `SIGTERM` also
saves eligible state before exiting. A macOS memory-pressure warning
lasting 30 seconds requests an unload; critically low memory makes Gall
exit. The idle timer runs between requests, so queued chats can finish
without unloading between them.

Gall saves model state for conversations of at least 512 tokens. On the
next request it restores a compatible prefix and processes the remaining
tokens. Clients must still send the conversation history: snapshots are
an inference cache, not a chat history API. Short conversations are
processed again after unloading.

Snapshots contain token IDs and model state, including the system prompt
and tool definitions. They live in `~/.cache/gall` alongside cached image
embeddings and the drafter’s precomputed table. Allow extra disk space for
these files beyond the model download.
