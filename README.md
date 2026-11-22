# Gall

> **gall** /ɡɔːl/ *noun*
> 1. A growth that forms on a plant where an insect lays its eggs; the oak
>    gall, or oak apple, is the source of the tannin in iron gall ink.
> 2. Bold and impudent behaviour.

Gall runs Thinking Machines' Inkling-2-Mini on an Apple silicon Mac with
24 GB of memory and serves it as an OpenAI-compatible API on your own
machine. The model is a mixture of experts (MoE) with 26 billion
parameters, of which 3.4 billion are active for each token.

When Gall unloads the model to free memory, it first saves the
conversation to disk. The next message loads the model again in about
0.1 seconds and continues from the saved conversation, so it does not
process the whole prompt again.

The engine is a Rust host with Metal kernels. It reads the Unsloth
UD-IQ3_XXS GGUF file in place. It does not use ggml or MLX.

## Requirements

- An Apple silicon Mac with 24 GB of memory. Gall was developed and
  measured on an M2 MacBook Air with macOS 27.2. Other machines and
  versions are untested.
- 14 GB of free disk space for the model files.
- For the command-line install only: Rust (`cargo`) and `curl`.

## App

`Gall.app` is a menu bar app. It needs no terminal.

1. Download `Gall.dmg` from the
   [releases page](https://github.com/ltrgoddard/gall/releases).
2. Drag `Gall.app` to the Applications folder, then open it.
3. On first use, the app checks for 24 GB of memory and 14 GB of free disk
   space. Then it downloads the model files (12.7 GB). The menu shows the
   progress.
4. When the menu shows `Ready`, select `Copy endpoint` and paste the URL
   (`http://127.0.0.1:9940/v1`) into an OpenAI-compatible client.

The oak leaf in the menu bar is filled while the model is loaded or
answering, and an outline when the model is unloaded. The menu shows the
server state, its memory use, and the cache use and speed of the last
answer. `Unload now` releases the model's memory and keeps the
conversation. `Open log` shows the server log.

The settings submenu sets:

- the idle time before an unload: instant, 1 to 60 minutes, or never
  (15 minutes by default)
- the context length: 32k or 100k tokens (100k by default)
- start at login
- access from other devices on the network. This makes a key in
  `~/.cache/gall/key`; `Copy key` copies it.

A change to a setting restarts the server and keeps the conversation.
`Quit Gall` saves the conversation and stops the server.

To build the app from source, run `mac/build`. It makes `target/Gall.app`
and signs it ad hoc. Set `GALL_SIGN` to a signing identity to sign it with
that identity. Also set `GALL_NOTARY` to a `notarytool` keychain profile
name to make a notarized `target/Gall.dmg`.

## Command line

```sh
cargo install --git https://github.com/ltrgoddard/gall
gall up      # downloads the model files on first use, then starts the server
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

## API

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

## Memory

With `--idle-unload N`, the server saves the conversation and unloads the
model after `N` seconds without a request. `POST /park`, a `SIGTERM` (from
`gall down`, a quit, or a logout), and a memory pressure warning from
macOS that lasts 30 seconds do the same. The next chat request loads the
model again and continues from the saved conversation. Gall exits if
memory becomes critically short.

Only one model fits in the GPU memory of a 24 GB Mac. Unload other local
models before you start Gall.

Gall keeps the saved conversations, including the system prompt and
tools, in `~/.cache/gall`. A new conversation with the same start begins
in about one second. The same directory holds the rows of the 32 newest
images and a table for the drafter.

## Speed

On the M2 MacBook Air, Gall writes 20.5 tokens per second at a short
context, 18 at 10.5k tokens and 16 at 30k. It reads a prompt at 106 to 130
tokens per second. With a saved conversation, the first token of an answer
comes in 0.5 to 1.2 seconds. A teacher-forced comparison with llama.cpp
found no greedy mismatches across 64 steps.

## License

Gall uses the MIT license. Parts derive from llama.cpp and ggml.
See `NOTICE`. The model has a separate license.
