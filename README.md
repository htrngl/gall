# Gall

> **gall** /ɡɔːl/ *noun*
> 1. A growth that forms on a plant where an insect lays its eggs; the oak
>    gall, or oak apple, is the source of the tannin in iron gall ink.
> 2. Bold and impudent behaviour.

Gall runs Thinking Machines' Inkling-2-Mini on an Apple silicon Mac with
24 GB of memory and serves it as an OpenAI-compatible API on your own
machine. The model is a mixture of experts (MoE) with 26 billion
parameters, of which 3.4 billion are active for each token. Gall supports
text, images, tool calls and structured JSON output.

The app sits in the menu bar. By default it unloads the model one second
after each answer and saves the conversation state to disk, so the next
message continues without processing the whole prompt again. On an M2
MacBook Air, Gall generates about 20 tokens per second.

The engine is a Rust host with Metal kernels. It reads the Unsloth
UD-IQ3_XXS GGUF file in place. It does not use ggml or MLX.

## Get started

You need an Apple silicon Mac running macOS 15 or later, with at least 24 GB
of memory and 14 GB of free disk space for installation. Saved conversation
state needs additional space. Development and measurements used an M2
MacBook Air running macOS 27.2; other machines and macOS versions have not
been tested.

1. Download `Gall.dmg` from the
   [releases page](https://github.com/ltrgoddard/gall/releases).
2. Drag `Gall.app` to Applications and open it.
3. The current release is not notarized by Apple, so macOS blocks the
   first launch. Select **Done**, then open **System Settings** >
   **Privacy & Security**, find the message about Gall and select
   **Open Anyway**, then confirm. You only do this once.
4. Let Gall download the model files (12.7 GB). Progress appears in the
   menu bar.
5. Select **Copy endpoint** and paste it into a client that supports an
   OpenAI-compatible API. Use `inkling-2-mini` as the model name.

To skip step 3 from a terminal, run
`xattr -dr com.apple.quarantine /Applications/Gall.app` before you open
the app. Compare the download with `Gall.dmg.sha256` from the same release
if you want to check it.

The endpoint is `http://127.0.0.1:9940/v1`. Gall supplies the model server;
you use your own chat or coding client. **Ready** and **Unloaded** both
mean it can accept a request. An unloaded model loads automatically.

For a terminal installation, see [Command line and API](docs/API.md).

## Memory

| State | RAM |
| --- | ---: |
| Waiting for the first request | About 32 MB |
| Idle after an answer | About 51–57 MB |
| Reading a prompt or generating | Budget about 18 GB |

Measured on the M2 Air with an 805-token prompt, a 100k context limit and
the drafter enabled. The active budget combines a measured 5.3 GB server
footprint with up to 12.7 GB of file-backed model weights; it is not a
measurement of resident RAM. The menu bar app adds about 25 MB. Longer
prompts and images change memory use.

**Instant** unload is the app’s default: after one idle second it saves
conversation state and releases the model’s working memory. The leaf
becomes an outline; the next message loads the model automatically.

Conversations of at least 512 tokens can reuse saved state. Shorter ones
are processed again, and your client still sends the history as usual.
Warm reloads took about 0.1 seconds; the first token with a saved prefix
took 0.5–1.2 seconds. Cold starts take longer.

## In the menu bar

The leaf is filled while the model is loaded or answering, and an outline
when unloaded. The menu shows the server’s memory use, how much of the
last prompt was cached, and its generation speed.

- **Unload now** releases memory after the current answer.
- **Unload after** selects Instant, 1, 5, 15 or 60 minutes, or Never.
  Existing saved preferences are kept when you update the app.
- **Context** selects 32k or 100k tokens; the default is 100k.
- **Start at login** opens Gall when you sign in.
- **Allow other devices on the network** enables network access with an
  API key. Use **Copy key** to configure the other device’s client, with
  your Mac’s network address in place of `127.0.0.1`.
- **Telegram bridge** passes your Telegram messages to the model while
  Gall runs. See [Telegram bridge](#telegram-bridge).
- **Open log** opens the server log in Console.

Changing the unload, context or network setting restarts the server.
**Quit Gall** saves eligible conversation state and stops it. Gall runs as
your user and does not prevent your Mac from sleeping, except while the
Telegram bridge is on.

## Telegram bridge

The bridge lets you message the model from your phone while Gall runs on
your Mac. Each message goes to [hax](https://github.com/OleksandrChekhovskyi/hax),
a small coding agent that runs as you and uses Gall's model. The bridge
sends hax's answer back.

1. Install hax: `brew install oleksandrchekhovskyi/hax/hax`, or see
   below for another agent.
2. Message @BotFather in Telegram, send `/newbot`, and keep the token.
   Find your own numeric ID with @userinfobot.
3. Put both in `~/.config/gall/bridge`, and run
   `chmod 600 ~/.config/gall/bridge`:

   ```sh
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_ALLOWED_USERS=12345678   # comma separated
   ```

4. Turn on **Settings** > **Telegram bridge**.

While the bridge is on, the Mac does not go to sleep when it is idle. It
still sleeps when you close the lid. The agent works in `~/Gall`; set
`BOT_DIR` in the file to change this. A run stops after `TIMEOUT` seconds
(default 1800). The log is `~/.cache/gall/bridge.log`.

- Any message continues the same conversation. `/new` starts a new one.
- A message that you send during a run stops it, and then runs. `/stop`
  only stops it. The conversation keeps the completed work.
- A draft at the foot of the chat shows the tool that hax ran last, or
  "working" for another agent.
- A photo is saved in `img/`, and the model looks at it.
- `AGENTS.md` in the folder tells the model to reply without markdown.
  Edit it to give the model other rules.

To use another coding agent, set `AGENT` in the file to a command. The
command reads the message on standard input and writes the answer on
standard output. It gets these variables:

- `GALL_URL`, `GALL_KEY` and `GALL_MODEL`: the endpoint, its key and the
  model name. Set up your agent to use them, or another model.
- `CONTINUE`: empty for the first message of a conversation, `1` after.

For example, hax without the tool drafts:

```sh
AGENT='hax -p ${CONTINUE:+--continue} --provider=llamacpp --model=$GALL_MODEL'
```

hax has no approval prompts. It runs whatever the model decides to run,
as your user. Keep `TELEGRAM_ALLOWED_USERS` to yourself, and keep the
token secret.

## Performance

Recorded on a 24 GB M2 MacBook Air, 21–22 November 2026:

| Work | Gall |
| --- | ---: |
| Generate at short context | 20.5 tokens/s |
| Generate at 10.5k context | 18.0 tokens/s |
| Generate at 29.9k context | 16.2–16.5 tokens/s |
| Read a 512-token prompt | 130 tokens/s |
| Read a 10.5k-token prompt | 106 tokens/s |
| First token with a saved prefix | 0.5–1.2 s |

These are individual runs, not averages. The fanless Air slows by roughly
10% after about 2.5 minutes of load. A configured 100k context is not a
100k throughput measurement; the longest measured generation context
here was 29.9k tokens.

The baseline, **patched llama.cpp b12600 + Unsloth pins** (21 November
2026), generated about 15 tokens/s with the same quantization. Stock
upstream Metal at that date lacked GPU support for these NVFP4 experts
and still built a dense prefill mask; it is not the measured baseline.
A teacher-forced comparison against the patched build found no
greedy mismatches over 64 steps, with mean KL divergence of 1.05e-4. That
checks engine agreement, not the model’s general capability.

See the [engine notes](docs/ENGINE.md) for memory layout and measurement
details, and the [drafter notes](docs/DRAFT.md) for speculative decoding.

## Build from source

With Rust and Apple’s command-line developer tools installed:

```sh
mac/build
open target/Gall.app
```

The build creates `target/Gall.app`, `target/Gall.dmg` and a SHA-256
checksum. By default the app is signed ad hoc for local use. For a public
release, set `GALL_SIGN` to your Developer ID Application identity and
`GALL_NOTARY` to a `notarytool` keychain profile. The build then signs,
notarizes and staples the DMG. See [releasing](docs/RELEASE.md).

For engine development, see [AGENTS.md](AGENTS.md) and the
[reference tools](tools/README.md). Model runs share `/tmp/gall.lock`;
unload other local models first.

## License

Gall is MIT licensed. Parts derive from llama.cpp and ggml; see
[NOTICE](NOTICE). The model has a separate license.
