# Models: which to test and how to connect them

## Which models to test

Any model one of the adapters can reach can be tested. A reference set:

**Closed models**, through the local Claude Code or Codex login or the provider's API:

| Model | `--model` | `--effort` | Notes |
|---|---|---|---|
| Claude Opus 5.5 | `claude-opus-5-5` | `medium` | priced: the USD budget applies |
| GPT-6.1 Sol | `gpt-6.1-sol` | `medium` | Codex reports no cost: the run has no USD limit (tokens, wall time and GPU hours still apply) |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | `medium` | a cheaper Claude point |

Runs with another effort land in a separate group of the results table, so keep the effort the
same within one comparison.

**Open-weight models**, self-hosted with vLLM 0.30 ([recipes](#open-weight-recipes)):

| Model | Weights | GPUs (A100 80 GB) | Notes |
|---|---|---|---|
| Qwen3.8-27B | BF16 | 4 | dense; thinking on |
| Qwen3.8-Flash-Next | BF16 (an FP8 checkpoint also works) | 8 | MoE, 125B total, 6B active; thinking on |
| gemma-4-31B-it | BF16 | 4 | dense; thinking switched on in the chat template |
| MiniMax-M3 | MXFP8 | 8 | MoE; on A100 the MoE runs through the Marlin kernels |
| GLM-5.3-Flash | as released | Hopper or newer | its sparse MLA attention has no A100 kernel in vLLM 0.30 |

## Connecting a model

A run takes two choices: the model (`--model`) and how to reach it (`--auth`). `cli` (the default)
uses a local Claude Code or Codex login and needs no key; `api` uses an API key or an
OpenAI-compatible server. The table of names, adapters, endpoints and key variables, and every
adapter parameter, are in [configuration.md](configuration.md#models-and-access-modes).

```bash
# Claude through the local Claude Code login (default, no API key needed)
rsib run new --task <task> --model claude-opus-5-5 --effort medium --mode full

# GPT through the local Codex login (default, no API key needed)
rsib run new --task <task> --model gpt-6.1-sol --effort medium --mode full

# provider APIs: Claude (ANTHROPIC_API_KEY), GPT (OPENAI_API_KEY), Gemini (GEMINI_API_KEY),
# Doubao on Volcano Ark (ARK_API_KEY)
rsib run new --task <task> --model claude-opus-5-5 --auth api --effort medium --mode full
rsib run new --task <task> --model gemini-<version> --auth api --mode full

# any OpenAI-compatible server (vLLM, SGLang, NIM, ...)
rsib run new --task <task> --model 'openai:<served name>@http://<host>:<port>/v1' \
    --model-params '{"api_key_env": "MY_KEY"}'
```

Only `claude-*` and `gpt-*` names have a `cli` mode; other models need `--auth api`. Full specs
`<adapter>:<model>[@<base_url>]` (`claude-cli:`, `codex-cli:`, `anthropic:`, `openai:`, `mock:`)
select the adapter and imply the mode.

## Claude Code and Codex logins

`claude-cli` drives `claude -p` as a plain model: Claude Code's own system prompt, tools,
settings, hooks, plugins, MCP servers and CLAUDE.md files are all switched off, tool calls come
back as schema-checked JSON, each conversation continues one Claude Code session (`--resume`), and
the cost of every call is taken from the CLI's report. The orchestrator must run on a machine where
`claude` is signed in.

`codex-cli` does the same with `codex exec`: Codex's own instructions, shell and other built-in
tools, user config, rules, skills, plugins and AGENTS.md are switched off, the harness prompt is the
model's instructions, replies are forced into a JSON schema, and each conversation continues one
Codex thread (`codex exec resume`). Codex reports tokens but no cost, so the USD budget applies only
when the model is priced in `pricing.toml`. Each CLI adds one short block that cannot be switched
off (the account e-mail for Claude Code, a multi-agent notice for Codex); neither carries task
instructions.

To use a non-default login profile, set `RSIB_CLAUDE_CONFIG_DIR` (passed to `claude` as
`CLAUDE_CONFIG_DIR`) or `RSIB_CODEX_HOME` (passed to `codex` as `CODEX_HOME`). Both adapters wait
out a provider's session or usage limit (`outage_wait_s`), and the wait is not charged as wall time.

## API keys

Keys are read from the environment, or else from a dotenv file: `$RSIB_ENV_FILE` when it is set,
otherwise the first of `~/.config/rsib/.env` and `<repo>/.env` (gitignored) that has the key. Use
`api_key_env` in `--model-params` for a key stored under another name, e.g.
`{"api_key_env": "GPT_API_KEY"}`. Only the orchestrator reads the keys; phases, experiments and
evaluators never see them. `rsib run new` prints which variable and file will be used, never the
value. Do not put a key itself into `--model-params`: it is committed with the run.

## OpenRouter

Use `openai:<id>@https://openrouter.ai/api/v1`, with the OpenRouter model id in place of `<id>`.
Store `OPENROUTER_API_KEY` in the [RSIB env file](#api-keys) and select it with `api_key_env`:

```bash
rsib run new --task <task> --model 'openai:<id>@https://openrouter.ai/api/v1' --mode full \
    --model-params '{"api_key_env": "OPENROUTER_API_KEY", "outage_wait_s": 7200,
                     "extra_body": {"provider": {"order": ["<provider slug>"],
                                                "allow_fallbacks": false,
                                                "require_parameters": true},
                                    "reasoning": {"enabled": true}}}'
```

For comparable runs, [pin the provider](https://openrouter.ai/docs/guides/routing/provider-selection):
`provider.order` lists the providers to try, `allow_fallbacks: false` prevents routing outside
that list, and `require_parameters: true` requires support for every requested parameter.
Enable [reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) through
`extra_body.reasoning`; use `--effort` to set `reasoning_effort` on models that support it.
The adapter preserves returned reasoning in the raw assistant message for later turns.

After the SDK's retries, HTTP 402 (credits exhausted) and 429 are retried within `outage_wait_s`,
alongside connection failures and 502/503/504. A 402 can mean exhausted prepaid credits or credits
tied up by in-flight requests; waiting lets a human top up the account. Both 402 and 429 use
`Retry-After` or `retry-after-ms` when valid, otherwise `outage_backoff_s`; each sleep is capped at
120 seconds. The outage wait is excluded from the wall budget. With the default
`outage_wait_s: 0`, a 402 raises `ModelError` immediately. It also raises `ModelError` when the
outage window expires.
When returned, the upstream `provider` and completion id (`response_id`) are recorded in
[`records/usage.jsonl`](records.md#usagejsonl).

## Self-hosted models

Any server with an OpenAI-compatible chat completions API works through the `openai` adapter. What
a run needs from the server:

- **Tool calls.** The harness acts only through tool calls, so the server must return them as
  structured `tool_calls` (in vLLM: `--enable-auto-tool-choice` with the model's
  `--tool-call-parser`).
- **Reasoning.** For thinking models, use the server's reasoning parser, so that the reasoning comes
  back separately or stays in the content; either way the adapter replays it in later turns.
  `--effort` is sent as `reasoning_effort`; switches of the chat template go in `extra_body`, e.g.
  `--model-params '{"extra_body": {"chat_template_kwargs": {"enable_thinking": true}}}'`.
- **Context length.** At least the protocol's `max_tokens_per_call` (64000) plus the conversation;
  a run with long phases needs 128K or more.
- **Sampling.** Set it on the server (for example the model's own generation config); the adapter
  does not send temperature or top-p.

A run lasts about as long as its wall budget (10 hours for most tasks), which is often longer than
one serving job may run on a cluster. Serve the model behind an address that stays the same, start
the next serving job before the current one ends, and let the adapter ride out the switch with
`outage_wait_s` (default 0 for the `openai` adapter, i.e. fail at once): calls are retried every
`outage_backoff_s` seconds until the server answers or the wait runs out. The time the model is
unreachable is not charged as wall time, and neither is the time experiment and evaluation jobs
wait in a queue ([configuration](configuration.md#when-does-a-run-stop)).

```bash
rsib run new --task <task> --model 'openai:<served name>@http://<host>:<port>/v1' --effort high \
    --model-params '{"api_key_env": "MY_KEY", "timeout_s": 3600, "max_retries": 2,
                     "outage_wait_s": 7200, "outage_backoff_s": 30}' \
    --notes '<weights and precision, tensor parallel size, parsers, sampling>'
```

Before a long run, check the server with a two-turn exchange the way the harness uses it: a first
reply that calls a tool, then, after the tool result, a reply that calls another tool or answers.
A short `--mode base` run with a small budget does the same through the whole pipeline. Record how
the model was served in `--notes`; it is kept in `records/run.json`.

### Open-weight recipes

Tested serving setup: vLLM 0.30 (on NVIDIA driver 535, the CUDA 12.9 build),
A100 80 GB GPUs, `--max-model-len 262144` unless noted, `--enable-prefix-caching`. Add
`--limit-mm-per-prompt '{"image": 16, "video": 0}'` for tasks that send images to the model
(`vsearch`, `hstar-search`).

| Model | `vllm serve` flags | `--model-params` beyond the common ones |
|---|---|---|
| Qwen3.8-27B | `--tensor-parallel-size 4 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml` | none (thinking is on by default) |
| Qwen3.8-Flash-Next | as Qwen3.8-27B with `--tensor-parallel-size 8` | as Qwen3.8-27B |
| gemma-4-31B-it | `--tensor-parallel-size 4 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 64}'` | `"extra_body": {"chat_template_kwargs": {"enable_thinking": true}}` |
| MiniMax-M3 | `--tensor-parallel-size 8 --block-size 128 --enable-auto-tool-choice --tool-call-parser minimax_m3 --reasoning-parser minimax_m3 --override-generation-config '{"temperature": 1.0, "top_p": 0.95}'`; on A100 add `--moe-backend marlin`; always `--limit-mm-per-prompt` with `"video": 0` (its video processor fails with transformers 5.18) | `"extra_body": {"chat_template_kwargs": {"thinking_mode": "enabled"}}` |
| MiniMax-M2.7 | `--tensor-parallel-size 8 --max-model-len 196608 --trust-remote-code --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 40}'`; A100 cannot run its block-FP8 weights, so dequantize them to BF16 first | none |

The common `--model-params` of these setups: `{"api_key_env": "<key variable>", "timeout_s": 3600,
"max_retries": 2, "outage_wait_s": 7200, "outage_backoff_s": 30}`.

## Notes for comparable measurements

- **The model and its effort are fixed per run** and recorded in `records/run.json`. The harness
  cannot change them.
- **Refusal fallbacks are not enabled.** Anthropic's server-side fallback would answer some
  requests with a different model, which would contaminate a per-model result. A refusal ends the
  phase and is recorded.
- **USD metering uses `src/rsibench/models/pricing.toml`** (`claude-cli`: the cost the CLI reports).
  Check the prices before a real run. A model missing from the table runs without a USD limit;
  token, wall-clock and GPU limits still apply, and `usd_metering: false` is recorded.
- Assistant turns are replayed exactly as returned (thinking blocks included), and context
  compaction starts a new conversation from a summary instead of editing old turns.
- Every adapter gets the same canonical context: ordered text and image parts. A change its
  transport forces (Codex, for example, shows images before the text) is recorded as a *lowering*
  in `usage.jsonl`; a run created with `--exact-context` refuses such calls instead
  ([harness](harness.md#context-fidelity)).
