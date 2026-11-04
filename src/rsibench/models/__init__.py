"""Model adapters. Specs look like `<adapter>:<model>[@<base_url>]`.

    anthropic:claude-opus-5-5          (API key, official SDK)
    claude-cli:claude-opus-5-5         (the local Claude Code login, no API key)
    codex-cli:gpt-5.6-sol              (the local Codex login, no API key)
    openai:gpt-5.6-sol
    openai:Qwen/Qwen3-32B@http://localhost:8000/v1
    mock:scripted | mock:finish

`rsib run new` also takes a bare model name plus an access mode (`resolve_model`):

    claude-opus-5-5 --auth cli         -> claude-cli:claude-opus-5-5 (the default mode)
    claude-opus-5-5 --auth api         -> anthropic:claude-opus-5-5, key ANTHROPIC_API_KEY
    gpt-5.6-sol --auth cli             -> codex-cli:gpt-5.6-sol
    gpt-5.6-sol --auth api             -> openai:gpt-5.6-sol, key OPENAI_API_KEY
    gemini-... / doubao-... --auth api -> openai:<model> at the provider's OpenAI-compatible URL
    any other name --auth api          -> openai:<model>[@<base_url>], key OPENAI_API_KEY
"""

from __future__ import annotations

from .base import ChatModel, ModelError, ModelResponse

ADAPTERS = ("anthropic", "claude-cli", "codex-cli", "openai", "mock")
AUTH_MODES = ("cli", "api")
ADAPTER_AUTH = {"claude-cli": "cli", "codex-cli": "cli", "anthropic": "api", "openai": "api"}

# Bare model names by prefix: the adapter for each access mode (None: not available) and the
# default parameters of the api adapter. Names that match no prefix are OpenAI-compatible.
PROVIDERS = (
    ("claude-", {"cli": "claude-cli", "api": "anthropic"}, {}),
    ("gpt-", {"cli": "codex-cli", "api": "openai"}, {}),
    ("gemini-", {"cli": None, "api": "openai"},
     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
      "api_key_env": "GEMINI_API_KEY"}),
    ("doubao-", {"cli": None, "api": "openai"},
     {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "api_key_env": "ARK_API_KEY"}),
)
OTHER = ({"cli": None, "api": "openai"}, {})


def resolve_model(model: str, auth: str | None = None) -> tuple[str, str | None, dict]:
    """A `--model` value and access mode -> (spec, auth, default adapter params).

    A full `<adapter>:<model>` spec passes through unchanged; `auth`, if given, must match its
    adapter. A bare name is resolved with `auth` (default "cli"). The params are defaults only:
    `--model-params` overrides them.
    """
    if auth is not None and auth not in AUTH_MODES:
        raise ValueError(f"unknown access mode {auth!r}; use cli or api")
    adapter, sep, _ = model.partition(":")
    if sep and adapter in ADAPTERS:
        implied = ADAPTER_AUTH.get(adapter)
        if auth is not None and implied is not None and auth != implied:
            raise ValueError(f"{model} uses {implied} access; drop --auth {auth} or pass a bare "
                             f"model name")
        return model, implied, {}
    auth = auth or "cli"
    name, at, base_url = model.partition("@")
    adapters, defaults = next(((a, d) for prefix, a, d in PROVIDERS if name.startswith(prefix)),
                              OTHER)
    adapter = adapters[auth]
    if adapter is None:
        raise ValueError(f"{name} has no CLI access mode (only claude-* via Claude Code and gpt-* "
                         f"via Codex); use --auth api (key from the environment or the .env file)")
    if at and adapter != "openai":
        raise ValueError(f"a custom endpoint (@{base_url}) needs an OpenAI-compatible model")
    params = dict(defaults)
    if at:
        params.pop("base_url", None)
    return f"{adapter}:{model}", auth, params


def parse_spec(spec: str) -> tuple[str, str, dict]:
    if ":" not in spec:
        raise ValueError(f"model spec {spec!r} must look like <adapter>:<model>")
    adapter, rest = spec.split(":", 1)
    params: dict = {}
    if "@" in rest and adapter == "openai":
        rest, base_url = rest.split("@", 1)
        params["base_url"] = base_url
    return adapter, rest, params


def load_model(spec: str, effort: str | None = None, params: dict | None = None) -> ChatModel:
    adapter, name, spec_params = parse_spec(spec)
    merged = {**spec_params, **(params or {})}
    if adapter == "anthropic":
        from .anthropic_model import AnthropicModel
        return AnthropicModel(name, effort, merged)
    if adapter == "openai":
        from .openai_model import OpenAIModel
        return OpenAIModel(name, effort, merged)
    if adapter == "claude-cli":
        from .claude_cli import ClaudeCLIModel
        return ClaudeCLIModel(name, effort, merged)
    if adapter == "codex-cli":
        from .codex_cli import CodexCLIModel
        return CodexCLIModel(name, effort, merged)
    if adapter == "mock":
        from .mock import FinishModel, ScriptedModel
        if name == "scripted":
            return ScriptedModel("mock-scripted", effort, merged)
        if name == "finish":
            return FinishModel("mock-finish", effort, merged)
        raise ValueError(f"unknown mock model {name!r}; use mock:scripted or mock:finish")
    raise ValueError(f"unknown model adapter {adapter!r}; use anthropic, claude-cli, codex-cli, "
                     f"openai or mock")


__all__ = ["ChatModel", "ModelError", "ModelResponse", "load_model", "parse_spec",
           "resolve_model"]
