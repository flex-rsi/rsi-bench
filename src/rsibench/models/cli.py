"""What the CLI transports (claude-cli, codex-cli) share.

A neutral conversation maps to one CLI session, which holds the history itself: a call that
continues the session sends only the delta (context.delta), the user-side messages after the last
assistant reply. A call that starts a session sends the whole history (context.replay), its
assistant turns as text, because a CLI session holds only the replies it produced: the first call
of a conversation, and any call whose history the session does not hold exactly (earlier messages
changed, or assistant turns the session did not produce, e.g. a history rebuilt after a
compaction or one a task's solution sends under a new session). Tools are described in the
system prompt because the CLIs run without native tool calling.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import context
from .base import ModelError

RETRYABLE = ("rate limit", "overloaded", "usage limit", "session limit", "limit reached",
             "529", "503", "502", "timeout", "temporarily", "try again")
# A provider limit that lasts until it resets (a subscription's session or usage limit): calls wait
# for it, up to the adapter's outage_wait_s, instead of failing the phase.
LIMITS = ("usage limit", "session limit", "weekly limit", "daily limit", "monthly limit",
          "limit reached", "rate limit", "quota", "hit your")
# A login that cannot be used (an account on hold, logged out, an expired token): calls wait the
# same way, polling every limit_poll_s, so that signing in again (another account) lets the run go
# on instead of failing every phase until the run ends idle.
ACCOUNT = ("account is on hold", "not logged in", "please run /login", "invalid api key",
           "oauth token has expired", "authentication_error")
# "resets 9:20pm", "resets at 10pm (America/Chicago)", "resets Oct 10, 10pm (America/Chicago)"
RESETS = re.compile(r"resets?\s+(?:at\s+)?(?:(?P<mon>[A-Z][a-z]{2})[a-z]*\.?\s+(?P<day>\d{1,2}),?\s+"
                    r"(?:at\s+)?)?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<half>[ap]\.?m\.?)?"
                    r"(?:\s*\((?P<tz>[A-Za-z_]+(?:/[A-Za-z0-9_+-]+)+|UTC)\))?", re.I)
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
CONTINUE = "Continue."
NEW = {"id": None, "seen": 0, "digest": None, "lowerings": ()}


def tool_block(tools: list[dict]) -> str:
    return "\n\n".join(f"### {t['name']}\n{t['description']}\n"
                       f"input schema: {json.dumps(t['input_schema'], sort_keys=True)}"
                       for t in tools)


NO_TOOLS_PROTOCOL = """

# Reply format

Every reply is a JSON object with one field, "message": your whole reply, exactly as you would
write it in plain text.
"""


def reply_protocol(tool_protocol: str, tools: list[dict]) -> str:
    """What the adapter appends to the system prompt: the tool protocol and the tools, or for a call
    without tools (a model call made by a task's solution) only the reply format, so that nothing
    asks the model to act through tools it does not have."""
    return tool_protocol + tool_block(tools) if tools else NO_TOOLS_PROTOCOL


def tool_name_schema(tools: list[dict]) -> dict:
    """JSON schema of a tool call's name in the reply schema: one of the tools; a call without
    tools gets no enum (an empty one is not a valid schema)."""
    names = [t["name"] for t in tools]
    return {"type": "string", "enum": names} if names else {"type": "string"}


def session_label(meta: dict) -> str:
    """The CLI session of a conversation: one per grant, so concurrent consumers (experiments,
    candidates) never continue each other's sessions."""
    return (f"{meta.get('phase')}|{meta.get('task_id')}|{meta.get('round')}|{meta.get('grant')}"
            f"|{meta.get('session')}")


def session(sessions: dict, meta: dict, messages: list[dict]) -> dict:
    """State ({"id", "seen", "digest", "lowerings"}) of the session this call continues, or NEW
    (id None: start a session and replay the history) unless the session holds exactly the
    messages before its last reply: that reply in place, the earlier messages unchanged and no
    assistant turn after it that the session did not produce."""
    state = sessions.get(session_label(meta))
    if state is None or not state["id"] or len(messages) < state["seen"] \
            or messages[state["seen"] - 1]["role"] != "assistant" \
            or any(m["role"] == "assistant" for m in messages[state["seen"]:]) \
            or context.digest(messages[:state["seen"] - 1]) != state["digest"]:
        return dict(NEW)
    return state


def remember(sessions: dict, meta: dict, session_id: str | None, messages: list[dict],
             applied: list[str], totals: dict | None = None) -> None:
    """Record that the session received `messages` (lowered by `applied`) and replied, with the
    running `totals` the CLI reported for it (per_call). The next call must hold that reply at
    messages[seen - 1] and sends only what follows it."""
    sessions[session_label(meta)] = {"id": session_id, "seen": len(messages) + 1,
                                     "digest": context.digest(messages), "lowerings": applied,
                                     "totals": dict(totals or {})}


def per_call(state: dict, session_id: str | None, totals: dict) -> dict:
    """This call's share of the running totals a CLI reports for a whole session (claude-cli's
    cost, codex-cli's usage): what the session had reported before is subtracted when the call
    continued it (`state`, from `session`); a new session starts from zero."""
    before = state.get("totals") or {} if state["id"] and state["id"] == session_id else {}
    return {k: max(0, v - before.get(k, 0)) for k, v in totals.items()}


def turn(messages: list[dict], state: dict) -> list[dict]:
    """The parts one call sends to the session in `state` (from `session`): the delta when it
    continues the session, the whole history (context.replay) when it starts one; CONTINUE when
    nothing follows the last reply."""
    if state["id"]:
        return context.delta(messages) or [context.text(CONTINUE)]
    items = context.replay(messages)
    if context.delta(messages):
        return items
    return context.merge(items + [context.text("\n\n" + CONTINUE)]) if items \
        else [context.text(CONTINUE)]


def lowerings(messages: list[dict], state: dict, own: tuple[str, ...] = ()) -> list[str]:
    """Lowerings of sending `messages` to the session in `state`, with `own` the transport's for
    this turn. A session keeps what it received, so its earlier lowerings still apply."""
    new = []
    if not state["id"] and any(m["role"] == "assistant" for m in messages):
        new.append("history_replayed")  # a session can hold only the replies it produced
    if not context.delta(messages):
        new.append("continue_prompt")
    out = list(state["lowerings"])
    return out + [name for name in (*new, *own) if name not in out]


def is_limit(message: str) -> bool:
    """Whether an error names a provider limit that lasts until it resets, or a login that cannot
    be used until someone signs in again."""
    low = message.lower()
    return any(s in low for s in LIMITS + ACCOUNT)


def wait_hint(message: str, now: datetime | None = None) -> float | None:
    """Seconds until the limit an error names resets ("resets 9:20pm", "resets Oct 10, 10pm
    (America/Chicago)"), else None. The time is read in the time zone the message names, else in
    local time; a reset that has passed is the next day's (or, with a date, next year's)."""
    m = RESETS.search(message)
    if not m:
        return None
    hour, minute = int(m.group("hour")), int(m.group("minute") or 0)
    half = (m.group("half") or "").lower().replace(".", "")
    if half == "pm" and hour != 12:
        hour += 12
    elif half == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    zone = None
    if m.group("tz"):
        try:
            zone = ZoneInfo(m.group("tz"))
        except (ZoneInfoNotFoundError, ValueError):
            zone = None
    now = now or datetime.now().astimezone()
    now = now.astimezone(zone) if zone else now
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if m.group("mon"):
        mon = m.group("mon")[:3].lower()
        if mon not in MONTHS:
            return None
        try:
            target = target.replace(month=MONTHS.index(mon) + 1, day=int(m.group("day")))
        except ValueError:
            return None
        if target <= now:
            target = target.replace(year=target.year + 1)
    elif target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def through_outages(model, attempt, label: str):
    """Run `attempt() -> (result, error)` until it succeeds. A provider limit (`is_limit`) is
    waited out: until the reset time the error names, plus a minute, or `model.limit_poll_s`, for
    at most `model.outage_wait_s` in all; other retryable errors get `model.retries` exponential
    backoffs. Returns (result, outage): outage is the (start, end) time.monotonic() interval from the
    first failure to the success, or None; the gateway leaves it out of the wall-time budget."""
    down, tries = None, 0
    while True:
        result, error = attempt()
        if error is None:
            if down is not None:
                model.note_available()
            return result, ((down, time.monotonic()) if down is not None else None)
        if not any(s in error.lower() for s in RETRYABLE) and not is_limit(error):
            raise ModelError(error)
        now = time.monotonic()
        down = now if down is None else down
        if is_limit(error):
            hint = wait_hint(error)
            pause = hint + 60 if hint is not None else float(model.limit_poll_s)
            if now - down + pause > float(model.outage_wait_s):
                raise ModelError(f"{error} (still limited after waiting {now - down:.0f}s; "
                                 f"outage_wait_s is {model.outage_wait_s})")
            model.note_unavailable(now + pause)
        elif tries < model.retries:
            pause = min(model.backoff_s * (2 ** tries), 1800)
            tries += 1
        else:
            raise ModelError(error)
        print(f"[{label}] {error[:200]}; retrying in {pause:.0f}s", file=sys.stderr, flush=True)
        time.sleep(pause)
