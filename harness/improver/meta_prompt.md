# Meta review · round {{round}} · trigger: {{trigger}}

This time you review your own improvement process, not the executor's work.

## Harness changes and what followed

{{harness_history}}

## Rounds so far on `{{task_id}}`

{{history_table}}

## Your current setup

`harness/improver/config.toml`, `harness/improver/accept.py`, the prompts `harness/improver/*.md` and the memory `harness/improver/memory/lessons.md`.

Answer from the evidence:

- Which kinds of harness edits were followed by accepted improvements, and which by stalls or regressions?
- Is the acceptance rule too strict (good candidates rejected) or too loose (noise accepted)?
- Do your review prompts lead to useful diagnoses, or to edits that do not pay off?
- What should later tasks inherit? Record it in `memory/lessons.md` as short, task-general lessons.

{{improver_policy}} Change only what the evidence supports, then call `finish` with what you changed and why.
