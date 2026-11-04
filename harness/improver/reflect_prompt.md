# Improver review · round {{round}} · task `{{task_id}}`

## This round

- Executor: {{ex_status}} after {{ex_steps}} model calls
- Executor report: {{ex_summary}}
- Validation: {{val_status}}, score {{val_score}} ({{metric}}, {{direction_text}}); decision **{{decision}}** ({{decision_reason}})
- Best accepted val so far: {{best_val}} (baseline {{baseline_val}})

## Experiments this round

{{experiments_table}}

## Earlier rounds

{{history_table}}

## Harness files

{{harness_files}}

## Lessons file (`harness/improver/memory/lessons.md`)

{{lessons}}

## Budget

{{budget_summary}}

The executor transcript for this round is `{{transcript}}`. It can be long; read it in parts with `read_file` (offset/limit). Then improve the harness if the evidence calls for it, and end with `finish`.
