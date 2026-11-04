# Round {{round}} · task `{{task_id}}`

## Task instruction

{{instruction}}

## Where things are

- Solution workspace: `{{workspace}}` (bash starts here)
- Dev data, inputs and labels, readable: `{{dev_data}}`
- Your notes: `{{notes_file}}`
- Scratch directory for candidate variants (emptied every round, never committed): `{{scratch}}`
- Primary metric: `{{metric}}` ({{direction_text}}). Baseline val {{baseline_val}}; best accepted val {{best_val}} (round {{accepted_round}})
- GPUs per experiment: up to {{gpus_max}}

## Previous rounds

{{history_table}}

## Your notes so far

{{notes}}

## Lessons from earlier work (kept by the improver)

{{lessons}}

## Submissions

{{submissions}}

## Budget

{{budget_summary}}

Work on the task now. Call `finish` when this round's change is measured and the workspace is in its best state.
