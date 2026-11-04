# Flex-RSI Bench

Flex-RSI evaluates whether an agent, given a task and a fixed budget, can improve not only its solution to the
task but also the harness through which it operates.

> 中文版说明见 [README.zh-CN.md](README.zh-CN.md).

## Problem setup

Each task provides the agent with a starter solution, compute resources, a budget, and data partitioned into three
disjoint splits. The **development split**, including its labels, is available to the agent for experimentation and
evaluation. The **validation split** is withheld from the agent and determines which candidate solutions are
retained. The **test split** is likewise withheld and is used only once, after the run terminates, to evaluate the
final solution. The objective of the agent is to maximize the test score of its final solution within the budget.

On a given task, every run starts from the same starter solution and is subject to the same budget, which limits
wall-clock time, GPU hours, model usage, the number of validation evaluations, and the number of rounds.

## The self-improving harness

A harness is the program that mediates between a language model and a task: through its prompts, tools, control
loop, and memory, it determines the context presented to the model and the actions available to it. The harness
$H=(E,I)$ comprises two components that query the same model:

- the **executor** $E$ (`harness/executor/`, `harness/core/`) operates on the task solution: it modifies the
  solution, conducts experiments, and evaluates the results on the development split;
- the **improver** $I$ (`harness/improver/`) operates on the harness: it analyses the executor's trajectories and
  results, diagnoses failures, and revises the prompts, tools, and workflow of the executor as well as its own
  instructions and accumulated lessons.

A run proceeds over rounds. In round $t$, the executor modifies the accepted solution $s_{t-1}$ using only the
development split and submits a candidate $\tilde{s}_t$. The candidate is evaluated on the validation split and
retained only if it improves on the accepted solution by more than a noise margin $\delta$, which the starting
harness sets to twice the standard deviation of the starter's validation score and which the improver may revise.
The improver then examines the records of all rounds so far and produces the harness for the next round. The run
terminates when the budget is exhausted or progress stops, and its result is the final solution $s_T$.

All runs start from the same general-purpose harness, which contains no task- or model-specific components. The
evaluator, the data splits, and the accounting of the budget are fixed by the orchestrator (`src/rsibench/`) and
cannot be modified by the agent. Protocol details: [docs/configuration.md](docs/configuration.md),
[docs/harness.md](docs/harness.md).

## Evaluation

A run reports the test score of its final solution, $S(s_T;\mathcal{D}_{\mathrm{test}})$, and its improvement over the
starter solution, $S(s_T;\mathcal{D}_{\mathrm{test}})-S(s_0;\mathcal{D}_{\mathrm{test}})$. Agent code is executed in a
sandbox without access to the validation or test data.

## Tasks

| Area | Task | `--task` | Source | Metric | Test split |
|---|---|---|---|---|---|
| Action | Active Search | `hstar-search` | H*Bench | SPL | 47 episodes |
| | LIBERO-MAX | `libero-max-agent` | LIBERO-MAX | Dynamic success rate | 24 episode pairs |
| | LIBERO-PRO | `libero-pro-agent` | LIBERO-PRO | Success rate | 30 episodes |
| | Embodied-SWE | `embodied-swe` | EmbodiedSWE-Bench | Rubric score | 4 seeds × 8 environments |
| Reasoning | Ego-Exo Memory | `egoexomem` | EgoExoMem (Ego-Exo4D) | Accuracy | 120 questions |
| | Kinematics | `quantiphy-kinematics` | QuantiPhy | MRA | 67 questions |
| | Mass | `quantiphy-mass` | Amazon Berkeley Objects | MRA | 600 items |
| | Spatial-VSI | `spatial-vsi` | VSI-Bench | VSI-Bench score | 320 questions |
| Creation | Image Editing | `edit-image-unlimited` | Ours | VLM judge + checks | 80 images |
| | Multi-turn Editing | `edit-multiturn` | MSE-Bench | VLM judge | 40 sessions |
| Reconstruction | 3D Tracking | `iggt-tracking` | IGGT (ScanNet, ScanNet++) | T-mIoU | 10 clips |
| | Scene from Photo | `scene3d-harmony` | HARMONY300 | Composite score | 60 photos |
| | Scene from Video | `scene3d-holoscene` | HoloScene | Composite score | 6 scenes |
| Perception | Visual Search | `vsearch` | V*Bench, HR-Bench 8K | Accuracy | 140 questions |
| | Video Quality | `video-quality` | YouTube-UGC, Veo 3.1 pairs | (SRCC + pairwise acc.) / 2 | 40 videos, 20 pairs |

Task cards (purpose, data and splits, setting, starter solution, metric), data preparation and the standard run of
each task: [docs/tasks.md](docs/tasks.md).

## Data

The prepared splits of 3D Tracking, Active Search, Visual Search, Video Quality and Image Editing are in the Hugging
Face dataset [Flex-RSI/RSI-Dataset](https://huggingface.co/datasets/Flex-RSI/RSI-Dataset), in the layout the
benchmark reads (`<task>/{dev,val,test}`):

```bash
export RSIB_DATA_ROOT=/path/to/rsib-data
huggingface-cli download Flex-RSI/RSI-Dataset --repo-type dataset --include '<task>/*' --local-dir "$RSIB_DATA_ROOT"
```

The other tasks are prepared from their sources with `rsib task prepare <task>` ([docs/tasks.md](docs/tasks.md)).

## Install

Python 3.11+ and git 2.30+.

```bash
git clone https://github.com/flex-rsi/rsi-bench.git
cd rsi-bench
python -m venv .venv && . .venv/bin/activate
pip install -e '.[all,tasks]'
pytest -q                          # offline checks with a scripted mock model
```

## Running a task

1. **Data.** Download or prepare the task's splits under `$RSIB_DATA_ROOT/<task>` (above).
2. **Machine.** A site profile names the compute backend (local GPUs or SLURM), the containers and the GPUs jobs
   may use; start from `examples/site.example.toml` ([operations](docs/operations.md#site-profile)).
3. **Model.** A local Claude Code or Codex login (`claude-cli:…`, `codex-cli:…`), a provider API key in `.env`
   (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`), or a self-hosted OpenAI-compatible server (`openai:MODEL@URL`)
   ([models](docs/models.md)).
4. **Run** with the task's standard flags ([docs/tasks.md](docs/tasks.md)):

```bash
rsib run new --task <task> --model mock:finish --mode base --max-rounds 1 --budget wall_hours=0.5
rsib run start <worktree>                      # smoke test: the package, data, backend and sandbox work
rsib run new --task <task> --model <model> --effort <effort> --mode full --seed 1 <the task's flags>
rsib run start <worktree> --backend slurm      # or --backend local
rsib report <worktree>                         # test score of the final solution, its gain over the starter
```

`rsib run new` creates a run branch and a worktree; `rsib run start` runs the rounds until the run terminates,
then scores the final solution, and every accepted solution, on the test split. Every model call, experiment and
evaluation is recorded in the run branch ([records](docs/records.md)). Step by step:
[testing guide](docs/testing-guide.md).

## Results

`rsib report <worktree>` summarizes one run; `rsib results table --branches 'run/*'` averages runs per task and
model ([results](docs/results.md)).

## Contributing to Flex-RSI

The channel for contributing to Flex-RSI, by submitting runs to be verified and added to the leaderboard on the
[project page](https://flex-rsi.github.io), will open as soon as possible. Until then, keep your run branches as
they are: they hold everything a submission will need.

## Documentation

| Document | What it covers |
|---|---|
| [docs/tasks.md](docs/tasks.md) | task cards, data preparation, standard runs |
| [docs/testing-guide.md](docs/testing-guide.md) | from a fresh clone to a scored run |
| [docs/models.md](docs/models.md) | connecting Claude, GPT, other APIs and self-hosted models |
| [docs/configuration.md](docs/configuration.md) | flags, stop conditions, protocol constants, budgets |
| [docs/harness.md](docs/harness.md) | the starting harness: executor, improver, phases, tools |
| [docs/operations.md](docs/operations.md) | environment variables, site profile, backends, sandbox, resuming |
| [docs/records.md](docs/records.md) | run branches, record files, integrity checks |
| [docs/results.md](docs/results.md) | what counts as a result, the results table and report |
| [docs/task-authoring.md](docs/task-authoring.md) | writing a new task |
| [CONTRIBUTING.md](CONTRIBUTING.md) | development setup, tasks, model adapters, metrics |

## License

To be added.
