#!/usr/bin/env bash
# Run on the GPU host that runs the LIBERO evaluations. Every environment, source, dataset and weight goes under PROJECT_ROOT.
set -euo pipefail

project_root="${1:?usage: setup_runtime.sh PROJECT_ROOT [FLEX_CHECKOUT]}"
flex_checkout="${2:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p "$project_root"
project_root="$(cd "$project_root" && pwd)"
flex_checkout="$(cd "$flex_checkout" && pwd)"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
export HF_HOME="$project_root/cache/huggingface"
export UV_CACHE_DIR="$project_root/cache/uv"
mkdir -p "$project_root/vendor" "$project_root/models" "$project_root/config" "$HF_HOME"

checkout() {
  local name="$1" url="$2" revision="$3"
  if [[ ! -d "$project_root/vendor/$name/.git" ]]; then
    GIT_CONFIG_GLOBAL=/dev/null git clone --filter=blob:none "$url" "$project_root/vendor/$name"
  fi
  GIT_CONFIG_GLOBAL=/dev/null git -C "$project_root/vendor/$name" checkout --detach "$revision"
}

checkout LIBERO-MAX https://github.com/liberomax/LIBERO-MAX.git a1e3cef258b3e00487db5282aef4e36010aaf401
checkout LIBERO-PRO https://github.com/Zxy-MLlab/LIBERO-PRO.git 2b910b5b5f53016bef9907632f6f840f1ce2229c
checkout LIBERO-plus https://github.com/sylvestf/LIBERO-plus.git 4976dc30028e805ff8094b55501d532c48fec182
checkout lerobot https://github.com/huggingface/lerobot.git b9cb121cb7d4e3e26ec5c906d08dda68d151ba52

if [[ ! -x "$project_root/robot-env/bin/python" ]]; then
  "$uv_bin" venv --python 3.12 "$project_root/robot-env"
fi
robot_python="$project_root/robot-env/bin/python"
"$robot_python" "$flex_checkout/scripts/libero/repair_plus_fog.py" \
  --plus-root "$project_root/vendor/LIBERO-plus"
"$uv_bin" pip install --python "$robot_python" \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0
"$uv_bin" pip install --python "$robot_python" \
  -e "$project_root/vendor/lerobot[xvla]" -e "$project_root/vendor/LIBERO-MAX" \
  torch==2.11.0 torchvision==0.26.0 mujoco==3.2.6 robosuite==1.4.0 \
  bddl==1.0.1 gym==0.25.2 numba==0.61.2 easydict future cloudpickle \
  matplotlib imageio pyyaml Wand scikit-image

PATH="$project_root/robot-env/bin:$PATH" "$robot_python" \
  "$project_root/vendor/LIBERO-MAX/scripts/setup_libero_pro_substrate.py" \
  --libero-pro-root "$project_root/vendor/LIBERO-PRO" \
  --dataset-root "$project_root/data/libero-pro" \
  --config-dir "$project_root/config/libero-pro" \
  --revision c86fc3b8293185a6f373677018ff3e37f8391602 \
  --runtime-revision 2b910b5b5f53016bef9907632f6f840f1ce2229c

"$robot_python" - "$project_root" <<'PY'
import json
import sys
import zipfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

root = Path(sys.argv[1])
source = root / "vendor/LIBERO-plus/libero/libero"
config = root / "config/libero-plus"
config.mkdir(parents=True, exist_ok=True)
(config / "config.yaml").write_text(json.dumps({
    "benchmark_root": str(source),
    "bddl_files": str(source / "bddl_files"),
    "init_states": str(source / "init_files"),
    "assets": str(source / "assets"),
    "datasets": str(root / "data/libero-plus"),
}, indent=2) + "\n")
api = HfApi()
versions = {
    "LIBERO-MAX": "a1e3cef258b3e00487db5282aef4e36010aaf401",
    "LIBERO-PRO": "2b910b5b5f53016bef9907632f6f840f1ce2229c",
    "LIBERO-plus": "4976dc30028e805ff8094b55501d532c48fec182",
    "lerobot": "b9cb121cb7d4e3e26ec5c906d08dda68d151ba52",
    "zhouxueyang/LIBERO-Pro": "c86fc3b8293185a6f373677018ff3e37f8391602",
    "Sylvest/LIBERO-plus": "dd2bd61b7d9a6fef1abc52d606e983b41886a149",
    "lerobot/xvla-libero": "12e8783e996944f5c97e490d37d4c145484ed70a",
}
versions_path = root / "runtime-versions.json"
if versions_path.exists():
    previous = json.loads(versions_path.read_text())
    versions["Qwen/Qwen3.8-27B"] = previous["Qwen/Qwen3.8-27B"]
    versions["facebook/bart-large"] = previous["facebook/bart-large"]
else:
    versions["Qwen/Qwen3.8-27B"] = api.model_info("Qwen/Qwen3.8-27B").sha
    versions["facebook/bart-large"] = api.model_info("facebook/bart-large").sha
versions_path.write_text(json.dumps(versions, indent=2) + "\n")
plus_data = root / "data/libero-plus"
asset_archive = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset",
                               revision=versions["Sylvest/LIBERO-plus"], local_dir=plus_data)
extracted = plus_data / "assets-extracted"
if not extracted.exists():
    with zipfile.ZipFile(asset_archive) as assets:
        # This pinned official archive includes the author's directory prefix.
        for entry in assets.infolist():
            entry.filename = entry.filename[entry.filename.index("assets/"):]
            assets.extract(entry, source)
    extracted.write_text(versions["Sylvest/LIBERO-plus"] + "\n")
snapshot_download("lerobot/xvla-libero", revision=versions["lerobot/xvla-libero"],
                  local_dir=root / "models/xvla-libero")
snapshot_download("facebook/bart-large", revision=versions["facebook/bart-large"],
                  allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt",
                                  "special_tokens_map.json", "added_tokens.json"])
snapshot_download("Qwen/Qwen3.8-27B", revision=versions["Qwen/Qwen3.8-27B"],
                  local_dir=root / "models/Qwen3.8-27B")
runtime = {
    "max_root": str(root / "vendor/LIBERO-MAX"),
    "pro_root": str(root / "vendor/LIBERO-PRO"),
    "plus_root": str(root / "vendor/LIBERO-plus"),
    "pro_config": str(root / "config/libero-pro"),
    "plus_config": str(root / "config/libero-plus"),
    "checkpoint": str(root / "models/xvla-libero"),
    "lerobot_root": str(root / "vendor/lerobot"),
    "python": str(root / "robot-env/bin/python"),
}
(root / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
PY

if [[ ! -x "$project_root/agent-env/bin/python" ]]; then
  "$uv_bin" venv --python 3.12 "$project_root/agent-env"
fi
"$uv_bin" pip install --python "$project_root/agent-env/bin/python" vllm==0.30.0 'transformers>=5.8.0'
if [[ ! -x "$project_root/harness-env/bin/python" ]]; then
  "$uv_bin" venv --python 3.12 "$project_root/harness-env"
fi
"$uv_bin" pip install --python "$project_root/harness-env/bin/python" -e "$flex_checkout[all,tasks]"
for runtime_name in robot agent harness; do
  "$uv_bin" pip freeze --python "$project_root/$runtime_name-env/bin/python" \
    > "$project_root/$runtime_name-packages.txt"
done

cat > "$project_root/site.toml" <<'TOML'
[local]
gpus = "0"
max_parallel = 1

[phase]
cpus = 8
threads = 8

[gateway]
host = "127.0.0.1"

[sandbox]
enabled = true
archive_days = 7
TOML

"$project_root/harness-env/bin/python" - "$project_root" "$flex_checkout" <<'PY'
import shlex
import sys
from pathlib import Path

root, repo = map(Path, sys.argv[1:])
site = root / "site.toml"
robot_env = root / "robot-env"
base_python = (robot_env / "bin/python").resolve().parents[1]
site.write_text(site.read_text().replace(
    "enabled = true", "enabled = true\nread_only = " +
    __import__("json").dumps([str(robot_env), str(base_python)])))
values = {
    "RSIB_REPO": str(repo),
    "RSIB_RUNS_ROOT": str(root / "runs"),
    "RSIB_DATA_ROOT": str(root / "data/flex"),
    "RSIB_ARTIFACT_ROOT": str(root / "artifacts"),
    "RSIB_SITE": str(root / "site.toml"),
    "LIBERO_RSI_RUNTIME": str(root / "runtime.json"),
    "HF_HOME": str(root / "cache/huggingface"),
    "RSIB_PASS_ENV": "MUJOCO_GL,PYOPENGL_PLATFORM,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    "OPENAI_API_KEY": "EMPTY",
}
env = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in values.items())
(root / "env.sh").write_text(env + "\n")
PY
printf 'Runtime ready: %s\n' "$project_root/runtime.json"
printf 'Load project environment: source %s/env.sh\n' "$project_root"
