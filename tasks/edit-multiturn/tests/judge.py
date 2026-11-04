"""Frozen VLM judge (GLM-4.5V through vLLM). Part of the task version: do not change between runs.

Input: a JSON file with a list of requests {key, images: [paths], prompt, group}.
Output: a JSON file mapping key -> raw text answer.
Settings: bf16, tensor parallel over the visible GPUs, temperature 0, thinking off, images at most
1536 px on the long side, at most 6 images per request (the source and five turns).

The requests of one group (the evaluator's group is a session) form one batch of their own, and
groups run one after another, without prefix caching: vLLM's answers depend on what else is in
a batch (VLLM_BATCH_INVARIANT does not remove that for this model on this stack), so this way a
session's scores depend on its own images only, never on the other sessions'.
"""

import base64
import io
import json
import os
import sys

from PIL import Image

MODEL = "zai-org/GLM-4.5V"
MAX_SIDE = 1536
MAX_IMAGES = 6


def data_uri(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main(req_path, out_path):
    import torch
    from vllm import LLM, SamplingParams

    reqs = json.loads(open(req_path).read())
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    tp = len(visible) if visible else torch.cuda.device_count()
    # No FlashInfer autotune: it picks kernels by timing (so answers could change between runs),
    # and with a cached result its rank-synchronized tuning hung on 4 PCIe GPUs.
    # 0.80: still ~23 GiB of KV cache per 96 GB GPU, and it starts while other processes hold up
    # to ~18 GiB of a GPU (shared machines).
    llm = LLM(MODEL, tensor_parallel_size=tp, max_model_len=32768,
              limit_mm_per_prompt={"image": MAX_IMAGES}, gpu_memory_utilization=0.80,
              enforce_eager=True, enable_flashinfer_autotune=False, enable_prefix_caching=False,
              seed=0)
    groups = {}
    for r in reqs:
        groups.setdefault(r.get("group", ""), []).append(r)
    res = {}
    for name in sorted(groups):
        batch, msgs = groups[name], []
        for r in batch:
            content = [{"type": "text", "text": r["prompt"]}]
            content += [{"type": "image_url", "image_url": {"url": data_uri(p)}} for p in r["images"]]
            msgs.append([{"role": "user", "content": content}])
        outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1024), use_tqdm=False,
                        chat_template_kwargs={"enable_thinking": False})
        res.update({r["key"]: o.outputs[0].text for r, o in zip(batch, outs)})
    with open(out_path, "w") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
