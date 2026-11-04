"""Frozen VLM judge (GLM-4.5V through vLLM). Part of the task version: do not change between runs.

Input: a JSON file with a list of requests {key, kind, images: [paths], prompt}.
Output: a JSON file mapping key -> raw text answer.
"""

import base64
import io
import json
import os
import sys

from PIL import Image

MODEL = "zai-org/GLM-4.5V"
MAX_SIDE = 1536  # fine texture must be visible for PR/PQ


def data_uri(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main(req_path, out_path):
    from vllm import LLM, SamplingParams

    reqs = json.loads(open(req_path).read())
    tp = len([g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g.strip()])
    llm = LLM(MODEL, tensor_parallel_size=tp, max_model_len=16384, limit_mm_per_prompt={"image": 2},
              gpu_memory_utilization=0.85, enforce_eager=True, seed=0)
    msgs = []
    for r in reqs:
        content = [{"type": "text", "text": r["prompt"]}]
        content += [{"type": "image_url", "image_url": {"url": data_uri(p)}} for p in r["images"]]
        msgs.append([{"role": "user", "content": content}])
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=512),
                    chat_template_kwargs={"enable_thinking": False})
    res = {r["key"]: o.outputs[0].text for r, o in zip(reqs, outs)}
    with open(out_path, "w") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
