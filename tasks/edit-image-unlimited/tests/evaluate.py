"""Evaluator for edit-image.

1. copy the split's inputs (image, instruction, lang; no labels) to a temp dir;
2. run the solution there in a child process (sandboxed when $RSIB_SANDBOX is set), one worker
   per GPU, which writes outputs/<id>.png;
3. score here: frozen VLM judge (SC / PQ / PR and VQA checks), OCR checks, mask-based
   aligned preservation (geometry.py), with the measured global deformation deducted from the
   judge's PR; write reward.json.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import fidelity  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = os.environ.get("EDIT_IMAGE_MODELS", "/models/edit-image/hf_models")
# pre-staged editing models (read-only HF cache, see instruction.md)
STAGED_MODELS = os.environ.get("EDIT_IMAGE_UNLIMITED_MODELS", "/models/edit-image/hf_open")
STARTUP_S = 600  # loading larger models than the original backbone
JUDGE_HOME = os.environ.get("EDIT_IMAGE_JUDGE_HOME", "/models/edit-image/hf_judge")
SEC_PER_IMAGE = 150  # per GPU, on average
GEOM_PR_PENALTY = 5.0  # PR points lost at geometry score 0
SC_GATE = 3
# Verification protocol of each sub-category: which content the judge's PR is about, and which
# pixel measurements apply. Pixel checks compare the aligned output with the source outside the
# edit region (mask file united with objects grounded by the judge), so they only make sense where
# the rest of the frame must stay pixel-identical.
PROTOCOL = {
    "A1": "local", "A2": "local", "A3": "local", "A4": "local", "A6": "local",
    "C1": "local", "C2": "local", "C3": "local", "E3": "local", "G2": "local",
    "A5": "motion", "A7": "motion",
    "B1": "style", "B2": "background", "B3": "environment",
    "E1": "tone", "E2": "restore", "E4": "reframe",
    "F1": "reasoning", "F2": "reasoning", "F3": "reasoning", "F4": "reasoning", "G1": "compound",
}
#   geometry: homography deformation g, deducted from PR (not for style transfer, where matches
#             between a photo and a painting are meaningless, nor for unmasked moves/poses);
#   preserve/detail: colour (MAD) and fine-texture (SSIM cs) fidelity outside the edit region -
#             samples with a mask, unmasked local edits whose region the judge grounds, and the
#             source frame inside an outpainted canvas; detail only for tone edits (the tone is
#             meant to change, the texture is not). Background, environment, style, restoration,
#             reasoning and compound edits re-render or may touch any part of the frame, so their
#             content preservation is left to the judge's PR with a protocol-specific rubric.
PR_GUIDE = {
    "local": "Only the requested object or region may change. Everything else must stay identical to image 1 down to fine detail: texture and grain (wood, fabric, skin, foliage, hair), sharpness, small objects, text, faces, colours, lighting, framing and position.",
    "motion": "Only the pose or position of the named subject may change. The subject's identity and appearance and everything else (background, other objects, fine texture, colours, framing) must stay identical to image 1.",
    "style": "The rendering style, brushwork, texture and colours are SUPPOSED to change; do not penalise the new style. Judge only whether the content is kept: the same objects with the same shapes, positions and poses, the same layout, composition and framing, nothing added or removed.",
    "background": "The background is SUPPOSED to change. Judge whether the subjects to keep are identical to image 1 (shape, fine detail, texture, colour, identity) at the same size and position in the frame.",
    "environment": "Lighting, weather, season or time of day are SUPPOSED to change the look of the whole scene. Judge whether the scene content is kept: the same buildings, objects, people, layout, composition and framing, with nothing added, removed or redrawn beyond what the change requires.",
    "tone": "Brightness, contrast and colour are SUPPOSED to change. Judge whether the content and fine detail are kept: the same objects, textures, edges and framing, nothing smoothed, redrawn or invented.",
    "restore": "Damage, blur, noise, missing colour, haze or fog, lens flare and glare, reflections on glass, cast shadows or smoke are SUPPOSED to be repaired or removed. Judge whether the content is kept: the same people, faces, objects, layout and framing, nothing invented or redrawn beyond the repair.",
    "reframe": "New content is SUPPOSED to appear outside the original frame. Judge whether the original picture is kept identical inside the new canvas (content, fine detail, scale, colours).",
    "reasoning": "Only what the intended result requires may change. Everything else must stay identical to image 1, including fine detail, colours and framing.",
    "compound": "Only what the listed parts require may change. Everything else must stay identical to image 1, including fine detail, colours and framing.",
}
GROUND_PROMPT = ("Locate {phrase} in the image. For every instance give one bounding box that covers it together "
                 "with everything it brings into the picture: its cast shadow, reflection, steam, smoke, splashes or glow. "
                 "Answer with a list [[x1,y1,x2,y2], ...] in coordinates normalized to 0-1000. Output only the list, or [] if there is none.")

SCPR_PROMPT = """You are a strict, careful evaluator of instruction-based image editing.
Image 1 is the ORIGINAL image. Image 2 is the EDITED image produced by an editing system.

Editing instruction: {instruction}
{extra}
Rate two criteria on an integer scale from 0 to 10. Be demanding: high scores are for professional-grade edits only.
- SC (instruction compliance): how completely and correctly image 2 carries out the instruction relative to image 1. 0 = the requested change is absent or wrong (also when image 2 is essentially identical to image 1); 5 = partially done or with clear mistakes; 10 = fully and precisely done. For multi-part instructions, SC reflects how many parts are done correctly.
- PR (preservation): how faithfully everything that should be kept is kept. {pr_guide} Smoothed, blurred or redrawn detail that should be kept counts as a change even when the picture looks the same at a glance.
  10 = everything that should be kept is indistinguishable from image 1, including fine detail;
  8 = only slight differences visible on close inspection;
  6 = clearly noticeable loss or redrawing of detail that should be kept, or a slight colour, tone or framing change;
  4 = many parts that should be kept are visibly altered or redrawn;
  2 = most of it is redrawn or replaced; 0 = unrelated to image 1.
Answer with JSON only: {{"reason": "<two sentences: first the instruction, then preservation>", "SC": <int>, "PR": <int>}}"""

PQ_PROMPT = """You are a strict judge of the visual quality of an AI-edited image. Ignore whether it matches any instruction.
Inspect the whole image at full resolution, including fine detail. Penalise: blur or softness, upscaling artifacts, over-smoothed, waxy or plastic textures, smeared or repetitive fake texture, warped or malformed objects, hands, faces or text, unnatural blending, seams or halos, inconsistent lighting, shadows or perspective, noise, banding and compression artifacts.
Rate PQ (perceptual quality) on an integer scale from 0 to 10:
10 = indistinguishable from a sharp, high-quality real photograph (or a well-made artwork in its medium), no artifacts even on close inspection;
8 = good, with minor flaws visible only on close inspection;
6 = noticeable softness, smoothed texture or small artifacts;
4 = obvious artifacts, unnatural blending, or clearly degraded detail;
2 = severe distortions or broken content; 0 = unrecognizable.
For paintings, prints, pixel art or old photographs, judge consistency with the medium rather than photographic sharpness.{medium}
Answer with JSON only: {{"reason": "<one or two sentences>", "PQ": <int>}}"""

YESNO_PROMPT = "Look at the image and answer with a single word, yes or no.\nQuestion: Is there {obj} in the image?"
COUNT_PROMPT = "Look at the image and count carefully. How many {obj} are there in the image? Answer with a single integer."
READ_PROMPT = ("Transcribe all text visible in the image exactly as written, character by character, in its original "
               "language, one text block per line. Do not correct, translate or guess unreadable text. "
               "Output only the transcription, or NONE if there is no text.")


def solution_cmd(python, io_dir):
    child = Path(io_dir) / "_child_run.py"
    shutil.copy(HERE / "child_run.py", child)
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    return prefix + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + cmd if prefix else cmd


def gpus():
    return [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]


def run_cache():
    """The run's download cache ($RSIB_CACHE of the phases): the read-only mount named `cache`
    in the sandbox prefix, next to the run's `home` directory. None without a sandbox."""
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    for flag, path in zip(prefix, prefix[1:]):
        p = Path(path)
        if flag == "--ro" and p.name == "cache" and (p.parent / "home").is_dir():
            return p
    return Path(os.environ["RSIB_CACHE"]) if os.environ.get("RSIB_CACHE") else None


def cache_env():
    """The same cache variables the phases and experiments see, so weights and packages the
    agent installed into $RSIB_CACHE load the same way at evaluation."""
    c = run_cache()
    if c is None:
        return {}
    return {"RSIB_CACHE": str(c), "HF_HOME": str(c / "huggingface"), "TORCH_HOME": str(c / "torch"),
            "XDG_CACHE_HOME": str(c / "xdg"), "PIP_CACHE_DIR": str(c / "pip")}


def base_env(tmp):
    env = {k: os.environ[k] for k in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "LANG", "TMPDIR",
                                        "RSIB_CACHE", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME") if k in os.environ}
    env.setdefault("HOME", tmp)
    env["HOME"] = os.environ.get("HOME", tmp)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ---------------------------------------------------------------- parsing helpers
def parse_scores(text, keys):
    text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    out = {}
    for k in keys:
        m = re.search(r'"?%s"?\s*[:=]\s*"?(\d+(?:\.\d+)?)' % k, text)
        out[k] = min(10.0, max(0.0, float(m.group(1)))) if m else None
    m = re.search(r'"reason"\s*:\s*"(.*?)"\s*[,}]', text, re.S)
    out["reason"] = m.group(1) if m else text[:300]
    return out


def parse_yes(text):
    t = text.replace("<|begin_of_box|>", "").strip().lower()
    return t.startswith("yes") or t.startswith("是")


def parse_int(text):
    m = re.search(r"\d+", text.replace("<|begin_of_box|>", ""))
    return int(m.group()) if m else None


def parse_boxes(text):
    return [[float(v) for v in m] for m in
            re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", text)]


# ---------------------------------------------------------------- verification protocol
def tools(s):
    """Pixel measurements that apply to a sample: geometry, preserve, detail, grounding."""
    proto, masked = PROTOCOL[s["category_l2"]], bool(s.get("edit_mask"))
    geometry = proto != "style" and (masked or proto != "motion")
    region = masked or proto == "local"  # an edit region exists or can be grounded
    preserve = region or proto == "reframe"
    return {"geometry": geometry, "preserve": preserve, "detail": preserve or proto == "tone",
            # the dataset masks already cover the edit, grounding only replaces a missing mask
            "ground": region and not masked}


def ground_phrases(s):
    """Objects to locate so that edited content is excluded from the pixel comparison: what appears
    (in the output) and what disappears or is edited (in the source)."""
    if not tools(s)["ground"]:
        return {}
    c = s.get("checks") or {}
    out = list(c.get("object_present") or []) + [f'the text "{t}"' for t in c.get("ocr_expect") or []]
    src = list(c.get("object_absent") or []) + [f'the text "{t}"' for t in c.get("ocr_forbid") or []]
    if c.get("count"):
        out.append(c["count"]["object"])
        src.append(c["count"]["object"])
    if s.get("edit_target"):  # the edited object, wherever it ends up (moved, posed, resized)
        out.append(s["edit_target"])
        src.append(s["edit_target"])
    return {k: v for k, v in (("out", out), ("src", src)) if v}


# ---------------------------------------------------------------- programmatic checks
def norm_text(s):
    return re.sub(r"[\s\W_]+", "", s.lower())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def best_cer(target, text):
    """Lowest character error rate of `target` against any substring of `text`."""
    t, s = norm_text(target), norm_text(text)
    if not t:
        return 0.0
    if not s:
        return 1.0
    best = 1.0
    for w in range(max(1, len(t) - 2), len(t) + 3):
        for i in range(0, max(1, len(s) - w + 1)):
            best = min(best, lev(t, s[i:i + w]) / len(t))
    return best


_ocr = None


def ocr_text(path):
    """EasyOCR readings of the image: the Chinese+English model and the English-only model (the
    former often reads Latin capitals as Chinese characters)."""
    global _ocr
    if _ocr is None:
        import easyocr
        import torch
        # the readers are small; skipping cuDNN avoids failures when the host preloads another cuDNN
        torch.backends.cudnn.enabled = False
        kw = dict(gpu=bool(gpus()), verbose=False, download_enabled=False,
                  model_storage_directory=str(Path(JUDGE_HOME) / "easyocr"))
        _ocr = [easyocr.Reader(["ch_sim", "en"], **kw), easyocr.Reader(["en"], **kw)]
    im = Image.open(path).convert("RGB")
    im.thumbnail((1600, 1600))
    return [" ".join(r[1] for r in reader.readtext(np.array(im))) for reader in _ocr]


def conform(im, src_path, reframe):
    """Resolution rule: outputs are scored at the input's resolution. Non-outpainting outputs are
    resized to the input's exact size; outpainting outputs keep their own aspect ratio and are
    resized to the input's pixel count."""
    sw, sh = Image.open(src_path).size
    w, h = im.size
    if reframe:
        s = math.sqrt(sw * sh / (w * h))
        size = (max(1, round(w * s)), max(1, round(h * s)))
    else:
        size = (sw, sh)
    return im if im.size == size else im.resize(size, Image.LANCZOS)


def comparison(src_path, out_path, dst):
    h = 384
    ims = []
    for p in (src_path, out_path):
        im = Image.open(p).convert("RGB")
        ims.append(im.resize((max(1, round(im.width * h / im.height)), h)))
    sheet = Image.new("RGB", (sum(i.width for i in ims) + 8, h), "white")
    x = 0
    for im in ims:
        sheet.paste(im, (x, 0))
        x += im.width + 8
    sheet.save(dst, quality=85)


# ---------------------------------------------------------------- main
def main():
    split = os.environ.get("RSIB_SPLIT", "dev")
    split_dir = Path(os.environ["RSIB_DATA_DIR"])
    workspace = Path(os.environ["RSIB_WORKSPACE"])
    out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
    python = os.environ.get("RSIB_PYTHON", sys.executable)
    samples = [json.loads(l) for l in (split_dir / "samples.jsonl").read_text().splitlines() if l.strip()]
    limit = int(os.environ.get("EDIT_IMAGE_LIMIT", "0"))  # for smoke tests only
    if limit:
        samples = samples[:limit]
    n_gpu = max(1, len(gpus()))
    # maintainer use: score saved outputs (<id>.png) without running the solution
    rescore = os.environ.get("EDIT_IMAGE_OUTPUTS")

    with tempfile.TemporaryDirectory(prefix="editimg-") as tmp:
        if rescore:
            res_dir = out_dir / "outputs"
            res_dir.mkdir(parents=True, exist_ok=True)
            for s in samples:
                if (Path(rescore) / f"{s['id']}.png").exists() and Path(rescore).resolve() != res_dir.resolve():
                    conform(Image.open(Path(rescore) / f"{s['id']}.png").convert("RGB"), split_dir / s["image"],
                            bool((s.get("checks") or {}).get("aspect"))).save(res_dir / f"{s['id']}.png")
            log, errors, worker_errs = f"rescored outputs from {rescore}", {}, []
        else:
            io = Path(tmp)
            (io / "images").mkdir()
            with open(io / "inputs.jsonl", "w") as f:
                for s in samples:
                    shutil.copy(split_dir / s["image"], io / "images" / Path(s["image"]).name)
                    f.write(json.dumps({"id": s["id"], "image": "images/" + Path(s["image"]).name,
                                        "instruction": s["instruction"], "lang": s["lang"]}, ensure_ascii=False) + "\n")
            env = base_env(tmp)
            env.update(cache_env())
            env.update({"RSIB_IO_DIR": tmp, "RSIB_SOLVER_DIR": str(workspace), "EDIT_IMAGE_MODELS": MODELS,
                        "EDIT_IMAGE_UNLIMITED_MODELS": STAGED_MODELS,
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
            timeout = STARTUP_S + SEC_PER_IMAGE * math.ceil(len(samples) / n_gpu)
            try:
                proc = subprocess.run(solution_cmd(python, io), env=env, cwd=tmp, capture_output=True,
                                      text=True, timeout=timeout)
                log = proc.stdout[-6000:] + "\n" + proc.stderr[-6000:]
            except subprocess.TimeoutExpired as e:
                log = f"solution timed out after {timeout}s\n" + str(e.stdout or "")[-3000:]
            produced = io / "outputs"
            res_dir = out_dir / "outputs"
            res_dir.mkdir(parents=True, exist_ok=True)
            errors = {}
            for s in samples:
                p = produced / f"{s['id']}.png"
                if p.exists():
                    try:
                        conform(Image.open(p).convert("RGB"), split_dir / s["image"],
                                bool((s.get("checks") or {}).get("aspect"))).save(res_dir / f"{s['id']}.png")
                    except Exception as e:  # unreadable output counts as missing
                        errors[s["id"]] = f"unreadable output: {e}"
                if (produced / f"{s['id']}.err").exists():
                    errors[s["id"]] = (produced / f"{s['id']}.err").read_text()[-1500:]
            worker_errs = [p.read_text()[-3000:] for p in produced.glob("_worker*.err")] if produced.exists() else []
    (out_dir / "solution.log").write_text(log + "\n".join(worker_errs))

    done = [s for s in samples if (res_dir / f"{s['id']}.png").exists()]
    if not done:
        print(log[-3000:], *worker_errs, sep="\n")
        sys.exit("the solution produced no output")

    # ---- judge requests
    reqs = []
    for s in done:
        c = s.get("checks") or {}
        src, outp = str(split_dir / s["image"]), str(res_dir / f"{s['id']}.png")
        extra = []
        if s.get("instruction_alt") and s["lang"] != "en":
            extra.append(f"(English version of the instruction: {s['instruction_alt']})")
        if c.get("reasoning_target"):
            extra.append(f"The intended result, made explicit for you (the editing system did not see this): {c['reasoning_target']}")
        if c.get("sub_edits"):
            extra.append("The instruction consists of these parts: " + "; ".join(c["sub_edits"]))
        if c.get("keep"):
            extra.append("These must stay unchanged: " + "; ".join(c["keep"]))
        if c.get("ocr_expect"):
            extra.append("Text that must appear exactly: " + "; ".join(f'"{t}"' for t in c["ocr_expect"]))
        proto = PROTOCOL[s["category_l2"]]
        reqs.append({"key": f"{s['id']}|scpr", "images": [src, outp],
                     "prompt": SCPR_PROMPT.format(instruction=s["instruction"], extra="\n".join(extra),
                                                  pr_guide=PR_GUIDE[proto])})
        medium = (f"\nThe image was requested as: \"{s['instruction_alt'] if s['lang'] != 'en' else s['instruction']}\"; "
                  "intended stylisation (brush strokes, pixelation, print texture) is not a flaw.") if proto == "style" else ""
        reqs.append({"key": f"{s['id']}|pq", "images": [outp], "prompt": PQ_PROMPT.format(medium=medium)})
        for side, phrases in ground_phrases(s).items():
            for i, ph in enumerate(phrases):
                reqs.append({"key": f"{s['id']}|ground|{side}|{i}", "images": [src if side == "src" else outp],
                             "prompt": GROUND_PROMPT.format(phrase=ph)})
        for i, o in enumerate(c.get("object_present") or []):
            reqs.append({"key": f"{s['id']}|present|{i}", "images": [outp], "prompt": YESNO_PROMPT.format(obj=o)})
        for i, o in enumerate(c.get("object_absent") or []):
            reqs.append({"key": f"{s['id']}|absent|{i}", "images": [outp], "prompt": YESNO_PROMPT.format(obj=o)})
        if c.get("count"):
            reqs.append({"key": f"{s['id']}|count", "images": [outp],
                         "prompt": COUNT_PROMPT.format(obj=c["count"]["object"])})
        if c.get("ocr_expect") or c.get("ocr_forbid"):
            reqs.append({"key": f"{s['id']}|read", "images": [outp], "prompt": READ_PROMPT})
    with tempfile.TemporaryDirectory(prefix="editimg-judge-") as jt:
        rq, rs = Path(jt) / "req.json", Path(jt) / "res.json"
        rq.write_text(json.dumps(reqs, ensure_ascii=False))
        jenv = base_env(jt)
        jenv.update({"HF_HOME": JUDGE_HOME, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "VLLM_NO_USAGE_STATS": "1", "TMPDIR": jt})
        jp = subprocess.run([python, str(HERE / "judge.py"), str(rq), str(rs)], env=jenv,
                            capture_output=True, text=True, timeout=1500)
        if jp.returncode != 0 or not rs.exists():
            print(jp.stdout[-3000:], jp.stderr[-6000:], sep="\n")
            sys.exit("judge failed")
        ans = json.loads(rs.read_text())

    # ---- per-sample scores
    done_ids = {s["id"] for s in done}
    per = {}
    for s in samples:
        sid, c = s["id"], s.get("checks") or {}
        r = {"category": s["category_l1"], "sub": s["category_l2"]}
        if sid not in done_ids:
            r.update(score=0.0, sc=0.0, pq=0.0, pr=0.0, pr_judge=0.0, geometry=0.0, error=errors.get(sid, "no output")[-500:])
            per[sid] = r
            continue
        outp = res_dir / f"{sid}.png"
        a = parse_scores(ans.get(f"{sid}|scpr", ""), ["SC", "PR"])
        b = parse_scores(ans.get(f"{sid}|pq", ""), ["PQ"])
        sc, pr_judge, pq = a["SC"] or 0.0, a["PR"] or 0.0, b["PQ"] or 0.0
        prog = {}
        # global deformation (zoom / shift / stretch / perspective) is invisible to the judge at the
        # few-pixel scale, so the measured geometry score g in [0, 1] is deducted from the judge's PR
        # (g = exp(-(p90 - 0.5 px) / 6 px): 1 px -> -0.4 points, 5 px -> -2.7, 10 px -> -4.0)
        g = 1.0
        t = tools(s)
        boxes = []
        for side, phrases in ground_phrases(s).items():
            for i, _ in enumerate(phrases):
                boxes += parse_boxes(ans.get(f"{sid}|ground|{side}|{i}", ""))
        if t["geometry"] or t["preserve"] or t["detail"]:
            fd = fidelity(split_dir / s["image"], outp,
                          split_dir / s["edit_mask"] if s.get("edit_mask") else None,
                          reframe=bool(c.get("aspect")), boxes=boxes, geometry_on=t["geometry"],
                          # without a mask or grounded region a local edit cannot be separated from
                          # the rest, so only the outlier-robust detail median is measured
                          preserve_on=t["preserve"] and bool(s.get("edit_mask") or boxes or c.get("aspect")),
                          detail_on=t["detail"])
            for k in ("preserve", "detail"):
                if fd[k] is not None:
                    prog[k] = fd[k]
            if fd["geometry"] is not None:
                g = fd["geometry"]
            r.update(fd["info"])
        r.update(protocol=PROTOCOL[s["category_l2"]], edit_boxes=boxes)
        pr = max(0.0, pr_judge - GEOM_PR_PENALTY * (1 - g))
        r.update(sc=sc, pr=pr, pr_judge=pr_judge, geometry=g, pq=pq, reason=a["reason"], pq_reason=b["reason"])
        judge = math.sqrt(sc * pq) / 10 * (0.5 + 0.5 * pr / 10)
        objs = [parse_yes(ans.get(f"{sid}|present|{i}", "")) for i, _ in enumerate(c.get("object_present") or [])]
        objs += [not parse_yes(ans.get(f"{sid}|absent|{i}", "")) for i, _ in enumerate(c.get("object_absent") or [])]
        if objs:
            prog["objects"] = sum(objs) / len(objs)
        if c.get("count"):
            got = parse_int(ans.get(f"{sid}|count", ""))
            prog["count"] = float(got == c["count"]["n"])
            r["count_got"] = got
        if c.get("ocr_expect") or c.get("ocr_forbid"):
            # three independent readings (two OCR models, the judge's transcription); a target counts
            # as present when any reading contains it
            judge_read = ans.get(f"{sid}|read", "").replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
            texts = ocr_text(outp) + [judge_read]
            cer = lambda t: min(best_cer(t, x) for x in texts)
            parts = [1 - cer(t) for t in c.get("ocr_expect") or []]
            parts += [float(cer(t) > 0.34) for t in c.get("ocr_forbid") or []]
            prog["text"] = sum(parts) / len(parts)
            r["ocr"] = " | ".join(x[:150] for x in texts)
        if c.get("aspect"):
            w, h = Image.open(outp).size
            prog["aspect"] = max(0.0, 1 - 5 * abs((w / h) / c["aspect"] - 1))
        r["checks"] = prog
        program = sum(prog.values()) / len(prog) if prog else None
        score = judge if program is None else 0.5 * judge + 0.5 * program
        if sc < SC_GATE:
            score = 0.0
        r.update(judge=judge, program=program, score=score)
        per[sid] = r

    cats = sorted({r["category"] for r in per.values()})
    cat_score = {c: float(np.mean([r["score"] for r in per.values() if r["category"] == c])) for c in cats}
    mean = lambda k: float(np.mean([r[k] for r in per.values()]))
    progs = [r["program"] for r in per.values() if r.get("program") is not None]
    reward = {"overall": float(np.mean(list(cat_score.values()))), "n": len(samples),
              "sc": mean("sc"), "pq": mean("pq"), "pr": mean("pr"), "pr_judge": mean("pr_judge"),
              "geometry": mean("geometry"),
              "program": float(np.mean(progs)) if progs else 0.0,
              "failed": len(samples) - len(done)}
    reward.update({f"cat_{c}": v for c, v in cat_score.items()})
    reward["metrics"] = {"per_sample": per}
    if split == "dev":
        cmp_dir = out_dir / "compare"
        cmp_dir.mkdir(exist_ok=True)
        arts = []
        for s in done:
            comparison(split_dir / s["image"], res_dir / f"{s['id']}.png", cmp_dir / f"{s['id']}.jpg")
            r = per[s["id"]]
            arts.append({"path": f"compare/{s['id']}.jpg", "role": "sample",
                         "note": f"{s['id']} score={r['score']:.2f} SC={r['sc']:.0f} PQ={r['pq']:.0f} PR={r['pr_judge']:.0f}x{r['geometry']:.2f} (geometry): {r.get('reason', '')[:200]}"})
        reward["artifacts"] = arts
        reward["info"] = {"solution_log_tail": log[-2000:]}
    (out_dir / "reward.json").write_text(json.dumps(reward, ensure_ascii=False))
    print(json.dumps({k: v for k, v in reward.items() if not isinstance(v, (dict, list))}))


if __name__ == "__main__":
    main()
