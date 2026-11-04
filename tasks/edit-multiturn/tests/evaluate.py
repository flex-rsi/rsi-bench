"""Evaluator for edit-multiturn.

1. Edit, turn by turn: one child process per GPU (sandboxed when $RSIB_SANDBOX is set) builds the
   solution's SessionEditor and is driven over a pipe (child_run.py): reset(source image), then
   step(instruction) once per turn. The next instruction is sent only after the child has returned
   the current turn's image, so a solution never sees a later turn; it gets no session file and no
   label. Time budget per GPU: STARTUP_S once plus SEC_PER_TURN for each of its turns, pooled over
   its sessions. A turn that fails (error, crash, budget used up, no image, aspect ratio changed by
   more than ASPECT_TOL) ends its session: it and the later turns score 0. Every turn is judged at
   the backbone's output size for the source (native_size): an image more than NATIVE_TOL px off
   it is resized to it first, so a smaller image hides no artifacts.
2. Score here with the frozen GLM-4.5V judge (judge.py, scoring.py): per turn, MSE-Bench's own
   prompt (prompt following PF, consistency CONS, with the source and all turns so far) and a
   perceptual-quality prompt (PQ, the turn's image alone).
   turn score = sqrt(PF*PQ)/10 * (0.5 + 0.5*CONS/10), 0 if PF < 3; session score = mean over its
   five turns; score = mean over sessions. Secondary: MSE-Bench's cumulative success rate per turn.
3. Write reward.json; on dev with every turn's judge reasons and a contact sheet per session.

Maintainer options: EDIT_MULTITURN_LIMIT=n scores the first n sessions only (smoke tests);
EDIT_MULTITURN_OUTPUTS=<dir> rescores saved images <dir>/<session>/turn<k>.png without running
the solution.
"""

import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scoring  # noqa: E402

TURNS = 5
STARTUP_S = 300        # per GPU, once: building the SessionEditor
SEC_PER_TURN = 60      # per GPU and turn, pooled over the GPU's sessions
ASPECT_TOL = 0.05      # a turn's image must keep the source's aspect ratio within 5 %
NATIVE_AREA = 1024 * 1024  # turns are judged at the backbone's output size, about 1 MP (native_size)
NATIVE_TOL = 32        # px; within one rounding step of that size an image is judged as it is
JUDGE_TIMEOUT_S = 900
MODELS = os.environ.get("EDIT_MULTITURN_MODELS")         # read-only HF cache with the allowed models
JUDGE_HOME = os.environ.get("EDIT_MULTITURN_JUDGE_HOME")  # HF_HOME of the judge; never shown to the solution


def solution_cmd(python, io_dir):
    """The solution's process; through the sandbox launcher when the run has one."""
    child = Path(io_dir) / "_child_run.py"
    shutil.copy(HERE / "child_run.py", child)
    cmd = [python, str(child)]
    prefix = json.loads(os.environ.get("RSIB_SANDBOX") or "[]")
    return prefix + ["--rw", str(io_dir), "--cwd", str(io_dir), "--"] + cmd if prefix else cmd


def gpus():
    return [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]


def base_env(home):
    env = {k: os.environ[k] for k in ("PATH", "LD_LIBRARY_PATH", "LANG", "TMPDIR", "CUDA_DEVICE_ORDER",
                                        "RSIB_CACHE", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME") if k in os.environ}
    env["HOME"] = os.environ.get("HOME", home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def descendants(pid):
    out = []
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            kids = [int(k) for k in (task / "children").read_text().split()]
        except (OSError, ValueError):
            continue
        for k in kids:
            out += [k] + descendants(k)
    return out


def kill_tree(proc):
    """Stop a child process and everything it started (vLLM workers, the sandbox's processes)."""
    if proc.poll() is None:  # still ours: its pid cannot have been reused
        for pid in [proc.pid] + descendants(proc.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    proc.wait()


def become_subreaper():
    """Orphans of a crashed process (e.g. its workers) are re-parented to this process instead of
    init, so stop_leftovers() still finds them before the judge needs the GPUs."""
    import ctypes
    try:
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        pass


def stop_leftovers():
    """Kill and reap every process still below this one (call when no child should be running)."""
    for pid in descendants(os.getpid()):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.1)


def native_size(w, h):
    """The size Qwen-Image-Edit outputs for an input of w x h (QwenImageEditPipeline: about
    NATIVE_AREA pixels at the input's aspect ratio, both sides rounded to multiples of 32)."""
    width = math.sqrt(NATIVE_AREA * w / h)
    return round(width / 32) * 32, round(width * h / w / 32) * 32


def keep_output(source, image, dest):
    """Store one turn's image at the size it is judged at; returns why it is unusable, or None.
    Within NATIVE_TOL px of the backbone's output size for the source the image is kept as it is
    (the starter's images are, across all five turns); otherwise it is resized to that size."""
    if not image.exists():
        return "no image written"
    try:
        with Image.open(image) as im:
            im.load()
            w, h = im.size
    except Exception as e:  # noqa: BLE001 - any unreadable file
        return f"unreadable image: {e}"
    with Image.open(source) as src:
        sw, sh = src.size
    if min(w, h) < 16 or abs((w / h) / (sw / sh) - 1) > ASPECT_TOL:
        return f"image {w}x{h} does not keep the aspect ratio of the source ({sw}x{sh})"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tw, th = native_size(sw, sh)
    if abs(w - tw) > NATIVE_TOL or abs(h - th) > NATIVE_TOL:
        with Image.open(image) as im:
            im.convert("RGB").resize((tw, th), Image.LANCZOS).save(dest, format="PNG")
    else:
        shutil.copy(image, dest)
    return None


class Driver(threading.Thread):
    """Feeds one GPU's share of the sessions to one child process, one turn at a time."""

    def __init__(self, gpu, sessions, ctx):
        super().__init__(daemon=True)
        self.gpu, self.sessions, self.ctx = gpu, sessions, ctx
        self.io = ctx["tmp"] / f"gpu{gpu}"
        self.io.mkdir()
        self.log_path = ctx["tmp"] / f"solution_gpu{gpu}.log"
        self.deadline = time.time() + STARTUP_S + SEC_PER_TURN * TURNS * len(sessions)
        self.proc = None
        self.result = {}   # session id -> [{"ok", "s", "error"}] for the turns tried
        self.events = []

    def start_child(self):
        env = base_env(str(self.io))
        env.update(CUDA_VISIBLE_DEVICES=self.gpu, RSIB_IO_DIR=str(self.io),
                   RSIB_SOLVER_DIR=str(self.ctx["workspace"]), EDIT_MULTITURN_MODELS=MODELS or "",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with open(self.log_path, "a") as log:
            self.proc = subprocess.Popen(solution_cmd(self.ctx["python"], self.io), env=env, cwd=self.io,
                                         text=True, bufsize=1, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=log)
        self.answers = queue.Queue()
        threading.Thread(target=self._read, args=(self.proc.stdout, self.answers), daemon=True).start()
        t = time.time()
        a = self.answer()
        self.events.append(f"GPU {self.gpu}: SessionEditor ready after {time.time() - t:.0f}s" if a.get("ok")
                           else f"GPU {self.gpu}: SessionEditor failed after {time.time() - t:.0f}s: "
                                f"{a['error'][-1500:]}")
        return a

    @staticmethod
    def _read(stream, answers):
        for line in stream:
            answers.put(line)
        answers.put(None)

    def answer(self):
        try:
            line = self.answers.get(timeout=max(0.0, self.deadline - time.time()))
        except queue.Empty:
            return {"ok": False, "fatal": True, "error": "time budget used up"}
        if line is None:
            self.proc.wait()
            return {"ok": False, "fatal": True,
                    "error": f"the solution's process exited (code {self.proc.returncode})"}
        try:
            return json.loads(line)
        except ValueError:
            return {"ok": False, "fatal": True, "error": "unreadable answer: " + line[:200]}

    def call(self, **msg):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except OSError:
            pass  # the process is gone; answer() reports it
        return self.answer()

    def stop(self, graceful=True):
        """End the child: asked to exit after the last session, killed at once when it failed."""
        if self.proc is None:
            return
        if graceful:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                self.proc.stdin.close()
                self.proc.wait(timeout=30)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        kill_tree(self.proc)
        self.proc = None

    def run(self):
        try:
            self._run()
        finally:
            self.stop()

    def _run(self):
        ready, crashes, start_error = False, 0, None
        for s in self.sessions:
            sid = s["id"]
            tried = self.result[sid] = []
            if not ready:
                why = None
                if time.time() >= self.deadline:
                    why = "time budget used up"
                elif start_error:
                    why = "SessionEditor failed to start: " + start_error
                elif crashes > 2:
                    why = "the solution's process failed too often"
                else:
                    a = self.start_child()
                    if a.get("ok"):
                        ready = True
                    else:
                        self.stop(graceful=False)
                        why = a["error"][-1500:]
                        if not a.get("fatal"):  # SessionEditor() raised: starting again would too
                            start_error = why
                            why = "SessionEditor failed to start: " + why
                if not ready:
                    tried.append({"ok": False, "s": 0.0, "error": "not run: " + why})
                    continue
            shutil.copy(self.ctx["split_dir"] / s["image"], self.io / "source.png")
            t = time.time()
            a = self.call(cmd="reset", image=str(self.io / "source.png"))
            if not a.get("ok"):
                tried.append({"ok": False, "s": time.time() - t, "error": "reset: " + a["error"][-1500:]})
            else:
                for k, instruction in enumerate(s["instructions"], 1):
                    out = self.io / "out.png"
                    out.unlink(missing_ok=True)
                    t = time.time()
                    a = self.call(cmd="step", instruction=instruction, out=str(out))
                    err = a["error"][-1500:] if not a.get("ok") else keep_output(
                        self.ctx["split_dir"] / s["image"], out, self.ctx["res_dir"] / sid / f"turn{k}.png")
                    tried.append({"ok": err is None, "s": time.time() - t, "error": err})
                    if err is not None:
                        break
            if a.get("fatal"):
                self.events.append(f"GPU {self.gpu}: {a['error']} during {sid}")
                self.stop(graceful=False)
                ready = False
                crashes += 1


def edit(sessions, ctx):
    """Run the solution on every session; returns session id -> turns tried, and log lines."""
    devices = gpus() or ["0"]
    drivers = [Driver(g, sessions[i::len(devices)], ctx) for i, g in enumerate(devices)
               if sessions[i::len(devices)]]
    for d in drivers:
        d.start()
    for d in drivers:
        d.join()
    result = {sid: tried for d in drivers for sid, tried in d.result.items()}
    events = [e for d in drivers for e in d.events]
    logs = []
    for d in drivers:
        if d.log_path.exists():
            logs.append(f"--- GPU {d.gpu}\n" + d.log_path.read_text(errors="replace")[-4000:])
    return result, events, "\n".join(logs)


def rescore(sessions, ctx, src_dir):
    """Maintainer use: take saved images instead of running the solution."""
    result = {}
    for s in sessions:
        tried = result[s["id"]] = []
        for k in range(1, TURNS + 1):
            err = keep_output(ctx["split_dir"] / s["image"], src_dir / s["id"] / f"turn{k}.png",
                              ctx["res_dir"] / s["id"] / f"turn{k}.png")
            tried.append({"ok": err is None, "s": 0.0, "error": err})
            if err:
                break
    return result, [f"rescored saved images from {src_dir}"], ""


def judge(reqs, tmp, python):
    """Answers of the frozen judge, key -> text; one retry (e.g. GPU memory not yet released)."""
    rq, rs = tmp / "judge_requests.json", tmp / "judge_answers.json"
    rq.write_text(json.dumps(reqs, ensure_ascii=False))
    env = base_env(str(tmp))
    env.update({"CUDA_VISIBLE_DEVICES": ",".join(gpus()), "HF_HOME": JUDGE_HOME, "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "TMPDIR": str(tmp)})
    if not gpus():
        env.pop("CUDA_VISIBLE_DEVICES")
    tail = ""
    for attempt in (1, 2):
        rs.unlink(missing_ok=True)
        log = tmp / f"judge_{attempt}.log"
        with open(log, "w") as f:
            proc = subprocess.Popen([python, str(HERE / "judge.py"), str(rq), str(rs)], env=env,
                                    stdout=f, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=JUDGE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                kill_tree(proc)
        if proc.returncode == 0 and rs.exists():
            return json.loads(rs.read_text())
        tail = log.read_text(errors="replace")[-3000:]
        stop_leftovers()
        time.sleep(30)
    print(tail)
    sys.exit("the judge failed twice")


def contact_sheet(images, labels, dest, height=240, lines=5):
    tiles = []
    for p in images:
        with Image.open(p) as im:
            im = im.convert("RGB")
            tiles.append(im.resize((max(1, round(im.width * height / im.height)), height)))
    top = 4 + 13 * lines
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), height + top), "white")
    draw, x = ImageDraw.Draw(sheet), 0
    for tile, label in zip(tiles, labels):
        sheet.paste(tile, (x, top))
        for i, text in enumerate(textwrap.wrap(label, max(8, (tile.width - 4) // 6))[:lines]):
            draw.text((x + 2, 2 + 13 * i), text, fill="black")
        x += tile.width + 6
    sheet.save(dest, quality=85)


def main():
    split = os.environ.get("RSIB_SPLIT", "dev")
    split_dir = Path(os.environ["RSIB_DATA_DIR"])
    out_dir = Path(os.environ["RSIB_OUTPUT_DIR"])
    python = os.environ.get("RSIB_PYTHON", sys.executable)
    saved = os.environ.get("EDIT_MULTITURN_OUTPUTS")
    if not JUDGE_HOME or not (MODELS or saved):
        sys.exit("set EDIT_MULTITURN_MODELS (the backbone's HF cache, read-only to agents) and "
                 "EDIT_MULTITURN_JUDGE_HOME (the judge's HF_HOME), and list both in RSIB_PASS_ENV")
    sessions = [json.loads(l) for l in (split_dir / "sessions.jsonl").read_text().splitlines() if l.strip()]
    labels = {x["id"]: x for x in map(json.loads, (split_dir / "labels.jsonl").read_text().splitlines())}
    limit = int(os.environ.get("EDIT_MULTITURN_LIMIT", "0"))  # smoke tests only
    if limit:
        sessions = sessions[:limit]
    res_dir = out_dir / "outputs"
    if res_dir.exists():
        shutil.rmtree(res_dir)
    res_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="editmt-") as tmp:
        tmp = Path(tmp)
        ctx = {"tmp": tmp, "split_dir": split_dir, "res_dir": res_dir, "python": python,
               "workspace": Path(os.environ.get("RSIB_WORKSPACE", "."))}
        t0 = time.time()
        become_subreaper()
        tried, events, log = rescore(sessions, ctx, Path(saved)) if saved else edit(sessions, ctx)
        stop_leftovers()
        edit_s = time.time() - t0
        (out_dir / "solution.log").write_text("\n".join(events) + "\n" + log)
        if not any(t["ok"] for turns in tried.values() for t in turns):
            print("\n".join(events), log[-3000:], sep="\n")
            sys.exit("the solution produced no usable image")

        reqs = []
        for s in sessions:
            sid = s["id"]
            done = [k for k in range(1, TURNS + 1) if (res_dir / sid / f"turn{k}.png").exists()]
            if not done:
                continue
            src = scoring.judge_source(str(split_dir / s["image"]), str(res_dir / sid / "turn1.png"),
                                       str(tmp / f"{sid}_source.png"))
            for k in done:
                history = [str(res_dir / sid / f"turn{j}.png") for j in range(1, k + 1)]
                reqs.append({"key": f"{sid}|{k}|mse", "images": [src] + history, "group": sid,
                             "prompt": scoring.mse_prompt(s["instructions"][:k])})
                reqs.append({"key": f"{sid}|{k}|pq", "images": [history[-1]], "group": sid,
                             "prompt": scoring.PQ_PROMPT})
        t0 = time.time()
        ans = judge(reqs, tmp, python)
        stop_leftovers()
        judge_s = time.time() - t0

    per, unparsed = {}, 0
    for s in sessions:
        sid, op = s["id"], labels[s["id"]]["operation"]
        turns = []
        for k in range(1, TURNS + 1):
            e = tried[sid][k - 1] if k <= len(tried[sid]) else {
                "ok": False, "s": 0.0, "error": "not run: an earlier turn failed"}
            t = {"turn": k, "operation": op[k - 1], "instruction": s["instructions"][k - 1],
                 "edit_s": round(e["s"], 1), "error": e["error"], "pf": None, "cons": None, "pq": None}
            if (res_dir / sid / f"turn{k}.png").exists():
                m = scoring.parse(ans.get(f"{sid}|{k}|mse"), ("prompt_following", "consistency"))
                q = scoring.parse(ans.get(f"{sid}|{k}|pq"), ("PQ",))
                t.update(pf=m["prompt_following"], cons=m["consistency"], pq=q["PQ"],
                         reason=m["reason"], pq_reason=q["reason"])
                unparsed += None in (t["pf"], t["cons"], t["pq"])
            t["score"] = scoring.turn_score(t["pf"], t["cons"], t["pq"])
            turns.append(t)
        for t, ok in zip(turns, scoring.cumulative_success(turns)):
            t["success"] = ok
        per[sid] = {"score": sum(t["score"] for t in turns) / TURNS, "turns": turns}

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    all_turns = [t for p in per.values() for t in p["turns"]]
    reward = {"score": mean([p["score"] for p in per.values()]), "n": len(sessions)}
    for k in range(1, TURNS + 1):
        reward[f"score_t{k}"] = mean([p["turns"][k - 1]["score"] for p in per.values()])
    for k in range(1, TURNS + 1):
        reward[f"success_t{k}"] = mean([p["turns"][k - 1]["success"] for p in per.values()])
    for key in ("pf", "cons", "pq"):
        reward[key] = mean([t[key] or 0.0 for t in all_turns])
    reward["failed_turns"] = sum(t["error"] is not None for t in all_turns)
    reward["judge_unparsed"] = unparsed
    reward["edit_s_per_turn"] = mean([t["edit_s"] for t in all_turns if t["error"] is None])
    ops = sorted({t["operation"] for t in all_turns})
    reward["metrics"] = {
        "per_operation": {o: mean([t["score"] for t in all_turns if t["operation"] == o]) for o in ops},
        "per_session": per}
    reward["info"] = {"edit_wall_s": round(edit_s), "judge_wall_s": round(judge_s), "events": events}
    if split == "dev":
        sheets, arts = out_dir / "compare", []
        sheets.mkdir(exist_ok=True)
        for s in sessions:
            sid, p = s["id"], per[s["id"]]
            done = [t for t in p["turns"] if (res_dir / sid / f"turn{t['turn']}.png").exists()]
            contact_sheet([split_dir / s["image"]] + [res_dir / sid / f"turn{t['turn']}.png" for t in done],
                          ["source"] + [f"t{t['turn']} PF {t['pf']} C {t['cons']} PQ {t['pq']} = {t['score']:.2f} "
                                        f"| {t['instruction']}" for t in done], sheets / f"{sid}.jpg")
            failed = next((t for t in p["turns"] if t["error"]), None)
            note = f"{sid} score={p['score']:.2f}: " + "; ".join(
                f"t{t['turn']} PF/CONS/PQ {t['pf']}/{t['cons']}/{t['pq']}" for t in done)
            if failed:
                note += f"; t{failed['turn']} failed: {failed['error'][:300]}"
            arts.append({"path": f"compare/{sid}.jpg", "role": "sample", "note": note})
        reward["artifacts"] = arts
        reward["info"]["solution_log_tail"] = log[-3000:]
    (out_dir / "reward.json").write_text(json.dumps(reward, ensure_ascii=False))
    print(json.dumps({k: v for k, v in reward.items() if not isinstance(v, (dict, list))}))


if __name__ == "__main__":
    main()
