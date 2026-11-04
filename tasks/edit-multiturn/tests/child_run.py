"""Runs the submitted SessionEditor in a process that never sees the session files or the labels.

The evaluator drives it over a pipe, one JSON line per command on stdin and one answer per line
on the original stdout (whatever the solution prints goes to stderr):
  (start)                                              -> {"ok": true, "ready": true}
  {"cmd": "reset", "image": <path>}                    -> {"ok": true}
  {"cmd": "step", "instruction": <str>, "out": <path>} -> {"ok": true}, the image saved as PNG
  {"cmd": "exit"}
A call that raises answers {"ok": false, "error": <traceback tail>}.
"""

import json
import os
import sys
import traceback


def main():
    answers = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # prints of the solution go to stderr, not into the answers

    def say(**kw):
        answers.write(json.dumps(kw) + "\n")

    sys.path.insert(0, os.environ["RSIB_SOLVER_DIR"])
    try:
        from PIL import Image

        import editor
        ed = editor.SessionEditor("cuda")
    except BaseException:
        say(ok=False, error=traceback.format_exc()[-3000:])
        return
    say(ok=True, ready=True)
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["cmd"] == "exit":
            return
        try:
            if msg["cmd"] == "reset":
                with Image.open(msg["image"]) as im:
                    ed.reset(im.convert("RGB"))
            else:
                out = ed.step(msg["instruction"])
                if not isinstance(out, Image.Image):
                    raise TypeError(f"step() returned {type(out).__name__}, not a PIL image")
                out.convert("RGB").save(msg["out"], format="PNG")
            say(ok=True)
        except Exception:
            say(ok=False, error=traceback.format_exc()[-3000:])


if __name__ == "__main__":
    main()
