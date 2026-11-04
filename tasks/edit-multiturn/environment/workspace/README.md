# edit-multiturn workspace

`editor.py` is the solution. The evaluator creates one `SessionEditor(device)` per GPU and, for
each session, calls `reset(image)` with the source image, then `step(instruction)` once per turn;
each call returns that turn's edited image. A turn's instruction is only given after the previous
turn's image has been returned. See `tasks/edit-multiturn/instruction.md` for the full interface
and rules.
