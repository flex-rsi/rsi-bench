# edit-image-unlimited workspace

`editor.py` is the solution. The evaluator imports it once per worker (one worker per
`GPUS_PER_WORKER` GPUs, default 1) and calls `Editor(device).edit(image_path, instruction,
out_path, meta)` for each sample. Any open-weight model or tool may be used; see
`tasks/edit-image-unlimited/instruction.md` for the interface, where models must live and the rules.
