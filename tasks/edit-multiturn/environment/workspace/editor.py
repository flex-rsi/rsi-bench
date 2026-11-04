"""Starter solution: each turn edits the previous turn's result with the current instruction.

Qwen-Image-Edit with the 8-step Lightning LoRA (and the scheduler it was distilled with), one
sample per turn, fixed seed.
"""

import math
import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from PIL import Image

MODELS = os.environ.get("EDIT_MULTITURN_MODELS")  # read-only HF cache with the allowed models
CACHE = os.path.join(MODELS, "hub") if MODELS else None
LORA = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"


class SessionEditor:
    """Edits one session at a time: reset() with its source image, then step() once per turn."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit", dtype=torch.bfloat16, cache_dir=CACHE).to(device)
        self.pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", cache_dir=CACHE, weight_name=LORA)
        self.pipe.fuse_lora()
        # the LoRA was distilled with shift 3 and no terminal shift (Qwen-Image-Lightning model card)
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(dict(
            self.pipe.scheduler.config, base_shift=math.log(3), max_shift=math.log(3), shift_terminal=None))
        self.pipe.set_progress_bar_config(disable=True)
        self.image = None

    def reset(self, image: Image.Image) -> None:
        """Start a new session from its source image."""
        self.image = image.convert("RGB")

    def step(self, instruction: str) -> Image.Image:
        """Apply this turn's instruction to the current image and return the result."""
        self.image = self.pipe(
            image=self.image, prompt=instruction, num_inference_steps=8, true_cfg_scale=1.0,
            generator=torch.Generator(self.device).manual_seed(0),
        ).images[0]
        return self.image
