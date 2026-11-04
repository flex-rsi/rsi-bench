"""Starter solution: pass the instruction unchanged to the fixed editing backbone.

Qwen-Image-Edit-2509 with the 8-step Lightning LoRA, one sample, fixed seed.
"""

import os

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

MODELS = os.environ.get("EDIT_IMAGE_MODELS", "/models/edit-image/hf_models")  # read-only HF cache (frozen starter models)
CACHE = os.path.join(MODELS, "hub") if MODELS else None


class Editor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16, cache_dir=CACHE).to(device)
        self.pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", cache_dir=CACHE,
            weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors")
        self.pipe.fuse_lora()
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, image_path: str, instruction: str, out_path: str, meta: dict) -> None:
        image = Image.open(image_path).convert("RGB")
        result = self.pipe(
            image=[image], prompt=instruction, negative_prompt=" ",
            num_inference_steps=8, true_cfg_scale=1.0,
            generator=torch.Generator(self.device).manual_seed(0),
        ).images[0]
        result.save(out_path)
