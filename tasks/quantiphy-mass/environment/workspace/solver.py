"""Starter: the main photo, the product type and the dimensions, one backbone call."""

from model import ask, image_part

WORKERS = 8


def answer(item: dict) -> str:
    d = item["dimensions_cm"]
    text = (f"This is a {item['product_type']}. Its dimensions are {d['length']} x {d['width']} x "
            f"{d['height']} cm (length x width x height).\n"
            "How much does it weigh in kilograms? Answer with a single number only.")
    return ask([image_part(item["images"][0]), {"type": "text", "text": text}], max_tokens=64)
