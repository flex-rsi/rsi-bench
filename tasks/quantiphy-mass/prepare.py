"""Build the quantiphy-mass splits under --out from Amazon Berkeley Objects (Collins et al., CVPR 2022;
CC BY 4.0): product listings with catalog photos, the item's dimensions and its weight.

    rsib task prepare quantiphy-mass                       # downloads listings and photos over HTTPS
    ABO_SOURCE=/path/with/listings_*.json.gz+images.csv.gz rsib task prepare quantiphy-mass

Items: listings with a weight, all three dimensions and a main photo; no multipacks or sets (by
name), no consumables (food, supplements, cleaning and medical supplies); weight 10 g to 150 kg
and weight over the box volume 5 to 5000 kg/m3 (drops unit and listing errors). Variants of one
product (same brand and first six words of the name, e.g. colors) form one group, and one item per
group is kept. 1200 groups are drawn (fixed seed) with at most 3% of a split from one product
type, and split by group:
    dev 300 items    val 300    test 600
Each split gets images/ (the main photo and up to two other photos, longer side at most 1024 px),
questions.jsonl ({"id", "images", "product_type", "dimensions_cm": {"length", "width",
"height"}}) and labels.jsonl ({"id", "answer"}: kg). Dev also gets items.jsonl with the names.
"""

import argparse
import collections
import csv
import gzip
import io
import json
import os
import random
import re
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"
SEED = 0
SIZES = {"dev": 300, "val": 300, "test": 600}
TYPE_CAP = 0.03
MAX_SIDE = 1024
KG = {"pounds": 0.45359237, "kilograms": 1.0, "grams": 1e-3, "ounces": 0.028349523}
CM = {"inches": 2.54, "centimeters": 1.0, "millimeters": 0.1, "meters": 100.0}
CONSUMABLE = {
    "GROCERY", "HEALTH_PERSONAL_CARE", "NUTRITIONAL_SUPPLEMENT", "DIETARY_SUPPLEMENT", "VITAMIN",
    "COFFEE", "TEA", "HERB", "SAUCE", "SPICE", "CONDIMENT", "SNACK_FOOD_BAR", "CANDY",
    "BEVERAGE", "PET_FOOD", "JANITORIAL_SUPPLY", "FOOD_SERVICE_SUPPLY", "SAFETY_SUPPLY",
    "INDUSTRIAL_SUPPLY", "LAB_SUPPLY", "PROFESSIONAL_HEALTHCARE", "MEDICATION", "BEAUTY",
    "SKIN_CARE_AGENT", "HAIR_CARE_PRODUCT", "PAPER_PRODUCT", "WASTE_BAG", "TOILET_PAPER",
    "BATTERY", "LIGHT_BULB", "BISS"}
MULTI = re.compile(r"\b(packs?|pk|count|ct|pcs|pieces?|pairs?|set of|sets|bulk|case of|bundle|"
                   r"lot of|qty|quantity|refills?|multipack|\d+\s*-?\s*(x|pack|count|pc|piece))\b",
                   re.I)


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def source(name: str, src: Path | None, cache: Path) -> Path:
    if src:
        return src / name
    dest = cache / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        rel = f"listings/metadata/{name}" if name.startswith("listings") else f"images/metadata/{name}"
        dest.write_bytes(fetch(f"{BASE}/{rel}"))
    return dest


def first_en(values) -> str | None:
    for v in values or []:
        if str(v.get("language_tag", "")).startswith("en"):
            return v.get("value")
    return None


def parse(r: dict) -> dict | None:
    w, dims, img = r.get("item_weight"), r.get("item_dimensions"), r.get("main_image_id")
    name = first_en(r.get("item_name"))
    if not (w and dims and img and name and r.get("product_type")):
        return None
    try:
        nw = w[0].get("normalized_value", w[0])
        kg = float(nw["value"]) * KG[nw["unit"]]
        cm = {}
        for k in ("length", "width", "height"):
            nv = dims[k].get("normalized_value", dims[k])
            cm[k] = round(float(nv["value"]) * CM[nv["unit"]], 2)
    except (KeyError, ValueError, TypeError):
        return None
    ptype = r["product_type"][0]["value"]
    if ptype in CONSUMABLE or MULTI.search(name):
        return None
    if not 0.01 <= kg <= 150 or min(cm.values()) < 0.3 or max(cm.values()) > 400:
        return None
    if not 5 <= kg / (cm["length"] * cm["width"] * cm["height"] * 1e-6) <= 5000:
        return None
    brand = first_en(r.get("brand")) or ""
    return {"item_id": r["item_id"], "kg": kg, "cm": cm, "ptype": ptype, "name": name,
            "group": (brand.lower(), " ".join(re.sub(r"[^a-z ]", "", name.lower()).split()[:6])),
            "photos": [img] + list(r.get("other_image_id") or [])[:2]}


def draw(groups: list[list[dict]], rng: random.Random) -> dict[str, list[dict]]:
    pool = [rng.choice(g) for g in groups]
    rng.shuffle(pool)
    out = {}
    for split, size in SIZES.items():
        cap, per, picked, rest = max(1, int(TYPE_CAP * size)), collections.Counter(), [], []
        for item in pool:
            if len(picked) < size and per[item["ptype"]] < cap:
                per[item["ptype"]] += 1
                picked.append(item)
            else:
                rest.append(item)
        out[split], pool = picked, rest
    return out


def save_photo(path_rel: str, src: Path | None, dest: Path) -> None:
    raw = (src / "images" / path_rel).read_bytes() if src and (src / "images").exists() else \
        fetch(f"{BASE}/images/original/{path_rel}")
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    im.save(dest, "JPEG", quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    src = Path(os.environ["ABO_SOURCE"]) if os.environ.get("ABO_SOURCE") else None
    cache = out / "_download"

    items = []
    for h in "0123456789abcdef":
        with gzip.open(source(f"listings_{h}.json.gz", src, cache), "rt") as f:
            items += [x for x in (parse(json.loads(line)) for line in f) if x]
    with gzip.open(source("images.csv.gz", src, cache), "rt") as f:
        paths = {row["image_id"]: row["path"] for row in csv.DictReader(f)}
    by_group: dict[tuple, list[dict]] = {}
    for x in items:
        x["photos"] = [p for p in x["photos"] if p in paths]
        if x["photos"]:
            by_group.setdefault(x["group"], []).append(x)
    groups = [by_group[k] for k in sorted(by_group)]
    splits = draw(groups, random.Random(SEED))
    print(len(items), "items,", len(groups), "groups")

    for split, chosen in splits.items():
        d = out / split
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        questions, labels, names, jobs = [], [], [], []
        for k, x in enumerate(chosen):
            qid = f"{split}_{k:04d}"
            rels = [f"images/{qid}_{j}.jpg" for j in range(len(x["photos"]))]
            jobs += [(paths[p], d / rel) for p, rel in zip(x["photos"], rels)]
            questions.append({"id": qid, "images": rels,
                              "product_type": x["ptype"].lower().replace("_", " "),
                              "dimensions_cm": x["cm"]})
            labels.append({"id": qid, "answer": round(x["kg"], 4)})
            names.append({"id": qid, "name": x["name"], "item_id": x["item_id"]})
        with ThreadPoolExecutor(32) as pool:
            list(pool.map(lambda j: save_photo(j[0], src, j[1]), jobs))
        (d / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
        (d / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
        if split == "dev":
            (d / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in names))
        types = collections.Counter(x["ptype"] for x in chosen)
        print(split, len(chosen), "items,", len(jobs), "photos,", len(types), "product types")
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
