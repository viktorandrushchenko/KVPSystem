from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
import sys
from typing import Any

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.model import DeepseekOcrKvpExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the KVP checkpoint on KVP10k-style data.")
    parser.add_argument("--gts", type=Path, default=Path("KVP10k/test/gts"))
    parser.add_argument("--images", type=Path, default=Path("KVP10k/test/images"))
    parser.add_argument("--limit", type=int, default=20, help="Maximum key-value pairs to evaluate.")
    parser.add_argument("--max-per-document", type=int, default=0, help="Limit pairs per document; 0 means unlimited.")
    parser.add_argument("--output", type=Path, default=Path("data/eval_kvp10k_predictions.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/eval_kvp10k_summary.json"))
    args = parser.parse_args()

    samples = list(iter_samples(args.gts, args.images, args.limit, args.max_per_document))
    if not samples:
        raise SystemExit(f"No samples found in {args.gts} and {args.images}")

    extractor = DeepseekOcrKvpExtractor()
    status = extractor.status()
    if not status.ready:
        raise SystemExit(status.message)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as out:
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] {sample['image_path'].name} :: {sample['key']}")
            try:
                pred = extractor.extract_key(sample["image_path"], sample["key"])
                record = {**sample_for_json(sample), "prediction": pred, "error": None}
            except Exception as exc:
                record = {**sample_for_json(sample), "prediction": None, "error": str(exc)}
            record["metrics"] = score_record(record)
            predictions.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    summary = summarize(predictions)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def iter_samples(gts_dir: Path, images_dir: Path, limit: int, max_per_document: int = 0):
    count = 0
    for json_path in sorted(gts_dir.glob("*.json")):
        image_path = find_image(images_dir, json_path.stem)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        data = json.loads(json_path.read_text(encoding="utf-8"))
        per_document = 0
        for item in data.get("kvps_list", []):
            key = ((item.get("key") or {}).get("text") or "").strip()
            value = ((item.get("value") or {}).get("text") or "").strip()
            bbox = (item.get("value") or {}).get("bbox")
            if not key or not value:
                continue
            yield {
                "json_path": json_path,
                "image_path": image_path,
                "key": key,
                "target_value": value,
                "target_bbox": normalize_bbox(bbox, width, height),
            }
            count += 1
            per_document += 1
            if limit and count >= limit:
                return
            if max_per_document and per_document >= max_per_document:
                break


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in [".png", ".jpg", ".jpeg"]:
        path = images_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def normalize_bbox(bbox: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * 999 / width)),
        int(round(y1 * 999 / height)),
        int(round(x2 * 999 / width)),
        int(round(y2 * 999 / height)),
    ]


def sample_for_json(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "json_path": str(sample["json_path"]),
        "image_path": str(sample["image_path"]),
        "key": sample["key"],
        "target_value": sample["target_value"],
        "target_bbox": sample["target_bbox"],
    }


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    pred = record.get("prediction") or {}
    target = normalize_text(record.get("target_value") or "")
    value = normalize_text(pred.get("value") or "")
    bbox_iou = iou(first_box(pred.get("bbox")), record.get("target_bbox"))
    return {
        "parsed": bool(pred and pred.get("value")),
        "value_exact": target == value and bool(target),
        "value_similarity": SequenceMatcher(None, target, value).ratio() if target or value else 0.0,
        "bbox_iou": bbox_iou,
        "bbox_iou_50": bbox_iou is not None and bbox_iou >= 0.5,
    }


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def first_box(value: Any) -> list[int] | None:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list) and len(value) == 4:
        return value
    return None


def iou(a: list[int] | None, b: list[int] | None) -> float | None:
    if not a or not b:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [record["metrics"] for record in records]
    bbox_values = [m["bbox_iou"] for m in metrics if m["bbox_iou"] is not None]
    return {
        "samples": len(records),
        "errors": sum(1 for record in records if record.get("error")),
        "parse_success_rate": mean([m["parsed"] for m in metrics]),
        "value_exact_accuracy": mean([m["value_exact"] for m in metrics]),
        "value_similarity_mean": mean([m["value_similarity"] for m in metrics]),
        "bbox_iou_mean": mean(bbox_values),
        "bbox_iou_50_accuracy": mean([m["bbox_iou_50"] for m in metrics if m["bbox_iou"] is not None]),
    }


def mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
