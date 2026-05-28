from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch LoRA training on manually annotated KVP data.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL file exported from the annotation UI.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the new LoRA checkpoint.")
    parser.add_argument("--steps", type=int, default=100, help="Training step count passed to KVP_TRAIN_COMMAND.")
    args = parser.parse_args()

    samples = validate_dataset(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "samples": samples,
        "steps": args.steps,
        "status": "prepared",
        "note": (
            "The annotation dataset is exported in DeepSeek-OCR conversation format. "
            "Set KVP_TRAIN_COMMAND to a real local training command if you want the web UI "
            "to launch full fine-tuning automatically."
        ),
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command_template = os.getenv("KVP_TRAIN_COMMAND", "").strip()
    if not command_template:
        print(f"Prepared {samples} annotated samples.")
        print(f"Dataset: {args.dataset}")
        print(f"Output directory: {args.output_dir}")
        print(
            "KVP_TRAIN_COMMAND is not set, so real GPU fine-tuning was not started. "
            "Use {dataset}, {output_dir}, and {steps} placeholders in that command."
        )
        return 2

    command = command_template.format(dataset=str(args.dataset), output_dir=str(args.output_dir), steps=args.steps)
    print(f"Running training command: {command}")
    result = subprocess.run(shlex.split(command), cwd=str(Path.cwd()))
    return result.returncode


def validate_dataset(path: Path) -> int:
    if not path.exists():
        raise SystemExit(f"Dataset file does not exist: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise SystemExit(f"Invalid messages at line {line_number}")
            count += 1
    if count == 0:
        raise SystemExit("Dataset is empty")
    return count


if __name__ == "__main__":
    sys.exit(main())
