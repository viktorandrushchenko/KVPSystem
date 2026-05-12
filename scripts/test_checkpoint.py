from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.model import DeepseekOcrKvpExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one DeepSeek-OCR KVP extraction smoke test.")
    parser.add_argument("image", type=Path, help="Path to a document image.")
    parser.add_argument("--base-model", type=Path, default=Path("deepseek_ocr"))
    parser.add_argument("--adapter", type=Path, default=Path("checkpoint-1"))
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    extractor = DeepseekOcrKvpExtractor(args.base_model, args.adapter)
    status = extractor.status()
    if not status.ready:
        print(status.message)
        return 2

    result = extractor.extract(args.image, prompt=args.prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
