from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DeepSeek-OCR base model from Hugging Face.")
    parser.add_argument("--repo-id", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--target", type=Path, default=Path("deepseek_ocr"))
    args = parser.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.target,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Downloaded {args.repo_id} to {args.target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
