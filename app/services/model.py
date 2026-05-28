from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..config import ADAPTER_PATH, BASE_MODEL_PATH, BASE_SIZE, CROP_MODE, DEFAULT_PROMPT, DEVICE_MAP, IMAGE_SIZE, MAX_MEMORY
from .parser import parse_grounded_value, parse_key_values


@dataclass
class ModelStatus:
    ready: bool
    base_model_path: str
    adapter_path: str
    message: str


class DeepseekOcrKvpExtractor:
    def __init__(self, base_model_path: Path = BASE_MODEL_PATH, adapter_path: Path = ADAPTER_PATH) -> None:
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_error: str | None = None

    def status(self) -> ModelStatus:
        if not self.base_model_path.exists():
            return ModelStatus(
                ready=False,
                base_model_path=str(self.base_model_path),
                adapter_path=str(self.adapter_path),
                message="Base model directory is missing. Add it or set DEEPSEEK_OCR_BASE_MODEL.",
            )
        if not self.adapter_path.exists():
            return ModelStatus(
                ready=False,
                base_model_path=str(self.base_model_path),
                adapter_path=str(self.adapter_path),
                message="Adapter checkpoint directory is missing. Add it or set DEEPSEEK_OCR_ADAPTER.",
            )
        missing = self._missing_weight_files()
        if missing:
            sample = ", ".join(missing[:3])
            return ModelStatus(
                ready=False,
                base_model_path=str(self.base_model_path),
                adapter_path=str(self.adapter_path),
                message=f"Base model weights are incomplete. Missing: {sample}.",
            )
        if self._load_error:
            return ModelStatus(
                ready=False,
                base_model_path=str(self.base_model_path),
                adapter_path=str(self.adapter_path),
                message=self._load_error,
            )
        return ModelStatus(
            ready=True,
            base_model_path=str(self.base_model_path),
            adapter_path=str(self.adapter_path),
            message="Model files are present.",
        )

    def extract(self, image_path: Path, prompt: str | None = None) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        prompt_text = prompt or DEFAULT_PROMPT
        with TemporaryDirectory(prefix="deepseek_ocr_") as temp_dir:
            captured = StringIO()
            with redirect_stdout(captured):
                result = self._run_infer(image_path, prompt_text, temp_dir)
            printed = captured.getvalue().strip()
            raw_output = printed if result is None else str(result)

        kv_pairs = parse_key_values(raw_output)
        return {"raw_output": raw_output, "kv_pairs": kv_pairs}

    def extract_key(self, image_path: Path, key: str) -> dict[str, Any]:
        prompt = f"<image>\n{key.strip()}"
        raw = self._infer_raw(image_path, prompt)
        parsed = parse_grounded_value(raw, key)
        return {
            "raw_output": raw,
            "key": parsed["key"],
            "value": parsed["value"],
            "bbox": parsed["bbox"],
            "ref": parsed["ref"],
            "confidence": parsed["confidence"],
        }

    def _infer_raw(self, image_path: Path, prompt: str) -> str:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        with TemporaryDirectory(prefix="deepseek_ocr_") as temp_dir:
            captured = StringIO()
            with redirect_stdout(captured):
                result = self._run_infer(image_path, prompt, temp_dir)
            printed = captured.getvalue().strip()
            return printed if result is None else str(result)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        status = self.status()
        if not status.ready:
            raise RuntimeError(status.message)

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModel, AutoTokenizer

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            max_memory = self._parse_max_memory()
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": dtype,
                "use_safetensors": True,
                "low_cpu_mem_usage": True,
            }
            if DEVICE_MAP:
                model_kwargs["device_map"] = DEVICE_MAP
            if max_memory:
                model_kwargs["max_memory"] = max_memory

            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(self.base_model_path, **model_kwargs)
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.eval()
            if not DEVICE_MAP:
                model = model.to(device)

            self._tokenizer = tokenizer
            self._model = model
        except Exception as exc:  # pragma: no cover - depends on local ML stack.
            self._load_error = f"Failed to load model: {exc}"
            raise RuntimeError(self._load_error) from exc

    def _run_infer(self, image_path: Path, prompt: str, output_path: str) -> Any:
        infer_target = self._find_infer_target()
        if infer_target is None:
            raise RuntimeError("Loaded model does not expose DeepSeek-OCR infer().")
        return infer_target.infer(
            self._tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=output_path,
            base_size=BASE_SIZE,
            image_size=IMAGE_SIZE,
            crop_mode=CROP_MODE,
            save_results=False,
        )

    def _find_infer_target(self) -> Any | None:
        if hasattr(self._model, "infer"):
            return self._model
        base_model = getattr(self._model, "base_model", None)
        if hasattr(base_model, "infer"):
            return base_model
        inner_model = getattr(base_model, "model", None)
        if hasattr(inner_model, "infer"):
            return inner_model
        return None

    def _missing_weight_files(self) -> list[str]:
        index_path = self.base_model_path / "model.safetensors.index.json"
        if not index_path.exists():
            return []
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ["model.safetensors.index.json is invalid"]
        files = sorted(set(index.get("weight_map", {}).values()))
        return [name for name in files if not (self.base_model_path / name).exists()]

    def _parse_max_memory(self) -> dict[int | str, str] | None:
        if not MAX_MEMORY:
            return None
        result: dict[int | str, str] = {}
        for item in MAX_MEMORY.split(","):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            clean_key = key.strip()
            result[int(clean_key) if clean_key.isdigit() else clean_key] = value.strip()
        return result or None


_extractors: dict[str, DeepseekOcrKvpExtractor] = {}


def get_extractor(adapter_path: Path | str | None = None) -> DeepseekOcrKvpExtractor:
    path = Path(adapter_path) if adapter_path else ADAPTER_PATH
    key = str(path.resolve()) if path.exists() else str(path)
    if key not in _extractors:
        _extractors[key] = DeepseekOcrKvpExtractor(adapter_path=path)
    return _extractors[key]


extractor = get_extractor()
