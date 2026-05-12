from __future__ import annotations

from pathlib import Path

from ..config import PAGE_DIR


def split_into_pages(path: Path, document_id: int, start_page: int = 1) -> list[Path]:
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        return render_pdf_pages(path, document_id, start_page=start_page)

    output = PAGE_DIR / f"{document_id}-page-{start_page}{_image_suffix(path)}"
    if not output.exists() or output.stat().st_mtime < path.stat().st_mtime:
        output.write_bytes(path.read_bytes())
    return [output]


def render_pdf_pages(path: Path, document_id: int, start_page: int = 1) -> list[Path]:
    import fitz

    pages: list[Path] = []
    with fitz.open(path) as pdf:
        for offset, page in enumerate(pdf):
            index = start_page + offset
            output = PAGE_DIR / f"{document_id}-page-{index}.png"
            if not output.exists() or output.stat().st_mtime < path.stat().st_mtime:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                pixmap.save(output)
            pages.append(output)
    return pages


def _image_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"} else ".png"
