from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    ALLOWED_EXTENSIONS,
    ANNOTATION_EXPORT_DIR,
    CHECKPOINT_NAMES_PATH,
    ROOT_DIR,
    TRAINED_CHECKPOINT_DIR,
    UPLOAD_DIR,
)
from .db import (
    connect,
    init_db,
    row_to_annotation,
    row_to_annotation_set,
    row_to_document,
    row_to_extraction,
    row_to_page,
    utc_now,
)
from .services.files import split_into_pages
from .services.model import extractor, get_extractor


app = FastAPI(title="DeepSeek-OCR KVP Tester")
app.mount("/static", StaticFiles(directory=ROOT_DIR / "app" / "static"), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT_DIR / "app" / "static" / "index.html")


@app.get("/api/model/status")
def model_status() -> dict[str, object]:
    return extractor.status().__dict__


@app.get("/api/checkpoints")
def list_checkpoints() -> list[dict[str, object]]:
    candidates = [ROOT_DIR / "checkpoint-1"]
    candidates.extend(sorted(ROOT_DIR.glob("checkpoint-*")))
    candidates.extend(sorted(TRAINED_CHECKPOINT_DIR.glob("*")))
    seen: set[str] = set()
    checkpoints: list[dict[str, object]] = []
    names = _load_checkpoint_names()
    for path in candidates:
        if not path.exists() or not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        checkpoints.append(
            {
                "name": names.get(resolved, path.name),
                "folder_name": path.name,
                "path": str(path),
                "ready": (path / "adapter_config.json").exists() or path.name.startswith("checkpoint-"),
            }
        )
    return checkpoints


@app.patch("/api/checkpoints/name")
def rename_checkpoint(path: str = Form(...), name: str = Form(...)) -> dict[str, object]:
    checkpoint_path = _resolve_checkpoint_path(path)
    if checkpoint_path is None:
        raise HTTPException(status_code=400, detail="checkpoint path is required")
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="name is required")
    names = _load_checkpoint_names()
    names[str(checkpoint_path.resolve())] = clean_name
    CHECKPOINT_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_NAMES_PATH.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"name": clean_name, "path": str(checkpoint_path)}


@app.get("/api/documents")
def list_documents(q: str | None = Query(default=None)) -> list[dict[str, object]]:
    params: list[object] = []
    where = ""
    if q:
        needle = f"%{q.lower()}%"
        where = """
        WHERE lower(d.filename) LIKE ?
           OR EXISTS (
                SELECT 1 FROM pages p
                LEFT JOIN page_extractions e ON e.page_id = p.id
                WHERE p.document_id = d.id
                  AND (
                    lower(COALESCE(p.raw_output, '')) LIKE ?
                    OR lower(COALESCE(p.kv_pairs_json, '')) LIKE ?
                    OR lower(COALESCE(p.page_title, '')) LIKE ?
                    OR lower(COALESCE(e.query_key, '')) LIKE ?
                    OR lower(COALESCE(e.value_text, '')) LIKE ?
                  )
           )
        """
        params.extend([needle, needle, needle, needle, needle, needle])

    with connect() as db:
        rows = db.execute(
            f"""
            SELECT
                d.*,
                COUNT(p.id) AS page_count,
                SUM(CASE WHEN p.status = 'processed' THEN 1 ELSE 0 END) AS processed_pages
            FROM documents d
            LEFT JOIN pages p ON p.document_id = d.id
            {where}
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """,
            params,
        ).fetchall()
    return [dict(row_to_document(row)) for row in rows]


@app.get("/api/documents/{document_id}")
def get_document(document_id: int) -> dict[str, object]:
    with connect() as db:
        row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return row_to_document(row)


@app.patch("/api/documents/{document_id}")
def update_document(document_id: int, filename: str | None = Form(default=None)) -> dict[str, object]:
    get_document(document_id)
    if filename is None or not filename.strip():
        raise HTTPException(status_code=400, detail="filename is required")
    with connect() as db:
        db.execute(
            "UPDATE documents SET filename = ?, updated_at = ? WHERE id = ?",
            (Path(filename).name, utc_now(), document_id),
        )
        db.commit()
    return get_document(document_id)


@app.get("/api/documents/{document_id}/pages")
def list_pages(document_id: int) -> list[dict[str, object]]:
    get_document(document_id)
    return _list_pages(document_id)


@app.post("/api/documents/{document_id}/pages")
def add_pages(document_id: int, files: list[UploadFile] = File(...)) -> dict[str, object]:
    get_document(document_id)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    existing_pages = _list_pages(document_id)
    start_page = max([int(page["page_number"]) for page in existing_pages], default=0) + 1
    now = utc_now()
    added_size = 0
    with connect() as db:
        next_page = start_page
        for file_index, file in enumerate(files):
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

            upload_path = UPLOAD_DIR / f"{document_id}-added-{int(time.time())}-{file_index}{suffix}"
            with upload_path.open("wb") as target:
                shutil.copyfileobj(file.file, target)
            added_size += upload_path.stat().st_size

            page_paths = split_into_pages(upload_path, document_id, start_page=next_page)
            if upload_path.exists() and upload_path.is_file():
                upload_path.unlink()
            for offset, page_path in enumerate(page_paths):
                db.execute(
                    """
                    INSERT INTO pages
                        (document_id, page_number, page_title, image_path, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'ready', ?, ?)
                    """,
                    (
                        document_id,
                        next_page + offset,
                        _default_page_title(file.filename, next_page + offset, len(page_paths)),
                        str(page_path),
                        now,
                        now,
                    ),
                )
            next_page += len(page_paths)
        db.execute(
            "UPDATE documents SET status = 'uploaded', size_bytes = size_bytes + ?, updated_at = ? WHERE id = ?",
            (added_size, now, document_id),
        )
        db.commit()
    doc = get_document(document_id)
    doc["pages"] = _list_pages(document_id)
    return doc


@app.get("/api/pages/{page_id}")
def get_page(page_id: int) -> dict[str, object]:
    with connect() as db:
        row = db.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Page not found")
    page = row_to_page(row)
    page["extractions"] = _list_extractions(page_id)
    return page


@app.patch("/api/pages/{page_id}")
def update_page(
    page_id: int,
    page_number: int | None = Form(default=None),
    page_title: str | None = Form(default=None),
) -> dict[str, object]:
    page = get_page(page_id)
    if page_number is None and page_title is None:
        raise HTTPException(status_code=400, detail="page_number or page_title is required")
    if page_number is not None and page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be >= 1")
    clean_title = Path(page_title).name.strip() if page_title is not None else None
    if page_title is not None and not clean_title:
        raise HTTPException(status_code=400, detail="page_title cannot be empty")
    with connect() as db:
        db.execute(
            """
            UPDATE pages
            SET page_number = COALESCE(?, page_number),
                page_title = COALESCE(?, page_title),
                updated_at = ?
            WHERE id = ?
            """,
            (page_number, clean_title, utc_now(), page_id),
        )
        db.commit()
    return get_page(page_id)


@app.delete("/api/pages/{page_id}")
def delete_page(page_id: int) -> dict[str, bool]:
    page = get_page(page_id)
    path = Path(str(page["image_path"]))
    with connect() as db:
        db.execute("DELETE FROM pages WHERE id = ?", (page_id,))
        db.commit()
    if path.exists() and path.is_file():
        path.unlink()
    _refresh_document_status(int(page["document_id"]))
    return {"ok": True}


@app.get("/api/pages/{page_id}/extractions")
def list_extractions(page_id: int) -> list[dict[str, object]]:
    get_page(page_id)
    return _list_extractions(page_id)


@app.get("/api/documents/{document_id}/file")
def get_document_file(document_id: int) -> FileResponse:
    doc = get_document(document_id)
    path = Path(str(doc["stored_path"]))
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Stored file not found")
    return FileResponse(path, media_type=str(doc.get("content_type") or "application/octet-stream"))


@app.get("/api/pages/{page_id}/image")
def get_page_image(page_id: int) -> FileResponse:
    page = get_page(page_id)
    path = Path(str(page["image_path"]))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page image not found")
    media_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(path, media_type=media_type)


@app.post("/api/documents")
def create_document_group(group_name: str = Form(...)) -> dict[str, object]:
    clean_name = Path(group_name).name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="group_name is required")
    now = utc_now()
    with connect() as db:
        cursor = db.execute(
            """
            INSERT INTO documents
                (filename, stored_path, content_type, size_bytes, status, created_at, updated_at)
            VALUES (?, '', ?, 0, 'uploaded', ?, ?)
            """,
            (clean_name, "application/x-document-group", now, now),
        )
        document_id = int(cursor.lastrowid)
        db.commit()

    doc = get_document(document_id)
    doc["pages"] = _list_pages(document_id)
    return doc


@app.post("/api/documents/{document_id}/extract")
def extract_document(
    document_id: int,
    key: str = Form(...),
    checkpoint_path: str | None = Form(default=None),
) -> dict[str, object]:
    get_document(document_id)
    pages = _list_pages(document_id)
    if not pages:
        raise HTTPException(status_code=400, detail="Group has no documents")
    if not key.strip():
        raise HTTPException(status_code=400, detail="key is required")

    try:
        for page in pages:
            _extract_page(int(page["id"]), key=key, checkpoint_path=checkpoint_path)
        _refresh_document_status(document_id)
        doc = get_document(document_id)
        doc["pages"] = _list_pages(document_id)
        return doc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/extract")
def extract_documents(
    key: str = Form(...),
    document_ids: list[int] | None = Form(default=None),
    all_documents: bool = Form(default=False),
    checkpoint_path: str | None = Form(default=None),
) -> dict[str, object]:
    if not key.strip():
        raise HTTPException(status_code=400, detail="key is required")

    ids = _document_ids_for_bulk_extract(document_ids or [], all_documents=all_documents)
    if not ids:
        raise HTTPException(status_code=400, detail="No groups selected")

    processed_documents: list[dict[str, object]] = []
    failed_documents: list[dict[str, object]] = []
    processed_pages = 0
    started_at = time.time()

    for document_id in ids:
        try:
            document = extract_document(document_id, key=key, checkpoint_path=checkpoint_path)
            processed_documents.append(document)
            processed_pages += len(document.get("pages", []))
        except Exception as exc:
            failed_documents.append({"id": document_id, "error": str(exc)})

    return {
        "key": key,
        "document_count": len(ids),
        "processed_document_count": len(processed_documents),
        "failed_document_count": len(failed_documents),
        "processed_page_count": processed_pages,
        "duration_seconds": round(time.time() - started_at, 3),
        "documents": processed_documents,
        "failed_documents": failed_documents,
    }


@app.post("/api/pages/{page_id}/extract")
def extract_page(
    page_id: int,
    key: str = Form(...),
    checkpoint_path: str | None = Form(default=None),
) -> dict[str, object]:
    if not key.strip():
        raise HTTPException(status_code=400, detail="key is required")
    try:
        return _extract_page(page_id, key=key, checkpoint_path=checkpoint_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.patch("/api/extractions/{extraction_id}")
def update_extraction(
    extraction_id: int,
    query_key: str | None = Form(default=None),
    value_text: str | None = Form(default=None),
    bbox_json: str | None = Form(default=None),
) -> dict[str, object]:
    extraction = _get_extraction(extraction_id)
    bbox_value = json.dumps(json.loads(bbox_json), ensure_ascii=False) if bbox_json else None
    with connect() as db:
        db.execute(
            """
            UPDATE page_extractions
            SET query_key = COALESCE(?, query_key),
                value_text = COALESCE(?, value_text),
                bbox_json = COALESCE(?, bbox_json),
                updated_at = ?
            WHERE id = ?
            """,
            (query_key, value_text, bbox_value, utc_now(), extraction_id),
        )
        db.commit()
    _sync_page_extractions(int(extraction["page_id"]))
    return _get_extraction(extraction_id)


@app.delete("/api/extractions/{extraction_id}")
def delete_extraction(extraction_id: int) -> dict[str, bool]:
    extraction = _get_extraction(extraction_id)
    with connect() as db:
        db.execute("DELETE FROM page_extractions WHERE id = ?", (extraction_id,))
        db.commit()
    _sync_page_extractions(int(extraction["page_id"]))
    return {"ok": True}


@app.get("/api/annotation-sets")
def list_annotation_sets() -> list[dict[str, object]]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT s.*, COUNT(a.id) AS annotation_count
            FROM annotation_sets s
            LEFT JOIN annotations a ON a.annotation_set_id = s.id
            GROUP BY s.id
            ORDER BY s.created_at DESC
            """
        ).fetchall()
    return [row_to_annotation_set(row) for row in rows]


@app.post("/api/annotation-sets")
def create_annotation_set(name: str = Form(...), description: str | None = Form(default=None)) -> dict[str, object]:
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="name is required")
    now = utc_now()
    with connect() as db:
        cursor = db.execute(
            """
            INSERT INTO annotation_sets
                (name, description, status, created_at, updated_at)
            VALUES (?, ?, 'draft', ?, ?)
            """,
            (clean_name, description, now, now),
        )
        set_id = int(cursor.lastrowid)
        db.commit()
    return get_annotation_set(set_id)


@app.get("/api/annotation-sets/{set_id}")
def get_annotation_set(set_id: int) -> dict[str, object]:
    with connect() as db:
        row = db.execute("SELECT * FROM annotation_sets WHERE id = ?", (set_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Annotation set not found")
    item = row_to_annotation_set(row)
    item["annotations"] = _list_annotations(set_id)
    return item


@app.delete("/api/annotation-sets/{set_id}")
def delete_annotation_set(set_id: int) -> dict[str, bool]:
    get_annotation_set(set_id)
    with connect() as db:
        db.execute("DELETE FROM annotation_sets WHERE id = ?", (set_id,))
        db.commit()
    return {"ok": True}


@app.post("/api/annotations")
def create_annotation(
    annotation_set_id: int = Form(...),
    page_id: int = Form(...),
    query_key: str = Form(...),
    value_text: str = Form(...),
    bbox_json: str = Form(...),
) -> dict[str, object]:
    get_annotation_set(annotation_set_id)
    get_page(page_id)
    key = query_key.strip()
    value = value_text.strip()
    if not key or not value:
        raise HTTPException(status_code=400, detail="query_key and value_text are required")
    bbox = _parse_bbox_json(bbox_json)
    now = utc_now()
    with connect() as db:
        cursor = db.execute(
            """
            INSERT INTO annotations
                (annotation_set_id, page_id, query_key, value_text, bbox_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (annotation_set_id, page_id, key, value, json.dumps(bbox, ensure_ascii=False), now, now),
        )
        db.execute("UPDATE annotation_sets SET updated_at = ? WHERE id = ?", (now, annotation_set_id))
        annotation_id = int(cursor.lastrowid)
        db.commit()
    return _get_annotation(annotation_id)


@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: int) -> dict[str, bool]:
    annotation = _get_annotation(annotation_id)
    with connect() as db:
        db.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        db.execute("UPDATE annotation_sets SET updated_at = ? WHERE id = ?", (utc_now(), annotation["annotation_set_id"]))
        db.commit()
    return {"ok": True}


@app.post("/api/annotation-sets/{set_id}/train")
def train_annotation_set(
    set_id: int,
    background_tasks: BackgroundTasks,
    steps: int = Form(default=100),
) -> dict[str, object]:
    if steps < 1:
        raise HTTPException(status_code=400, detail="steps must be >= 1")
    annotation_set = get_annotation_set(set_id)
    annotations = annotation_set.get("annotations", [])
    if not annotations:
        raise HTTPException(status_code=400, detail="Annotation set is empty")
    export_path = _export_annotation_set(set_id)
    output_dir = TRAINED_CHECKPOINT_DIR / f"annotation-set-{set_id}"
    now = utc_now()
    with connect() as db:
        db.execute(
            """
            UPDATE annotation_sets
            SET status = 'training', export_path = ?, checkpoint_path = ?, training_log = ?, updated_at = ?
            WHERE id = ?
            """,
            (str(export_path), str(output_dir), f"Training process was queued. Steps: {steps}.", now, set_id),
        )
        db.commit()
    background_tasks.add_task(_run_training_job, set_id, export_path, output_dir, steps)
    return get_annotation_set(set_id)


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int) -> dict[str, bool]:
    doc = get_document(document_id)
    pages = _list_pages(document_id)
    path = Path(str(doc["stored_path"]))
    with connect() as db:
        db.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
        db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        db.commit()
    if path.exists() and path.is_file():
        path.unlink()
    for upload_path in UPLOAD_DIR.glob(f"{document_id}-added-*"):
        if upload_path.is_file():
            upload_path.unlink()
    for page in pages:
        page_path = Path(str(page["image_path"]))
        if page_path.exists() and page_path.is_file():
            page_path.unlink()
    return {"ok": True}


def _list_pages(document_id: int) -> list[dict[str, object]]:
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number",
            (document_id,),
        ).fetchall()
    pages = [row_to_page(row) for row in rows]
    for page in pages:
        page["extractions"] = _list_extractions(int(page["id"]))
    return pages


def _default_page_title(filename: str | None, page_number: int, page_count: int) -> str:
    base_name = Path(filename or "").name.strip()
    stem = Path(base_name).stem if base_name else "Page"
    if page_count > 1:
        return f"{stem} - document {page_number}"
    return stem or f"Document {page_number}"


def _document_ids_for_bulk_extract(document_ids: list[int], *, all_documents: bool) -> list[int]:
    with connect() as db:
        if all_documents:
            rows = db.execute("SELECT id FROM documents ORDER BY created_at DESC").fetchall()
            return [int(row["id"]) for row in rows]

        unique_ids = list(dict.fromkeys(int(item) for item in document_ids))
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        rows = db.execute(
            f"SELECT id FROM documents WHERE id IN ({placeholders}) ORDER BY created_at DESC",
            unique_ids,
        ).fetchall()
    return [int(row["id"]) for row in rows]


def _resolve_checkpoint_path(checkpoint_path: str | None) -> Path | None:
    if not checkpoint_path or not checkpoint_path.strip():
        return None
    path = Path(checkpoint_path.strip())
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail="Checkpoint path does not exist")
    return path


def _load_checkpoint_names() -> dict[str, str]:
    if not CHECKPOINT_NAMES_PATH.exists():
        return {}
    try:
        data = json.loads(CHECKPOINT_NAMES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if str(value).strip()}


def _parse_bbox_json(value: str) -> list[list[int]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="bbox_json is invalid") from exc
    box = parsed[0] if parsed and isinstance(parsed[0], list) else parsed
    if not isinstance(box, list) or len(box) != 4:
        raise HTTPException(status_code=400, detail="bbox must contain four coordinates")
    coords = [max(0, min(999, int(round(float(item))))) for item in box]
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=400, detail="bbox has invalid size")
    return [[x1, y1, x2, y2]]


def _list_annotations(set_id: int) -> list[dict[str, object]]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT
                a.*,
                p.page_title,
                p.page_number,
                p.image_path,
                p.document_id,
                d.filename AS group_name
            FROM annotations a
            JOIN pages p ON p.id = a.page_id
            JOIN documents d ON d.id = p.document_id
            WHERE a.annotation_set_id = ?
            ORDER BY a.created_at DESC, a.id DESC
            """,
            (set_id,),
        ).fetchall()
    return [row_to_annotation(row) for row in rows]


def _get_annotation(annotation_id: int) -> dict[str, object]:
    with connect() as db:
        row = db.execute(
            """
            SELECT
                a.*,
                p.page_title,
                p.page_number,
                p.image_path,
                p.document_id,
                d.filename AS group_name
            FROM annotations a
            JOIN pages p ON p.id = a.page_id
            JOIN documents d ON d.id = p.document_id
            WHERE a.id = ?
            """,
            (annotation_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return row_to_annotation(row)


def _export_annotation_set(set_id: int) -> Path:
    annotations = _list_annotations(set_id)
    if not annotations:
        raise HTTPException(status_code=400, detail="Annotation set is empty")
    export_path = ANNOTATION_EXPORT_DIR / f"annotation_set_{set_id}.jsonl"
    with export_path.open("w", encoding="utf-8") as file:
        for item in annotations:
            bbox_str = json.dumps(item["bbox"], separators=(",", ":"), ensure_ascii=False)
            record = {
                "messages": [
                    {
                        "role": "<|User|>",
                        "content": f"<image>\n{item['query_key']}",
                        "images": [str(item["image_path"])],
                    },
                    {
                        "role": "<|Assistant|>",
                        "content": f"<|ref|>text<|/ref|><|det|>{bbox_str}<|/det|>{item['value_text']}",
                    },
                ],
                "meta": {
                    "annotation_id": item["id"],
                    "annotation_set_id": set_id,
                    "group_name": item.get("group_name"),
                    "document_title": item.get("page_title"),
                },
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return export_path


def _run_training_job(set_id: int, export_path: Path, output_dir: Path, steps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "train_annotations_lora.py"),
        "--dataset",
        str(export_path),
        "--output-dir",
        str(output_dir),
        "--steps",
        str(steps),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            text=True,
            capture_output=True,
            timeout=60 * 60 * 24,
        )
        status = "trained" if result.returncode == 0 and (output_dir / "adapter_config.json").exists() else "failed"
        log = (result.stdout + "\n" + result.stderr).strip() or f"Command finished with code {result.returncode}"
    except Exception as exc:  # pragma: no cover - depends on local ML stack.
        status = "failed"
        log = str(exc)
    with connect() as db:
        db.execute(
            """
            UPDATE annotation_sets
            SET status = ?, training_log = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, log[-12000:], utc_now(), set_id),
        )
        db.commit()


def _extract_page(page_id: int, key: str, checkpoint_path: str | None = None) -> dict[str, object]:
    page = get_page(page_id)
    page_path = Path(str(page["image_path"]))
    document_id = int(page["document_id"])
    with connect() as db:
        db.execute(
            "UPDATE pages SET status = 'processing', error = NULL, updated_at = ? WHERE id = ?",
            (utc_now(), page_id),
        )
        db.execute(
            "UPDATE documents SET status = 'processing', error = NULL, updated_at = ? WHERE id = ?",
            (utc_now(), document_id),
        )
        db.commit()

    try:
        active_extractor = get_extractor(_resolve_checkpoint_path(checkpoint_path))
        result = active_extractor.extract_key(page_path, key=key)
        query_key = str(result.get("key") or key).strip()
        now = utc_now()
        with connect() as db:
            db.execute(
                """
                UPDATE pages
                SET status = 'processed', raw_output = ?, error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (result["raw_output"], now, page_id),
            )
            existing_rows = db.execute(
                """
                SELECT id FROM page_extractions
                WHERE page_id = ? AND lower(trim(query_key)) = lower(trim(?))
                ORDER BY updated_at DESC, id DESC
                """,
                (page_id, query_key),
            ).fetchall()
            bbox_json = json.dumps(result["bbox"], ensure_ascii=False) if result["bbox"] is not None else None
            if existing_rows:
                keep_id = int(existing_rows[0]["id"])
                db.execute(
                    """
                    UPDATE page_extractions
                    SET query_key = ?, value_text = ?, bbox_json = ?, ref_text = ?,
                        confidence = ?, raw_output = ?, status = 'processed',
                        error = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        query_key,
                        result["value"],
                        bbox_json,
                        result["ref"],
                        result["confidence"],
                        result["raw_output"],
                        now,
                        keep_id,
                    ),
                )
                duplicate_ids = [int(row["id"]) for row in existing_rows[1:]]
                if duplicate_ids:
                    db.executemany("DELETE FROM page_extractions WHERE id = ?", [(item_id,) for item_id in duplicate_ids])
            else:
                db.execute(
                    """
                    INSERT INTO page_extractions
                        (page_id, query_key, value_text, bbox_json, ref_text, confidence, raw_output, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'processed', ?, ?)
                    """,
                    (
                        page_id,
                        query_key,
                        result["value"],
                        bbox_json,
                        result["ref"],
                        result["confidence"],
                        result["raw_output"],
                        now,
                        now,
                    ),
                )
            db.commit()
        _sync_page_extractions(page_id)
        _refresh_document_status(document_id)
        return get_page(page_id)
    except Exception as exc:
        with connect() as db:
            db.execute(
                "UPDATE pages SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
                (str(exc), utc_now(), page_id),
            )
            db.execute(
                "UPDATE documents SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
                (str(exc), utc_now(), document_id),
            )
            db.commit()
        raise


def _list_extractions(page_id: int) -> list[dict[str, object]]:
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM page_extractions WHERE page_id = ? ORDER BY created_at DESC, id DESC",
            (page_id,),
        ).fetchall()
    return [row_to_extraction(row) for row in rows]


def _get_extraction(extraction_id: int) -> dict[str, object]:
    with connect() as db:
        row = db.execute("SELECT * FROM page_extractions WHERE id = ?", (extraction_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Extraction not found")
    return row_to_extraction(row)


def _sync_page_extractions(page_id: int) -> None:
    extractions = _list_extractions(page_id)
    kv_pairs = [
        {
            "key": item["query_key"],
            "value": item["value_text"],
            "bbox": item["bbox"],
            "confidence": item["confidence"],
        }
        for item in extractions
        if item.get("status") == "processed"
    ]
    with connect() as db:
        db.execute(
            "UPDATE pages SET kv_pairs_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(kv_pairs, ensure_ascii=False), utc_now(), page_id),
        )
        db.commit()


def _refresh_document_status(document_id: int) -> None:
    pages = _list_pages(document_id)
    statuses = {str(page["status"]) for page in pages}
    if "failed" in statuses:
        status = "failed"
    elif statuses and statuses <= {"processed"}:
        status = "processed"
    elif "processing" in statuses:
        status = "processing"
    else:
        status = "uploaded"
    with connect() as db:
        db.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
            (status, utc_now(), document_id),
        )
        db.commit()
