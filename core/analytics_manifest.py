from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
PIPELINE_VERSION = "0.1.1"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SOURCE_DIRS = ("data/xlsx", "data/docs")
SOURCE_EXTENSIONS = (".xlsx", ".pdf", ".docx", ".md", ".txt")

RUNTIME_ARTIFACTS = (
    "data/analytics/sphera.parquet",
    "data/analytics/sphera_embeddings.npz",
    "data/analytics/ws_embeddings_pt.parquet",
    "data/analytics/ws_embeddings_pt.jsonl",
    "data/analytics/ws_embeddings_pt.npz",
    "data/analytics/prec_embeddings_pt.parquet",
    "data/analytics/prec_embeddings_pt.jsonl",
    "data/analytics/prec_embeddings_pt.npz",
    "data/analytics/cp_labels.parquet",
    "data/analytics/cp_labels.jsonl",
    "data/analytics/cp_embeddings.npz",
)


def manifest_path(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "analytics" / "manifest.json"


def _rel(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _sha256(path: Path) -> tuple[str, str]:
    with path.open("rb") as f:
        prefix = f.read(128)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(data).hexdigest(), "git-lfs-pointer-normalized"

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest(), "raw"


def _file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    stat = path.stat()
    sha256, hash_strategy = _sha256(path)
    return {
        "path": _rel(path, repo_root),
        "sha256": sha256,
        "hash_strategy": hash_strategy,
        "size": int(stat.st_size),
    }


def iter_source_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for rel_dir in SOURCE_DIRS:
        root = Path(repo_root) / rel_dir
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
                files.append(path)
    return sorted(files, key=lambda p: _rel(p, Path(repo_root)).casefold())


def scan_sources(repo_root: Path) -> dict[str, dict[str, Any]]:
    repo_root = Path(repo_root)
    return {
        rec["path"]: rec
        for rec in (_file_record(path, repo_root) for path in iter_source_files(repo_root))
    }


def load_manifest(repo_root: Path) -> dict[str, Any] | None:
    path = manifest_path(Path(repo_root))
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compare_sources(
    manifest: dict[str, Any] | None,
    current_sources: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    previous = {}
    if manifest:
        previous = manifest.get("sources", {}) or {}

    added = sorted([p for p in current_sources if p not in previous])
    removed = sorted([p for p in previous if p not in current_sources])
    changed = sorted(
        [
            p
            for p in current_sources
            if p in previous
            and current_sources[p].get("sha256") != previous[p].get("sha256")
        ]
    )
    return {"added": added, "changed": changed, "removed": removed}


def inspect_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    matrix = None
    matrix_key = None
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            matrix = arr
            matrix_key = key
            break
    if matrix is None:
        return {"readable": True, "error": "nenhuma matriz 2D encontrada"}

    norms = np.linalg.norm(matrix.astype("float32"), axis=1)
    return {
        "readable": True,
        "matrix_key": matrix_key,
        "rows": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "normalized": bool(matrix.size > 0 and float(abs(norms.mean() - 1.0)) < 1e-3),
    }


def inspect_jsonl(path: Path) -> dict[str, Any]:
    rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line)
                rows += 1
    return {"readable": True, "rows": rows}


def inspect_parquet(path: Path) -> dict[str, Any]:
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        return {"readable": True, "rows": int(len(df)), "columns": list(df.columns)}
    except Exception as exc:
        try:
            with path.open("rb") as f:
                first = f.read(4)
                f.seek(max(path.stat().st_size - 4, 0))
                last = f.read(4)
            if first == b"PAR1" and last == b"PAR1":
                return {
                    "readable": True,
                    "format": "parquet",
                    "rows": None,
                    "note": "Parquet valido; contagem de linhas nao inspecionada neste ambiente.",
                }
        except Exception:
            pass
        return {"readable": False, "error": str(exc).splitlines()[0]}


def inspect_artifact(path: Path, repo_root: Path) -> dict[str, Any]:
    rec = _file_record(path, repo_root)
    try:
        if path.suffix.lower() == ".npz":
            rec.update(inspect_npz(path))
        elif path.suffix.lower() == ".jsonl":
            rec.update(inspect_jsonl(path))
        elif path.suffix.lower() == ".parquet":
            rec.update(inspect_parquet(path))
        else:
            rec["readable"] = True
    except Exception as exc:
        rec.update({"readable": False, "error": str(exc)})
    return rec


def inspect_runtime_artifacts(repo_root: Path) -> dict[str, dict[str, Any]]:
    repo_root = Path(repo_root)
    artifacts: dict[str, dict[str, Any]] = {}
    for rel_path in RUNTIME_ARTIFACTS:
        path = repo_root / rel_path
        if not path.exists():
            artifacts[rel_path] = {"path": rel_path, "exists": False}
            continue
        rec = inspect_artifact(path, repo_root)
        rec["exists"] = True
        artifacts[rel_path] = rec
    return artifacts


def validate_runtime_artifacts(artifacts: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for rel_path in RUNTIME_ARTIFACTS:
        rec = artifacts.get(rel_path, {})
        if not rec.get("exists"):
            errors.append(f"Artefato ausente: {rel_path}")
            continue
        if not rec.get("readable"):
            errors.append(f"Artefato nao legivel: {rel_path} ({rec.get('error', 'erro desconhecido')})")

    def rows(rel_path: str) -> int | None:
        value = artifacts.get(rel_path, {}).get("rows")
        return int(value) if isinstance(value, int) else None

    def require_npz(rel_path: str) -> tuple[int | None, int | None]:
        rec = artifacts.get(rel_path, {})
        r = rows(rel_path)
        d = rec.get("dimensions")
        if r is None:
            errors.append(f"NPZ sem contagem de linhas: {rel_path}")
        if not isinstance(d, int) or d <= 0:
            errors.append(f"NPZ sem dimensao valida: {rel_path}")
            d = None
        if rec.get("normalized") is False:
            warnings.append(f"Embeddings possivelmente nao normalizados: {rel_path}")
        return r, d if isinstance(d, int) else None

    sph_rows, dim = require_npz("data/analytics/sphera_embeddings.npz")
    ws_rows, ws_dim = require_npz("data/analytics/ws_embeddings_pt.npz")
    prec_rows, prec_dim = require_npz("data/analytics/prec_embeddings_pt.npz")
    cp_rows, cp_dim = require_npz("data/analytics/cp_embeddings.npz")

    dims = [d for d in [dim, ws_dim, prec_dim, cp_dim] if d is not None]
    if dims and len(set(dims)) > 1:
        errors.append(f"Dimensoes de embeddings divergentes: {dims}")

    pair_checks = [
        ("WS", ws_rows, rows("data/analytics/ws_embeddings_pt.jsonl"), rows("data/analytics/ws_embeddings_pt.parquet")),
        (
            "Precursores",
            prec_rows,
            rows("data/analytics/prec_embeddings_pt.jsonl"),
            rows("data/analytics/prec_embeddings_pt.parquet"),
        ),
        ("CP", cp_rows, rows("data/analytics/cp_labels.jsonl"), rows("data/analytics/cp_labels.parquet")),
        ("Sphera", sph_rows, None, rows("data/analytics/sphera.parquet")),
    ]
    for label, npz_rows, jsonl_rows, parquet_rows in pair_checks:
        expected = jsonl_rows if jsonl_rows is not None else parquet_rows
        if npz_rows is not None and expected is not None and npz_rows != expected:
            errors.append(f"{label}: linhas desalinhadas entre labels/parquet e NPZ ({expected} vs {npz_rows})")
        if expected is None and label != "Sphera":
            warnings.append(f"{label}: nao foi possivel confirmar contagem de linhas do Parquet; JSONL sera usado como fallback.")

    return {"errors": errors, "warnings": warnings}


def build_manifest(
    repo_root: Path,
    *,
    artifacts: dict[str, dict[str, Any]] | None = None,
    generated_by: str = "tools/build_analytics.py",
    mode: str = "manifest",
    embedding_model: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    return {
        "manifest_version": MANIFEST_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": generated_by,
        "mode": mode,
        "embedding_model": embedding_model
        or os.getenv("ST_MODEL_NAME", DEFAULT_EMBEDDING_MODEL),
        "source_dirs": list(SOURCE_DIRS),
        "source_extensions": list(SOURCE_EXTENSIONS),
        "sources": scan_sources(repo_root),
        "artifacts": artifacts if artifacts is not None else inspect_runtime_artifacts(repo_root),
    }


def write_manifest(repo_root: Path, manifest: dict[str, Any]) -> Path:
    path = manifest_path(Path(repo_root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def get_refresh_status(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    manifest = load_manifest(repo_root)
    sources = scan_sources(repo_root)
    diff = compare_sources(manifest, sources)
    missing_artifacts = []
    if manifest:
        for rel_path in RUNTIME_ARTIFACTS:
            if not (repo_root / rel_path).exists():
                missing_artifacts.append(rel_path)

    stale = manifest is None or bool(diff["added"] or diff["changed"] or diff["removed"] or missing_artifacts)
    return {
        "manifest_exists": manifest is not None,
        "stale": stale,
        "diff": diff,
        "missing_artifacts": missing_artifacts,
        "source_count": len(sources),
        "manifest_generated_at": (manifest or {}).get("generated_at"),
        "manifest_embedding_model": (manifest or {}).get("embedding_model"),
        "manifest_path": manifest_path(repo_root).as_posix(),
    }
