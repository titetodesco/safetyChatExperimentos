from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.analytics_manifest import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    build_manifest,
    get_refresh_status,
    inspect_runtime_artifacts,
    validate_runtime_artifacts,
    write_manifest,
)


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("_x000D_", "\n").strip()
    if text.lower() == "nan":
        return ""
    return text


def _first_present(row: pd.Series, columns: Iterable[str]) -> str:
    for col in columns:
        if col in row.index:
            text = _clean_text(row.get(col))
            if text:
                return text
    return ""


def _read_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo fonte não encontrado: {path}")
    df = pd.read_excel(path)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _encode_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    encoder = _load_encoder(model_name)
    vecs = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    matrix = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def _save_embeddings(path: Path, embeddings: np.ndarray, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"embeddings": embeddings.astype(np.float32, copy=False)}
    for key, values in metadata.items():
        payload[key] = np.asarray(values)
    np.savez_compressed(path, **payload)


def build_sphera(repo_root: Path, model_name: str) -> dict[str, int]:
    src = repo_root / "data" / "xlsx" / "TRATADO_safeguardOffShore.xlsx"
    out_parquet = repo_root / "data" / "analytics" / "sphera.parquet"
    out_npz = repo_root / "data" / "analytics" / "sphera_embeddings.npz"

    df = _read_excel(src)
    if "Event ID" in df.columns:
        df = df.drop_duplicates(subset=["Event ID"], keep="first")
    elif "EventID" in df.columns:
        df = df.drop_duplicates(subset=["EventID"], keep="first")
    df = df.reset_index(drop=True)

    texts = [
        _first_present(row, ["Description", "DESCRIPTION", "Observation", "OBSERVATION", "Title", "TITLE"])
        for _, row in df.iterrows()
    ]
    ids = [
        _first_present(row, ["Event ID", "EventID", "EVENTID", "ID", "Id", "id"]) or str(i)
        for i, (_, row) in enumerate(df.iterrows())
    ]

    df.to_parquet(out_parquet, index=False)
    embeddings = _encode_texts(texts, model_name)
    _save_embeddings(out_npz, embeddings, ids=np.asarray(ids, dtype=str), texts=np.asarray(texts, dtype=str))
    return {"rows": len(df), "dimensions": int(embeddings.shape[1])}


def build_weak_signals(repo_root: Path, model_name: str) -> dict[str, int]:
    src = repo_root / "data" / "xlsx" / "DicionarioWeakSignals.xlsx"
    df_src = _read_excel(src)
    text_col = "Termo (PT)" if "Termo (PT)" in df_src.columns else df_src.columns[-1]
    labels = pd.DataFrame(
        {
            "id": range(len(df_src)),
            "text": df_src[text_col].map(_clean_text),
            "lang": "pt",
        }
    )
    labels = labels[labels["text"].astype(bool)].reset_index(drop=True)
    labels["id"] = range(len(labels))

    out_dir = repo_root / "data" / "analytics"
    labels.to_parquet(out_dir / "ws_embeddings_pt.parquet", index=False)
    _write_jsonl(labels, out_dir / "ws_embeddings_pt.jsonl")

    texts = labels["text"].tolist()
    embeddings = _encode_texts(texts, model_name)
    _save_embeddings(
        out_dir / "ws_embeddings_pt.npz",
        embeddings,
        ids=labels["id"].to_numpy(dtype=np.int32),
        texts=np.asarray(texts, dtype=str),
        meta=np.asarray(f"model={model_name}; family=weak_signals; lang=pt"),
    )
    return {"rows": len(labels), "dimensions": int(embeddings.shape[1])}


def build_precursors(repo_root: Path, model_name: str) -> dict[str, int]:
    src = repo_root / "data" / "xlsx" / "precursores_expandido.xlsx"
    df_src = _read_excel(src)
    text_col = "Precursor_PT" if "Precursor_PT" in df_src.columns else df_src.columns[-1]
    cat_col = "Categoria" if "Categoria" in df_src.columns else None

    rows = []
    for i, row in df_src.iterrows():
        text = _clean_text(row.get(text_col))
        if not text:
            continue
        hto = _clean_text(row.get(cat_col)) if cat_col else ""
        rows.append(
            {
                "id": len(rows),
                "text": text,
                "lang": "pt",
                "hto": hto,
                "label": f"{hto} — {text}" if hto else text,
            }
        )
    labels = pd.DataFrame(rows)

    out_dir = repo_root / "data" / "analytics"
    labels.to_parquet(out_dir / "prec_embeddings_pt.parquet", index=False)
    _write_jsonl(labels, out_dir / "prec_embeddings_pt.jsonl")

    texts = labels["text"].tolist()
    embeddings = _encode_texts(texts, model_name)
    _save_embeddings(
        out_dir / "prec_embeddings_pt.npz",
        embeddings,
        ids=labels["id"].to_numpy(dtype=np.int32),
        texts=np.asarray(texts, dtype=str),
        hto=labels["hto"].to_numpy(dtype=str),
        labels=labels["text"].to_numpy(dtype=str),
        meta=np.asarray(f"model={model_name}; family=precursors; lang=pt"),
    )
    return {"rows": len(labels), "dimensions": int(embeddings.shape[1])}


def build_cp(repo_root: Path, model_name: str) -> dict[str, int]:
    src = repo_root / "data" / "xlsx" / "TaxonomiaCP_Por.xlsx"
    df_src = _read_excel(src)
    rows = []
    texts = []

    for _, row in df_src.iterrows():
        dimensao = _clean_text(row.get("Dimensão"))
        fator = _clean_text(row.get("Fatores"))
        sub1 = _clean_text(row.get("Subfator 1"))
        sub2 = _clean_text(row.get("Subfator 2"))
        bag_pt = _clean_text(row.get("Bag de termos"))
        bag_en = _clean_text(row.get("Bag of terms"))
        label = " / ".join([p for p in [dimensao, fator, sub1, sub2] if p])
        if not label:
            continue
        text = " | ".join([p for p in [label, dimensao, fator, sub1, sub2, bag_pt, bag_en] if p])
        rows.append(
            {
                "label": label,
                "dimensao": dimensao,
                "fator": fator,
                "sub1": sub1,
                "sub2": sub2,
                "bag_pt": bag_pt,
                "bag_en": bag_en,
            }
        )
        texts.append(text)

    labels = pd.DataFrame(rows)
    out_dir = repo_root / "data" / "analytics"
    labels.to_parquet(out_dir / "cp_labels.parquet", index=False)
    _write_jsonl(labels, out_dir / "cp_labels.jsonl")

    embeddings = _encode_texts(texts, model_name)
    _save_embeddings(
        out_dir / "cp_embeddings.npz",
        embeddings,
        ids=np.arange(len(labels), dtype=np.int32),
        texts=np.asarray(texts, dtype=str),
    )
    return {"rows": len(labels), "dimensions": int(embeddings.shape[1])}


BUILDERS = {
    "sphera": build_sphera,
    "ws": build_weak_signals,
    "precursors": build_precursors,
    "cp": build_cp,
}


def _selected_families(raw: str) -> list[str]:
    if raw == "all":
        return list(BUILDERS)
    families = [p.strip() for p in raw.split(",") if p.strip()]
    unknown = [p for p in families if p not in BUILDERS]
    if unknown:
        raise SystemExit(f"Familia(s) desconhecida(s): {', '.join(unknown)}")
    return families


def _print_status(repo_root: Path) -> None:
    status = get_refresh_status(repo_root)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate Safety Chat analytics artifacts.")
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--model", default=os.getenv("ST_MODEL_NAME", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--families", default="all", help="all or comma list: sphera,ws,precursors,cp")
    parser.add_argument("--check", action="store_true", help="Only print source/manifest status.")
    parser.add_argument("--validate", action="store_true", help="Validate current runtime artifacts.")
    parser.add_argument("--strict", action="store_true", help="With --check, return non-zero when stale.")
    parser.add_argument("--manifest-only", action="store_true", help="Write manifest for current sources/artifacts.")
    parser.add_argument("--build", action="store_true", help="Rebuild selected runtime artifacts, then write manifest.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()

    if args.check:
        status = get_refresh_status(repo_root)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2 if args.strict and status.get("stale") else 0

    if args.validate:
        artifacts = inspect_runtime_artifacts(repo_root)
        validation = validate_runtime_artifacts(artifacts)
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 1 if validation["errors"] else 0

    if args.manifest_only:
        artifacts = inspect_runtime_artifacts(repo_root)
        manifest = build_manifest(repo_root, artifacts=artifacts, mode="manifest-only", embedding_model=args.model)
        manifest["validation"] = validate_runtime_artifacts(artifacts)
        path = write_manifest(repo_root, manifest)
        print(f"Manifesto atualizado: {path}")
        _print_status(repo_root)
        return 0

    if not args.build:
        parser.error("Use --check, --manifest-only ou --build.")

    summaries = {}
    for family in _selected_families(args.families):
        print(f"Gerando familia: {family}")
        summaries[family] = BUILDERS[family](repo_root, args.model)

    artifacts = inspect_runtime_artifacts(repo_root)
    validation = validate_runtime_artifacts(artifacts)
    if validation["errors"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 1

    manifest = build_manifest(
        repo_root,
        artifacts=artifacts,
        mode="build",
        embedding_model=args.model,
    )
    manifest["build_summary"] = summaries
    manifest["validation"] = validation
    path = write_manifest(repo_root, manifest)
    print(f"Manifesto atualizado: {path}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
