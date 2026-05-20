from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import json
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DICT_LANG,
    SPH_PQ_PATH, SPH_NPZ_PATH,
    GOSEE_PQ_PATH, GOSEE_NPZ_PATH,
    INC_JSONL_PATH, INC_NPZ_PATH, INC_PQ_PATH,
    WS_NPZ, WS_LBL_PARQ, WS_LBL_JSONL,
    PREC_NPZ, PREC_LBL_PARQ, PREC_LBL_JSONL,
    CP_NPZ_MAIN, CP_NPZ_ALT, CP_LBL_PARQ, CP_LBL_JSONL,
)


# ---------------------------------------------------------------------
# Utilitários de IO
# ---------------------------------------------------------------------
def _coerce_path(p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    if isinstance(p, Path):
        return p
    try:
        return Path(str(p))
    except Exception:
        return None


def _load_parquet(path: Optional[Path]) -> Optional[pd.DataFrame]:
    path = _coerce_path(path)
    if not path:
        return None
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _load_jsonl(path: Optional[Path]) -> Optional[pd.DataFrame]:
    path = _coerce_path(path)
    if not path:
        return None
    if not path.exists():
        return None

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return None
    return pd.DataFrame(rows)


def _load_npz_embeddings_strict(path: Optional[Path]) -> np.ndarray:
    path = _coerce_path(path)
    if not path or not path.exists():
        raise FileNotFoundError(f"Embeddings NPZ não encontrado: {path}")

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"Não foi possível carregar o arquivo de embeddings {path}. "
            "Verifique se os objetos do Git LFS foram baixados corretamente."
        ) from exc

    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr.astype(np.float32, copy=False)

    raise ValueError(f"NPZ inválido (nenhuma matriz 2D encontrada): {path}")


def _load_labels_any(parquet_path: Optional[Path], jsonl_path: Optional[Path]) -> pd.DataFrame:
    try:
        df = _load_parquet(parquet_path)
    except Exception:
        df = None
    if df is not None:
        return df.reset_index(drop=True)
    df = _load_jsonl(jsonl_path)
    if df is not None:
        return df.reset_index(drop=True)
    raise FileNotFoundError(
        f"Rótulos não encontrados (nem Parquet nem JSONL): {parquet_path} / {jsonl_path}"
    )


# ---------------------------------------------------------------------
# Normalização de labels (CRÍTICO para evitar WS “inventado”)
# Garante que exista df['label'] com o texto correto (preferindo PT).
# ---------------------------------------------------------------------
def _pick_first_existing_col(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _normalize_labels_df(df: pd.DataFrame, family: str, lang: str = "pt") -> pd.DataFrame:
    """
    Cria/garante coluna 'label' contendo o termo a ser usado no matching e exibido ao LLM.
    - Prefere PT quando lang="pt"
    - Se já existir 'label', mantém.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Se já tem coluna label, só garante string/strip
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()
        return df

    # Possíveis nomes comuns vindos de parquet/jsonl/xlsx processados
    # (ordem importa! Inclui variações com espaços e parênteses)
    if lang == "pt":
        preferred = [
            "text",  # formato do ws_embeddings_pt (id, text, lang)
            "Termo (PT)", "Termo(PT)", "Termo_PT", "term_pt", "pt", "texto_pt", "descricao_pt", "Descrição (PT)",
            "descricao", "term", "Term", "LABEL", "label",
            "Termo (EN)", "Termo(EN)", "Termo_EN", "term_en", "en", "texto_en", "description",
        ]
    else:
        preferred = [
            "text",  # formato do ws_embeddings_pt
            "Termo (EN)", "Termo(EN)", "Termo_EN", "term_en", "en", "texto_en", "description",
            "term", "Term", "LABEL", "label",
            "Termo (PT)", "Termo(PT)", "Termo_PT", "term_pt", "pt", "texto_pt", "descricao",
        ]

    col = _pick_first_existing_col(df, preferred)
    if col is None:
        # fallback: primeira coluna não-índice
        for c in df.columns:
            if c not in ["index", "Index", "INDEX", "_rowid", "_index", "id", "ID", "lang"]:
                col = c
                break
        if col is None:
            col = df.columns[0]

    df["label"] = df[col].astype(str).str.strip()
    return df


def _align_embeddings_and_labels(E: np.ndarray, L: pd.DataFrame, what: str) -> Tuple[np.ndarray, pd.DataFrame]:
    if E is None or L is None or len(L) == 0:
        return E, L
    m = min(E.shape[0], len(L))
    if E.shape[0] != len(L):
        # corta para o menor para evitar desalinhamento silencioso
        E = E[:m, :]
        L = L.iloc[:m].reset_index(drop=True)
    return E, L


# ---------------------------------------------------------------------
# Loaders principais (cacheados)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sphera() -> Tuple[pd.DataFrame, np.ndarray]:
    df = _load_parquet(SPH_PQ_PATH)
    if df is None:
        raise FileNotFoundError(f"Sphera parquet não encontrado: {SPH_PQ_PATH}")

    E = _load_npz_embeddings_strict(SPH_NPZ_PATH)

    # RowID estável para alinhamento com embeddings
    if "_rowid" not in df.columns:
        df = df.reset_index(drop=True).copy()
        df["_rowid"] = np.arange(len(df), dtype=np.int64)

    # sanity: corta se mismatch
    if E.shape[0] != len(df):
        m = min(E.shape[0], len(df))
        df = df.iloc[:m].reset_index(drop=True)
        df["_rowid"] = np.arange(len(df), dtype=np.int64)
        E = E[:m, :]

    return df, E


@st.cache_data(show_spinner=False)
def load_gosee() -> Tuple[pd.DataFrame, np.ndarray]:
    df = _load_parquet(GOSEE_PQ_PATH)
    if df is None:
        raise FileNotFoundError(f"GoSee parquet não encontrado: {GOSEE_PQ_PATH}")
    E = _load_npz_embeddings_strict(GOSEE_NPZ_PATH)
    m = min(len(df), E.shape[0])
    return df.iloc[:m].reset_index(drop=True), E[:m, :]


@st.cache_data(show_spinner=False)
def load_history() -> Tuple[pd.DataFrame, np.ndarray]:
    # aceita parquet ou jsonl
    df = _load_parquet(INC_PQ_PATH) if INC_PQ_PATH else None
    if df is None:
        df = _load_jsonl(INC_JSONL_PATH)
    if df is None:
        raise FileNotFoundError(f"Histórico não encontrado: {INC_PQ_PATH} / {INC_JSONL_PATH}")

    E = _load_npz_embeddings_strict(INC_NPZ_PATH)
    m = min(len(df), E.shape[0])
    return df.iloc[:m].reset_index(drop=True), E[:m, :]


@st.cache_data(show_spinner=False)
def load_prompts_md(path: Path):
    # compatibilidade: app não usa diretamente, mas mantém para não quebrar imports
    path = _coerce_path(path)
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_datasets_context(path: Path):
    path = _coerce_path(path)
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_dicts():
    """
    Retorna: (E_ws, L_ws, E_prec, L_prec, E_cp, L_cp)
    Nesta aplicação, os NPZ são obrigatórios para WS, Precursores e CP.
    E garante L_* com coluna 'label' (preferindo DICT_LANG).
    """
    # WS
    E_ws = _load_npz_embeddings_strict(WS_NPZ)
    L_ws = _load_labels_any(WS_LBL_PARQ, WS_LBL_JSONL)
    L_ws = _normalize_labels_df(L_ws, family="ws", lang=DICT_LANG)
    E_ws, L_ws = _align_embeddings_and_labels(E_ws, L_ws, "WS")

    # Precursores
    E_prec = _load_npz_embeddings_strict(PREC_NPZ)
    L_prec = _load_labels_any(PREC_LBL_PARQ, PREC_LBL_JSONL)
    L_prec = _normalize_labels_df(L_prec, family="prec", lang=DICT_LANG)
    E_prec, L_prec = _align_embeddings_and_labels(E_prec, L_prec, "Precursores")

    # CP
    cp_npz_path = CP_NPZ_MAIN if isinstance(CP_NPZ_MAIN, Path) else None
    if cp_npz_path is None and isinstance(CP_NPZ_ALT, Path):
        cp_npz_path = CP_NPZ_ALT

    E_cp = _load_npz_embeddings_strict(cp_npz_path)
    L_cp = _load_labels_any(CP_LBL_PARQ, CP_LBL_JSONL)
    L_cp = _normalize_labels_df(L_cp, family="cp", lang=DICT_LANG)
    E_cp, L_cp = _align_embeddings_and_labels(E_cp, L_cp, "CP")

    return E_ws, L_ws, E_prec, L_prec, E_cp, L_cp
