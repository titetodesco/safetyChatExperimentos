# app.py
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

import config as cfg

from core.data_loader import (
    load_sphera,
    load_datasets_context,
    load_prompts_md,
    load_dicts,
)
from core.analytics_manifest import get_refresh_status

from core.sphera import filter_sphera, get_sphera_location_col, topk_similar
from core.context_builder import (
    hits_dataframe,
    build_dic_matches_md,
    build_sphera_context_md,
)
from core.dictionaries import aggregate_dict_matches_over_hits

from services.upload_extract import extract_any
from services.llm_client import chat

# ---------------- Senha de proteção ----------------
PASSWORD = "cdshell"  # Troque por uma senha forte

def check_password():
    """Exibe um campo de senha e retorna True se a senha estiver correta."""
    st.sidebar.header("🔒 Área protegida")
    password = st.sidebar.text_input("Digite a senha para acessar o app:", type="password")
    if password == PASSWORD:
        return True
    elif password:
        st.sidebar.error("Senha incorreta. Tente novamente.")
        return False
    else:
        return False

# ---------------- Helpers ----------------
def _ensure_eventid_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "EventID" in df.columns:
        return df

    candidates = [
        "EVENTID", "EVENT_ID", "Event ID", "EVENT ID", "ID", "Id", "id",
        "EventId", "eventid", "event_id",
    ]
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: "EventID"})

    df = df.copy()
    if "_rowid" in df.columns:
        df["EventID"] = df["_rowid"].apply(lambda x: f"ROW_{x}")
    else:
        df["EventID"] = [f"ROW_{i}" for i in range(len(df))]
    return df


def _safe_event_ids_from_hits(hits) -> list[str]:
    ids: list[str] = []
    for evid, _, _ in hits:
        s = str(evid).strip() if evid is not None else ""
        if s and s not in ids:
            ids.append(s)
    return ids


def _canonical_event_id(value: Any) -> str:
    s = str(value).strip() if value is not None else ""
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s.casefold()


def _parse_event_ids(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    parts = re.split(r"[\n,;]+", raw)
    if len(parts) == 1:
        parts = re.split(r"\s+", raw)

    ids: list[str] = []
    seen: set[str] = set()
    for part in parts:
        event_id = part.strip()
        key = _canonical_event_id(event_id)
        if event_id and key not in seen:
            ids.append(event_id)
            seen.add(key)
    return ids


def _row_text(row: Any, columns: list[str], default: str = "") -> str:
    for col in columns:
        try:
            if hasattr(row, "get"):
                value = row.get(col, "")
            else:
                value = ""
        except Exception:
            value = ""
        text = str(value).strip() if value is not None else ""
        if text and text.lower() != "nan":
            return text
    return default


def _description_from_row(row: Any) -> str:
    return _row_text(
        row,
        [
            "Description", "DESCRIPTION", "DescriÃ§Ã£o", "DESCRIÃ‡ÃƒO",
            "Observation", "OBSERVATION", "Resumo", "Summary",
        ],
    )


def _lookup_sphera_events_by_ids(
    df: pd.DataFrame | None,
    event_ids: list[str],
) -> tuple[list[tuple[str, pd.Series]], list[str]]:
    if not event_ids:
        return [], []
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "EventID" not in df.columns:
        return [], event_ids

    lookup: dict[str, int] = {}
    for idx, value in df["EventID"].items():
        key = _canonical_event_id(value)
        if key and key not in lookup:
            lookup[key] = int(idx)

    found: list[tuple[str, pd.Series]] = []
    missing: list[str] = []
    for event_id in event_ids:
        idx = lookup.get(_canonical_event_id(event_id))
        if idx is None:
            missing.append(event_id)
            continue
        row = df.loc[idx]
        found.append((str(row.get("EventID", event_id)).strip(), row))

    return found, missing


def _safe_event_ids_from_event_rows(events: list[tuple[str, Any]]) -> list[str]:
    ids: list[str] = []
    for evid, _ in events or []:
        s = str(evid).strip() if evid is not None else ""
        if s and s not in ids:
            ids.append(s)
    return ids


def _unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _canonical_event_id(value)
        if value and key not in seen:
            out.append(value)
            seen.add(key)
    return out


def _build_event_rows_dataframe(events: list[tuple[str, Any]], loc_col: str | None) -> pd.DataFrame:
    rows = []
    for evid, row in events or []:
        loc = _row_text(row, [loc_col], "N/D") if loc_col else "N/D"
        rows.append(
            {
                "EventID": evid,
                "Event Type": _row_text(
                    row,
                    [
                        "Event Type", "EVENT TYPE", "EventType", "EVENTTYPE",
                        "Tipo Evento", "TIPO EVENTO", "Tipo de Evento", "TYPE",
                    ],
                    "N/D",
                ),
                "LOCATION": loc,
                "Description": _description_from_row(row),
            }
        )
    return pd.DataFrame(rows)


def _build_input_events_context_md(events: list[tuple[str, Any]], loc_col: str | None) -> str:
    if not events:
        return ""

    lines = [
        "=== EVENTOS_INFORMADOS_PELO_USUARIO ===",
        "EventID\tEvent Type\tLOCATION\tDescription",
    ]
    for evid, row in events:
        loc = _row_text(row, [loc_col], "N/D") if loc_col else "N/D"
        event_type = _row_text(
            row,
            [
                "Event Type", "EVENT TYPE", "EventType", "EVENTTYPE",
                "Tipo Evento", "TIPO EVENTO", "Tipo de Evento", "TYPE",
            ],
            "N/D",
        )
        desc = _description_from_row(row).replace("\n", " ").strip()
        lines.append(f"{evid}\t{event_type}\t{loc}\t{desc}")
    return "\n".join(lines) + "\n"


def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[conteudo truncado para manter a conversa dentro do limite do modelo]"


def _build_hits_md_table(hits, loc_col: str | None) -> str:
    if not hits:
        return ""

    def _pick_event_type(row) -> str:
        candidates = [
            "Event Type", "EVENT TYPE", "EventType", "EVENTTYPE",
            "Tipo Evento", "TIPO EVENTO", "Tipo de Evento", "TIPO DE EVENTO",
            "Type", "TYPE", "Classification", "Classificação",
        ]
        for c in candidates:
            if hasattr(row, "get"):
                v = str(row.get(c, "")).strip()
                if v:
                    return v
        return "N/D"

    lines = [
        "| EventID | Event Type | Similaridade (cos) | LOCATION | Description |",
        "|---|---|---:|---|---|",
    ]
    for evid, sim, row in hits:
        loc = str(row.get(loc_col, "N/D")).strip() if (loc_col and hasattr(row, "get")) else "N/D"
        event_type = _pick_event_type(row)
        desc = str(row.get("Description", "")).replace("\n", " ").strip() if hasattr(row, "get") else ""
        lines.append(
            f"| {str(evid).strip()} | {event_type} | {float(sim):.3f} | {loc} | {desc[:220]} |"
        )
    return "\n".join(lines)


def _build_terms_md_table(title: str, terms: list[tuple]) -> str:
    lines = [f"### {title}", "| Termo | Score agregado |", "|---|---:|"]
    if not terms:
        lines.append("| (nenhum) | 0.000 |")
        return "\n".join(lines)
    for label, score in terms:
        lines.append(f"| {str(label).strip()} | {float(score):.3f} |")
    return "\n".join(lines)


def _build_prompt3_deterministic_reply(
    query_text: str,
    hits,
    ws_matches,
    prec_matches,
    cp_matches,
    loc_col: str | None,
    thr_sph: float,
) -> str:
    q = (query_text or "").strip() or "(texto não informado)"
    lines = [
        "## Análise determinística de Weak Signals",
        f"Consulta: {q}",
        f"Eventos Sphera recuperados com similaridade ≥ {float(thr_sph):.2f}: {len(hits)}",
        "",
        "### Eventos recuperados (Top-K)",
        _build_hits_md_table(hits, loc_col) if hits else "(nenhum evento recuperado)",
        "",
        _build_terms_md_table("Weak Signals encontrados", ws_matches or []),
        "",
        _build_terms_md_table("Precursores encontrados", prec_matches or []),
        "",
        _build_terms_md_table("Condicionantes de Performance (CP) encontrados", cp_matches or []),
        "",
    ]

    if not ws_matches:
        lines.append("Observação: nenhum WS acima dos critérios atuais (limiar/suporte/agregação).")
    else:
        lines.append("Observação: os WS acima já estão restritos aos eventos recuperados do Sphera.")

    return "\n".join(lines)


def _sanitize_model_reply(reply: str) -> str:
    reply = str(reply or "").strip() or "(sem conteudo)"
    internal_markers = (
        "REGRAS OBRIGAT",
        "EVENTOS_OBRIGATORIOS",
        "WS_MATCHES:",
        "PRECURSORES_MATCHES:",
        "CP_MATCHES:",
    )
    reply_lines = []
    for line in reply.splitlines():
        upper = line.strip().upper()
        if any(upper.startswith(marker) for marker in internal_markers):
            continue
        reply_lines.append(line)
    reply = "\n".join(reply_lines).strip() or "(sem conteudo)"

    if re.search(r"\bws\s*(id|code)\b|\bws\s*\d+\b", reply, flags=re.IGNORECASE):
        reply = re.sub(r"\bWS\s*(ID|Code)\b", "WS", reply, flags=re.IGNORECASE)
        reply = re.sub(r"\bWS\s*\d+\b", "WS", reply, flags=re.IGNORECASE)

    reply = re.sub(r"Contribui\S*\s+Principal", "Condicionantes de Performance", reply, flags=re.IGNORECASE)
    reply = re.sub(
        r"Fatores\s+de\s+Contribui\S*\s+Principal\s*\(\s*CP\s*\)",
        "Condicionantes de Performance (CP)",
        reply,
        flags=re.IGNORECASE,
    )
    return reply


def clear_chat():
    st.session_state["chat"] = []
    st.session_state["draft_prompt"] = ""
    st.session_state["analysis_text"] = ""
    st.session_state["event_id_text"] = ""
    st.session_state["upld_texts"] = []
    st.session_state["last_upload_fingerprint"] = None
    for k in [
        "messages", "history", "chat_messages", "last_reply", "last_ctx", "last_hits",
        "initial_analysis", "analysis_context", "analysis_guardrails", "analysis_cp_glossary",
        "analysis_required_events", "analysis_match_blocks", "analysis_context_summary",
        "followup_messages",
    ]:
        if k in st.session_state:
            del st.session_state[k]


# ---------------- Page ----------------
st.set_page_config(page_title="SAFETY • CHAT", layout="wide")

# Verificação de senha
if not check_password():
    st.stop()  # Interrompe o app até digitar a senha correta

st.title("SAFETY • CHAT")

ss = st.session_state
ss.setdefault("draft_prompt", "")
ss.setdefault("analysis_text", "")
ss.setdefault("event_id_text", "")
ss.setdefault("upld_texts", [])
ss.setdefault("chat", [])
ss.setdefault("last_upload_fingerprint", None)
ss.setdefault("initial_analysis", "")
ss.setdefault("analysis_context", "")
ss.setdefault("analysis_guardrails", "")
ss.setdefault("analysis_cp_glossary", "")
ss.setdefault("analysis_required_events", "")
ss.setdefault("analysis_match_blocks", "")
ss.setdefault("analysis_context_summary", {})
ss.setdefault("followup_messages", [])

analytics_status = get_refresh_status(ROOT_DIR)
if analytics_status.get("stale"):
    st.warning(
        "Fontes em data/docs ou data/xlsx mudaram, ou o manifesto ainda nao existe. "
        "O Safety Chat continua usando os artefatos atuais em data/analytics."
    )

# carregamentos silenciosos
_ = load_datasets_context(cfg.DATASETS_CONTEXT_PATH)
prompts_md = load_prompts_md(cfg.PROMPTS_MD_PATH)

# Parse prompts.md para extrair títulos e corpos
def parse_prompts(md_text: str) -> list[dict]:
    """Extrai prompts do markdown (### título seguido de corpo)"""
    prompts = []
    lines = md_text.split('\n')
    current_title = None
    current_body = []

    for line in lines:
        if line.startswith('### '):
            # Salva prompt anterior
            if current_title and current_body:
                prompts.append({
                    'title': current_title,
                    'body': '\n'.join(current_body).strip()
                })
            # Novo prompt
            current_title = line[4:].strip()
            current_body = []
        elif line.startswith('## ') or line.startswith('# '):
            # Ignora cabeçalhos de seção
            continue
        elif current_title is not None:
            # Adiciona linha ao corpo do prompt atual
            current_body.append(line)
    
    # Salva último prompt
    if current_title and current_body:
        prompts.append({
            'title': current_title,
            'body': '\n'.join(current_body).strip()
        })

    return prompts

prompts_list = parse_prompts(prompts_md)

# Debug: verificar se prompts_list foi populado
if not prompts_list:
    st.error(f"⚠️ Nenhum prompt encontrado no arquivo. Verificar data/prompts/prompts.md")

df_sph, E_sph = load_sphera()
df_sph = _ensure_eventid_column(df_sph)

# ---------------- Sidebar ----------------
with st.sidebar:
    with st.expander("Status dos dados", expanded=bool(analytics_status.get("stale"))):
        if not analytics_status.get("manifest_exists"):
            st.warning("Manifesto de analytics nao encontrado.")
        elif analytics_status.get("stale"):
            st.warning("Fontes alteradas desde a ultima geracao.")
        else:
            st.success("Artefatos sincronizados com as fontes.")

        if analytics_status.get("manifest_generated_at"):
            st.caption(f"Ultima geracao: {analytics_status['manifest_generated_at']}")
        if analytics_status.get("manifest_embedding_model"):
            st.caption(f"Modelo: {analytics_status['manifest_embedding_model']}")
        st.caption(f"Fontes monitoradas: {analytics_status.get('source_count', 0)}")

        diff = analytics_status.get("diff", {}) or {}
        for label, key in [("Novos", "added"), ("Alterados", "changed"), ("Removidos", "removed")]:
            items = diff.get(key, []) or []
            if items:
                st.markdown(f"**{label}:**")
                for item in items[:8]:
                    st.caption(f"- {item}")
                if len(items) > 8:
                    st.caption(f"... mais {len(items) - 8}")

        missing = analytics_status.get("missing_artifacts", []) or []
        if missing:
            st.markdown("**Artefatos ausentes:**")
            for item in missing[:8]:
                st.caption(f"- {item}")

        st.caption("Atualize com: python tools/build_analytics.py --build")

    st.markdown("---")
    st.subheader("Assistente de Prompts")
    
    prompt_titles = [p['title'] for p in prompts_list]
    sel_prompt = st.selectbox(
        "Selecione um modelo", 
        options=["(vazio)"] + prompt_titles, 
        index=0,
        key="sb_prompt_sel"
    )

    if st.button("Carregar no rascunho", use_container_width=True):
        if sel_prompt != "(vazio)":
            body = next((p['body'] for p in prompts_list if p['title'] == sel_prompt), "")
            if body:
                st.session_state.draft_prompt = body
                st.success(f"✅ '{sel_prompt}' carregado!")
                st.rerun()
        else:
            st.warning("Selecione um prompt primeiro.")

    st.markdown("---")
    st.header("Recuperação – Sphera")
    k_sph = st.slider("Top-K Sphera", 5, 300, 50, step=5, key="sb_topk_sph")
    thr_sph = st.slider("Limiar Sphera (cos)", 0.0, 1.0, 0.45, 0.01, key="sb_thr_sph")
    years = st.slider("Últimos N anos", 0, 10, 5, 1, key="sb_years")

    st.header("Filtros avançados – Sphera")
    substr = st.text_input("Description contém (substring)", value="", key="sb_substr")

    loc_col_detected = get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None
    loc_opts = (
        sorted(df_sph[loc_col_detected].dropna().unique().tolist())
        if (isinstance(df_sph, pd.DataFrame) and loc_col_detected in df_sph.columns)
        else []
    )
    locations = st.multiselect("Location", options=loc_opts, default=[], key="sb_locations")

    st.header("Agregação sobre eventos recuperados (Sphera)")
    agg_mode = st.selectbox("Agregação", options=["max", "mean"], index=1, key="sb_agg_mode")
    per_event_thr = st.slider("Limiar por evento (dicionários)", 0.0, 1.0, 0.40, 0.01, key="sb_per_event_thr")
    support_min = st.slider("Suporte mínimo (nº eventos)", 1, 10, 3, 1, key="sb_support_min")

    st.markdown("---")
    thr_ws = st.slider("Limiar WS", 0.0, 1.0, 0.40, 0.01, key="sb_thr_ws")
    thr_prec = st.slider("Limiar Precursores", 0.0, 1.0, 0.40, 0.01, key="sb_thr_prec")
    thr_cp = st.slider("Limiar CP", 0.0, 1.0, 0.40, 0.01, key="sb_thr_cp")

    top_ws = st.slider("Top-N WS", 1, 100, 25, 1, key="sb_top_ws")
    top_prec = st.slider("Top-N Precursores", 1, 100, 25, 1, key="sb_top_prec")
    top_cp = st.slider("Top-N CP", 1, 100, 25, 1, key="sb_top_cp")

    st.markdown("---")
    if not cfg.OLLAMA_API_KEY:
        st.error("⚠️ OLLAMA_API_KEY não configurada. O chat do modelo não irá responder.")

# ---------------- Main ----------------
st.subheader("Conteúdo do prompt")
draft = st.text_area(
    "Prompt", 
    key="draft_prompt", 
    height=220, 
    label_visibility="collapsed",
    placeholder="Digite aqui suas instruções para o modelo (ex: 'Analise os eventos', 'Liste WS', etc.)..."
)

st.subheader("EventID(s) do Sphera (opcional)")
event_id_text = st.text_area(
    "EventIDs",
    key="event_id_text",
    height=80,
    label_visibility="collapsed",
    placeholder="Informe um ou mais EventIDs separados por virgula, ponto e virgula ou quebra de linha.",
)

st.subheader("Texto de analise complementar")
analysis = st.text_area(
    "Análise", 
    key="analysis_text", 
    height=220, 
    label_visibility="collapsed",
    placeholder="Digite aqui a descrição do incidente/cenário para buscar eventos similares no Sphera..."
)

st.subheader("Anexar arquivo (opcional)")
upl = st.file_uploader(
    "Upload",
    type=["txt", "md", "pdf", "docx"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)
if upl is not None:
    uploaded_text = extract_any(upl)
    if uploaded_text.strip():
        current_fingerprint = f"{upl.name}:{getattr(upl, 'size', 0)}"
        if ss.get("last_upload_fingerprint") != current_fingerprint:
            ss.upld_texts.append(uploaded_text)
            ss["last_upload_fingerprint"] = current_fingerprint
            st.success(f"✅ Upload recebido: {upl.name} ({len(uploaded_text)} caracteres)")
        else:
            st.info(f"ℹ️ Arquivo já anexado: {upl.name}")
    else:
        st.warning(
            f"⚠️ Não foi possível extrair texto de {upl.name}. "
            "Possíveis causas: PDF escaneado/imagem, arquivo protegido ou formato não suportado. "
         )

c1, c2 = st.columns([1, 1])
with c1:
    go_btn = st.button("Gerar analise inicial", type="primary")
with c2:
    st.button("Limpar chat e campos", on_click=clear_chat)

# ---------------- Run ----------------
if go_btn:
    progress_box = st.empty()
    progress_box.info("⏳ Processando consulta (na primeira execução pode demorar para carregar embeddings/modelo)...")

    # ✅ Retrieval usa texto de análise + uploads (não usa o prompt como fallback)
    selected_event_ids = _parse_event_ids(event_id_text)
    input_events, missing_event_ids = _lookup_sphera_events_by_ids(df_sph, selected_event_ids)
    input_event_descriptions = [_description_from_row(row) for _, row in input_events]

    if missing_event_ids:
        st.warning("EventID(s) nao encontrado(s) no Sphera local: " + ", ".join(missing_event_ids))

    if input_events:
        st.subheader("Evento(s) informado(s) como contexto")
        st.dataframe(
            _build_event_rows_dataframe(input_events, get_sphera_location_col(df_sph)),
            use_container_width=True,
            hide_index=True,
        )

    retrieval_base = analysis.strip() if isinstance(analysis, str) else ""
    retrieval_parts = input_event_descriptions + [retrieval_base] + (ss.upld_texts or [])
    query_for_retrieval = "\n\n".join([p for p in retrieval_parts if p]).strip()

    # input do chat pode ter tudo (prompt + análise + uploads)
    input_events_block = _build_input_events_context_md(
        input_events,
        get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None,
    )
    user_parts = [draft, input_events_block, analysis] + (ss.upld_texts or [])
    user_input = "\n\n".join([p for p in user_parts if p]).strip()

    # sempre inicializa (evita NameError)
    hits = []
    dic_res = {"WS": [], "Precursores": [], "CP": []}
    debug_raw = {"RAW_WS": [], "RAW_PREC": [], "RAW_CP": []}
    ws_matches, prec_matches, cp_matches = [], [], []

    # 1) filtra df
    loc_col = get_sphera_location_col(df_sph) if isinstance(df_sph, pd.DataFrame) else None
    df_base = filter_sphera(df_sph, locations, substr, years)

    # 2) topk
    if not query_for_retrieval:
        st.info(
            "Texto de análise vazio: não foi possível buscar eventos semelhantes. "
            "Informe o contexto do evento e, se necessário, ajuste o limiar/Top-K."
        )
    elif isinstance(df_base, pd.DataFrame) and not df_base.empty and E_sph is not None:
        if "_rowid" not in df_base.columns:
            st.error("Sphera sem coluna _rowid. Verifique load_sphera() em core/data_loader.py.")
        else:
            rowids = df_base["_rowid"].to_numpy()
            E_base = E_sph[rowids]
            df_base2 = df_base.reset_index(drop=True)

            hits = topk_similar(
                query_for_retrieval,
                df_base2,
                E_base,
                topk=int(k_sph),
                min_sim=float(thr_sph),
            )

    st.subheader(f"Eventos do Sphera (Top-{min(int(k_sph), len(hits))})")
    if hits:
        df_hits = hits_dataframe(hits, loc_col)
        if "EventID" not in df_hits.columns:
            df_hits.insert(0, "EventID", [h[0] for h in hits])
        st.dataframe(df_hits, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum evento recuperado. Ajuste texto/limiar/Top-K.")

    # 3) dicionários (✅ agora com os 6 argumentos obrigatórios)
    if hits:
        E_ws, L_ws, E_prec, L_prec, E_cp, L_cp = load_dicts()

        dic_res, debug_raw = aggregate_dict_matches_over_hits(
            hits,
            E_ws, L_ws,
            E_prec, L_prec,
            E_cp, L_cp,
            per_event_thr=float(per_event_thr),
            support_min=int(support_min),
            agg_mode=str(agg_mode),
            thr_ws=float(thr_ws),
            thr_prec=float(thr_prec),
            thr_cp=float(thr_cp),
            top_ws=int(top_ws),
            top_prec=int(top_prec),
            top_cp=int(top_cp),
        )

        ws_matches = dic_res.get("WS", []) if isinstance(dic_res, dict) else []
        prec_matches = dic_res.get("Precursores", []) if isinstance(dic_res, dict) else []
        cp_matches = dic_res.get("CP", []) if isinstance(dic_res, dict) else []

    st.subheader("Diagnóstico da recuperação")
    cdx1, cdx2, cdx3, cdx4 = st.columns(4)
    cdx1.metric("Eventos recuperados", len(hits))
    cdx2.metric("WS finais", len(ws_matches))
    cdx3.metric("Precursores finais", len(prec_matches))
    cdx4.metric("CP finais", len(cp_matches))

    with st.expander("Ver detalhes do diagnóstico"):
        st.markdown(
            f"- Parâmetros: `Top-K={int(k_sph)}`, `limiar_sphera={float(thr_sph):.2f}`, "
            f"`per_event_thr={float(per_event_thr):.2f}`, `support_min={int(support_min)}`, `agg_mode={str(agg_mode)}`"
        )
        st.markdown(
            f"- Limiar dicionários: `WS={float(thr_ws):.2f}`, `Precursores={float(thr_prec):.2f}`, `CP={float(thr_cp):.2f}`"
        )

        raw_ws = debug_raw.get("RAW_WS", []) if isinstance(debug_raw, dict) else []
        raw_prec = debug_raw.get("RAW_PREC", []) if isinstance(debug_raw, dict) else []
        raw_cp = debug_raw.get("RAW_CP", []) if isinstance(debug_raw, dict) else []

        st.markdown(
            f"- Candidatos brutos (antes de suporte+limiar final): "
            f"`WS={len(raw_ws)}`, `Precursores={len(raw_prec)}`, `CP={len(raw_cp)}`"
        )

        if raw_ws:
            st.markdown("**Top WS brutos (debug)**")
            st.dataframe(
                pd.DataFrame(raw_ws, columns=["WS", "score_bruto_max"]).head(10),
                use_container_width=True,
                hide_index=True,
            )
        if raw_prec:
            st.markdown("**Top Precursores brutos (debug)**")
            st.dataframe(
                pd.DataFrame(raw_prec, columns=["Precursor", "score_bruto_max"]).head(10),
                use_container_width=True,
                hide_index=True,
            )
        if raw_cp:
            st.markdown("**Top CP brutos (debug)**")
            st.dataframe(
                pd.DataFrame(raw_cp, columns=["CP", "score_bruto_max"]).head(10),
                use_container_width=True,
                hide_index=True,
            )

        if not ws_matches and raw_ws:
            st.info("WS tiveram candidatos brutos, mas foram filtrados por limiar/suporte/agregação.")
        if not prec_matches and raw_prec:
            st.info("Precursores tiveram candidatos brutos, mas foram filtrados por limiar/suporte/agregação.")
        if not cp_matches and raw_cp:
            st.info("CP tiveram candidatos brutos, mas foram filtrados por limiar/suporte/agregação.")

    # Weak Signals: ocultado da saída conforme solicitado

    # 4) contexto e guardrails
    allowed_event_ids = _unique_preserve_order(
        _safe_event_ids_from_event_rows(input_events) + _safe_event_ids_from_hits(hits)
    )

    ctx_full = "\n".join([
        input_events_block,
        build_sphera_context_md(hits, loc_col),
        build_dic_matches_md(dic_res),
    ])

    ws_list = [str(t[0]).strip() for t in ws_matches]
    prec_list = [str(t[0]).strip() for t in prec_matches]
    cp_list = [str(t[0]).strip() for t in cp_matches]

    ws_block = "WS_MATCHES:\n" + (
        "\n".join([f"- {t}" for t in ws_list]) if ws_list else "- (nenhum)\n"
    )

    prec_block = "PRECURSORES_MATCHES:\n" + (
        "\n".join([f"- {t}" for t in prec_list]) if prec_list else "- (nenhum)\n"
    )
    cp_block = "CP_MATCHES:\n" + (
        "\n".join([f"- {t}" for t in cp_list]) if cp_list else "- (nenhum)\n"
    )

    guardrails = (
        "REGRAS OBRIGATÓRIAS:\n"
        f"1) Considere TODOS os eventos Sphera recuperados (limiar de recuperação: ≥ {thr_sph:.2f}). "
        "NÃO crie ou aplique novos limiares arbitrários (como 0,60 ou outros valores).\n"
        "2) NÃO invente WS/Precursores/CP. Use APENAS os termos listados em nos dicionários *_MATCHES.\n"
        "3) NÃO use 'WS ID', 'WS code', 'WS1/WS2' ou numeração. O dicionário não tem IDs.\n"
        "4) Ao citar eventos, use APENAS EventIDs desta lista (não invente): "
        f"{', '.join(allowed_event_ids) if allowed_event_ids else '(nenhum)'}\n"
        "5) 'Event Type' e 'Tipo de Evento' são equivalentes; use os valores do Sphera sem traduzir livremente.\n"
        "6) Se não houver termos acima do limiar, diga explicitamente que não encontrou.\n"
        "7) WS/Precursores/CP devem ser APENAS dos blocos *_MATCHES recebidos; não invente termos por interpretação livre.\n"
        "8) Se WS_MATCHES estiver '(nenhum)', NÃO liste Weak Signals; apenas diga que não houve WS acima do limiar.\n"
        "9) CP significa EXCLUSIVAMENTE 'Condicionantes de Performance'. NUNCA expanda CP como 'Contribuição Principal'.\n"
        "10) NÃO recrie tabela/lista de eventos no texto final: a tabela oficial de eventos já é exibida pelo app com EventID/Event Type corretos.\n"
        "11) No texto final, foque em síntese, padrões, recomendações e lições aprendidas com base nos eventos recuperados.\n"
        "12) NÃO repita seções com o mesmo conteúdo; se não houver distinção, explique que é a mesma base.\n"
        "13) Use português claro e correto; não use rótulos confusos ou palavras sem sentido.\n"
        "14) NÃO crie seção 'Histórico' separada. A única fonte de dados é Sphera.\n"
        "15) NÃO aplique novo corte de similaridade (ex.: 0.59, 0.60). Use somente o limiar já aplicado na recuperação.\n"
    )

    cp_glossary = (
        "GLOSSÁRIO DE TERMOS:\n"
        "- CP = Condicionantes de Performance (taxonomia CP).\n"
        "- É proibido usar: 'Contribuição Principal' para CP.\n"
    )

    required_events_block = (
        "EVENTOS_OBRIGATORIOS (deve citar todos):\n"
        + ("\n".join([f"- {eid}" for eid in allowed_event_ids]) if allowed_event_ids else "- (nenhum)")
    )

    prompt3_mode = isinstance(sel_prompt, str) and (
        "weak signals" in sel_prompt.lower() or "sinais fracos" in sel_prompt.lower()
    )

    if prompt3_mode:
        reply = _build_prompt3_deterministic_reply(
            query_text=query_for_retrieval,
            hits=hits,
            ws_matches=ws_matches,
            prec_matches=prec_matches,
            cp_matches=cp_matches,
            loc_col=loc_col,
            thr_sph=float(thr_sph),
        )
    else:
        messages = [
            {"role": "system", "content": "Você é o SAFETY • CHAT. Seja preciso e não alucine."},
            {"role": "system", "content": cp_glossary},
            {"role": "system", "content": guardrails},
            {"role": "system", "content": required_events_block},
            {"role": "system", "content": ws_block + "\n\n" + prec_block + "\n\n" + cp_block},
            {"role": "system", "content": "CONTEXTO (eventos recuperados do Sphera):\n" + ctx_full},
            {"role": "user", "content": user_input},
        ]

        try:
            if not cfg.OLLAMA_API_KEY:
                reply = (
                    "Falha ao consultar o modelo: OLLAMA_API_KEY não configurada. "
                    "Defina a variável de ambiente/secrets e tente novamente."
                )
            else:
                res = chat(messages, stream=False, timeout=int(cfg.OLLAMA_TIMEOUT))
                reply = res.get("message", {}).get("content", "(sem conteúdo)")
        except Exception as e:
            reply = f"Falha ao consultar o modelo: {e}"

    if not prompt3_mode:
        # higieniza vazamento de blocos internos/guardrails na resposta visível
        leaked_prefixes = (
            "REGRAS OBRIGATÓRIAS:",
            "EVENTOS_OBRIGATORIOS",
            "WS_MATCHES:",
            "PRECURSORES_MATCHES:",
            "CP_MATCHES:",
        )
        reply_lines = [ln for ln in str(reply).splitlines() if not ln.strip().startswith(leaked_prefixes)]
        reply = "\n".join(reply_lines).strip() or "(sem conteúdo)"

        # normaliza menções inválidas de códigos WS sem bloquear a resposta inteira
        has_ws_code = bool(re.search(r"\bws\s*(id|code)\b|\bws\s*\d+\b", reply, flags=re.IGNORECASE))
        if has_ws_code:
            reply = re.sub(r"\bWS\s*(ID|Code)\b", "WS", reply, flags=re.IGNORECASE)
            reply = re.sub(r"\bWS\s*\d+\b", "WS", reply, flags=re.IGNORECASE)

        # normaliza expansão incorreta de CP
        reply = re.sub(r"Contribui[cç][aã]o\s+Principal", "Condicionantes de Performance", reply, flags=re.IGNORECASE)
        reply = re.sub(r"Fatores\s+de\s+Contribui[cç][aã]o\s+Principal\s*\(\s*CP\s*\)", "Condicionantes de Performance (CP)", reply, flags=re.IGNORECASE)

    progress_box.success("✅ Processamento concluído.")

    ss.initial_analysis = reply
    ss.analysis_context = ctx_full
    ss.analysis_guardrails = guardrails
    ss.analysis_cp_glossary = cp_glossary
    ss.analysis_required_events = required_events_block
    ss.analysis_match_blocks = ws_block + "\n\n" + prec_block + "\n\n" + cp_block
    ss.analysis_context_summary = {
        "event_ids_informados": _safe_event_ids_from_event_rows(input_events),
        "event_ids_recuperados": _safe_event_ids_from_hits(hits),
        "ws": ws_list,
        "precursores": prec_list,
        "cp": cp_list,
    }
    ss.followup_messages = []
    ss.chat = [{"role": "assistant", "content": reply}]

# ---------------- Follow-up chat ----------------
followup_question = None
if ss.get("initial_analysis"):
    followup_question = st.chat_input(
        "Pergunte algo sobre a analise, peca aprofundamento ou acrescente novas informacoes..."
    )

if followup_question:
    ss.chat.append({"role": "user", "content": followup_question})

    followup_system = (
        "Voce esta em modo de conversa investigativa multi-turno do SAFETY CHAT. "
        "A analise inicial ja foi gerada. Responda apenas a nova pergunta do usuario, "
        "usando a analise inicial, o contexto recuperado e os eventos Sphera autorizados. "
        "Nao repita a estrutura completa da analise inicial, a menos que o usuario peca explicitamente. "
        "Se o usuario trouxer novas informacoes, integre-as como complemento e deixe claro quando uma nova "
        "rodada de recuperacao for recomendada."
    )
    messages = [
        {"role": "system", "content": "Voce e o SAFETY CHAT. Seja preciso, rastreavel e nao alucine."},
        {"role": "system", "content": followup_system},
        {"role": "system", "content": ss.get("analysis_cp_glossary", "")},
        {"role": "system", "content": ss.get("analysis_guardrails", "")},
        {"role": "system", "content": ss.get("analysis_required_events", "")},
        {"role": "system", "content": ss.get("analysis_match_blocks", "")},
        {
            "role": "system",
            "content": "ANALISE_INICIAL:\n" + _truncate_text(ss.get("initial_analysis", ""), 12000),
        },
        {
            "role": "system",
            "content": "CONTEXTO_RAG_DA_ANALISE_INICIAL:\n" + _truncate_text(ss.get("analysis_context", ""), 16000),
        },
    ]
    messages.extend(ss.get("followup_messages", [])[-8:])
    messages.append({"role": "user", "content": followup_question})

    try:
        if not cfg.OLLAMA_API_KEY:
            followup_reply = (
                "Falha ao consultar o modelo: OLLAMA_API_KEY nao configurada. "
                "Defina a variavel de ambiente/secrets e tente novamente."
            )
        else:
            with st.spinner("Respondendo ao follow-up..."):
                res = chat(messages, stream=False, timeout=int(cfg.OLLAMA_TIMEOUT))
            followup_reply = res.get("message", {}).get("content", "(sem conteudo)")
    except Exception as e:
        followup_reply = f"Falha ao consultar o modelo: {e}"

    followup_reply = _sanitize_model_reply(followup_reply)
    ss.followup_messages.extend(
        [
            {"role": "user", "content": followup_question},
            {"role": "assistant", "content": followup_reply},
        ]
    )
    ss.chat.append({"role": "assistant", "content": followup_reply})

# ---------------- History ----------------
if ss.get("chat"):
    st.divider()
    st.subheader("Histórico")
    for m in ss.chat[-12:]:
        role = m.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(m.get("content", ""))
