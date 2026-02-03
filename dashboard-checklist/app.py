# app.py
# Streamlit dashboard: CSV Transações -> Checklist + Cards
# + CSV BaaS (opcional) -> bloqueio cautelar por conta
#
# Requisitos esperados:
# - CSV Transações: col0=data, col2=person("NNNN - EMPRESA"), col3=credit, col4=debit
# - CSV BaaS: colunas (ou variações): Posição, Conta, Agência, Nome, Plano, Saldo, Saldo Bloqueado, BaaS
#
# Rode:
#   streamlit run app.py
#
# Sugestão de requirements.txt:
#   streamlit
#   pandas
#   python-dateutil

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ===== Import do seu core (ajuste o caminho conforme sua pasta) =====
# Este arquivo deve existir conforme a resposta anterior:
# core/checklist_rules.py  com:
#   - Thresholds
#   - process_transactions_csv(file_bytes, companies_param, thresholds) -> (day_ref, checklist_rows, cards_rows)
from core.checklist_rules import Thresholds, process_transactions_csv


# ============================================================
# Helpers de UI / Normalização
# ============================================================

def _normalize_companies_input(text: str) -> str:
    """
    Aceita empresas por linha ou vírgula.
    Remove duplicados (case-insensitive) e devolve em CSV (vírgula).
    """
    raw = (text or "")
    parts = []
    for chunk in re.split(r"[\n,]+", raw):
        c = chunk.strip()
        if c:
            parts.append(c)

    seen = set()
    out = []
    for c in parts:
        k = c.upper()
        if k not in seen:
            seen.add(k)
            out.append(c)

    return ",".join(out)


def fmt_int_pt(x) -> str:
    try:
        n = float(x)
    except Exception:
        return "0"
    # arredonda "normal" para unidade mais próxima
    v = int(round(n))
    s = str(v)
    return re.sub(r"(?<=\d)(?=(\d{3})+(?!\d))", ".", s)


def fmt_money_ptbr(x) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    s = f"{v:,.2f}"
    # 1,234,567.89 -> 1.234.567,89
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def fmt_pct_int(pct_int) -> str:
    try:
        return f"{int(pct_int)}%"
    except Exception:
        return "0%"


def extract_accounts_list(accounts_list: str) -> List[str]:
    """
    "11717, 11758" -> ["11717","11758"]
    """
    s = str(accounts_list or "").strip()
    if not s:
        return []
    out = []
    for a in s.split(","):
        t = a.strip()
        if t:
            out.append(t)
    return out


def status_kind(status: str) -> str:
    s = str(status or "")
    if s == "Escalar (queda)":
        return "bad"
    if s == "Investigar":
        return "warn"
    if s == "Gerenciar (aumento)":
        return "info"
    if s == "Normal":
        return "ok"
    return "neutral"


def lock_icon(has_block: bool) -> str:
    return "🔒" if has_block else "🔓"


# ============================================================
# Leitura do CSV BaaS (opcional)
# ============================================================

def _guess_sep(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    if sample.count(";") >= sample.count(","):
        return ";"
    return ","


def _to_float(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    # suporta "1.234,56" ou "1234,56" ou "1234.56"
    if re.search(r"\d+,\d+", s):
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def read_baas_file(file_bytes: bytes) -> pd.DataFrame:
    """
    Retorna DataFrame normalizado:
      posicao, conta, agencia, nome, saldo_bloqueado
    """
    if not file_bytes:
        return pd.DataFrame()

    text = file_bytes.decode("utf-8", errors="replace")
    sep = _guess_sep(text[:4000])

    # Tenta com header
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str, header=0)
    if df.shape[1] < 7:
        # tenta sem header
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str, header=None)

    # Mapeia por nomes (quando existir)
    cols_lower = [str(c).strip().lower() for c in df.columns]

    def find_col(name_candidates: List[str], fallback_idx: Optional[int]) -> Optional[str]:
        for cand in name_candidates:
            if cand in cols_lower:
                i = cols_lower.index(cand)
                return df.columns[i]
        if fallback_idx is not None and df.shape[1] > fallback_idx:
            return df.columns[fallback_idx]
        return None

    col_pos = find_col(["posição", "posicao"], 0)
    col_conta = find_col(["conta"], 1)
    col_ag = find_col(["agência", "agencia"], 2)
    col_nome = find_col(["nome"], 3)
    col_block = find_col(["saldo bloqueado", "saldo_bloqueado", "saldo bloqueio", "bloqueado"], 6)

    if not col_conta or not col_block:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["posicao"] = df[col_pos].astype(str) if col_pos else ""
    out["conta"] = df[col_conta].astype(str).str.strip()
    out["agencia"] = df[col_ag].astype(str).str.strip() if col_ag else ""
    out["nome"] = df[col_nome].astype(str).str.strip() if col_nome else ""
    out["saldo_bloqueado"] = df[col_block].apply(_to_float)

    # remove linhas sem conta
    out = out[out["conta"].astype(str).str.len() > 0].copy()

    return out


def build_baas_index(df_baas: pd.DataFrame) -> Dict[str, Dict]:
    """
    Index por conta:
      {
        "11717": {"blocked": True, "saldo_bloqueado": 6322.07, "row": {...}}
      }
    Se a conta aparecer mais de uma vez, soma os bloqueios.
    """
    idx: Dict[str, Dict] = {}
    if df_baas is None or df_baas.empty:
        return idx

    for _, r in df_baas.iterrows():
        acc = str(r.get("conta", "")).strip()
        if not acc:
            continue
        blocked_val = float(r.get("saldo_bloqueado", 0.0) or 0.0)
        cur = idx.get(acc)
        if not cur:
            idx[acc] = {
                "blocked": blocked_val > 0,
                "saldo_bloqueado": blocked_val,
                "row": {
                    "posicao": r.get("posicao", ""),
                    "conta": acc,
                    "agencia": r.get("agencia", ""),
                    "nome": r.get("nome", ""),
                }
            }
        else:
            cur["saldo_bloqueado"] = float(cur.get("saldo_bloqueado", 0.0)) + blocked_val
            cur["blocked"] = cur["saldo_bloqueado"] > 0

    return idx


# ============================================================
# Paginação simples para DataFrame
# ============================================================

def paginate_df(df: pd.DataFrame, page_size: int = 15, key_prefix: str = "pg"):
    if df is None or df.empty:
        st.info("Sem dados.")
        return

    total = len(df)
    pages = max(1, (total + page_size - 1) // page_size)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.caption(f"Total: {total}")
    with col2:
        page = st.number_input(
            "Página",
            min_value=1,
            max_value=pages,
            value=1,
            step=1,
            key=f"{key_prefix}_page",
            label_visibility="collapsed",
        )
    with col3:
        st.caption(f"{page}/{pages}")

    start = (int(page) - 1) * page_size
    end = start + page_size
    st.dataframe(df.iloc[start:end], use_container_width=True, hide_index=True)


# ============================================================
# App
# ============================================================

st.set_page_config(page_title="Alertas - CSV", layout="wide")

st.title("Alertas (CSV → Checklist + Cards)")
st.caption(
    "Envie o CSV de transações e informe as empresas. Opcionalmente, envie o CSV BaaS para enriquecer com bloqueio cautelar."
)

with st.sidebar:
    st.header("Entradas")

    companies = st.text_area(
        "Empresas",
        height=160,
        placeholder="Ex: Treeal, Empresa X\nou uma por linha",
    )

    tx_file = st.file_uploader(
        "CSV Transações",
        type=["csv"],
        accept_multiple_files=False,
    )

    baas_file = st.file_uploader(
        "CSV BaaS (opcional)",
        type=["csv"],
        accept_multiple_files=False,
    )

    st.divider()
    st.subheader("Parâmetros")

    queda_critica = st.number_input("Queda crítica", value=-0.60, step=0.01, format="%.2f")
    aumento_relevante = st.number_input("Aumento relevante", value=0.80, step=0.01, format="%.2f")
    investigar_abs = st.number_input("Investigar abs", value=0.30, step=0.01, format="%.2f")

    run = st.button("Processar", type="primary", use_container_width=True)

# estado
if "result" not in st.session_state:
    st.session_state.result = None

# processamento
if run:
    if not companies.strip():
        st.error("Informe pelo menos 1 empresa.")
        st.session_state.result = None
    elif not tx_file:
        st.error("Envie o CSV de transações.")
        st.session_state.result = None
    else:
        thresholds = Thresholds(
            queda_critica=float(queda_critica),
            aumento_relevante=float(aumento_relevante),
            investigar_abs=float(investigar_abs),
        )

        companies_param = _normalize_companies_input(companies)

        try:
            with st.spinner("Processando transações..."):
                day_ref, checklist_rows, cards_rows = process_transactions_csv(
                    file_bytes=tx_file.getvalue(),
                    companies_param=companies_param,
                    thresholds=thresholds,
                )

            df_check = pd.DataFrame(checklist_rows or [])
            df_cards = pd.DataFrame(cards_rows or [])

            df_baas = pd.DataFrame()
            idx_baas: Dict[str, Dict] = {}

            if baas_file:
                with st.spinner("Lendo CSV BaaS..."):
                    df_baas = read_baas_file(baas_file.getvalue())
                    idx_baas = build_baas_index(df_baas)

            st.session_state.result = {
                "day_ref": day_ref,
                "df_check": df_check,
                "df_cards": df_cards,
                "df_baas": df_baas,
                "idx_baas": idx_baas,
            }

        except Exception as e:
            st.error("Falha ao processar. Detalhes abaixo:")
            st.exception(e)
            st.session_state.result = None

res = st.session_state.result

# ============================================================
# KPIs
# ============================================================

if res:
    df_check = res["df_check"]
    df_cards = res["df_cards"]

    day_ref = res["day_ref"] or "—"
    companies_count = int(len(df_check)) if df_check is not None else 0
    alerts_count = int((df_check["status"] != "Normal").sum()) if (df_check is not None and "status" in df_check.columns) else 0
    critical_count = int((df_check["status"] == "Escalar (queda)").sum()) if (df_check is not None and "status" in df_check.columns) else 0

    # contas com bloqueio: somente contas que estão no checklist
    blocked_accounts_count = 0
    if res.get("idx_baas") and df_check is not None and not df_check.empty and "accounts_list" in df_check.columns:
        accounts_set = set()
        for s in df_check["accounts_list"].astype(str).tolist():
            for acc in extract_accounts_list(s):
                accounts_set.add(acc)

        idx = res["idx_baas"]
        blocked_accounts_count = sum(
            1 for acc in accounts_set
            if acc in idx and idx[acc].get("blocked")
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Dia de referência", day_ref)
        st.caption("Aguardando fechamento diário")
    with c2:
        st.metric("Empresas no checklist", fmt_int_pt(companies_count))
        st.caption("Filtradas pelo input")
    with c3:
        st.metric("Alertas", fmt_int_pt(alerts_count))
        st.caption("status != Normal")
    with c4:
        st.metric("Críticos", fmt_int_pt(critical_count))
        st.caption("Escalar (queda)")
    with c5:
        st.metric("Contas com bloqueio", fmt_int_pt(blocked_accounts_count))
        st.caption("via CSV BaaS")

    st.divider()

# ============================================================
# Conteúdo principal: tabs
# ============================================================

tab1, tab2, tab3 = st.tabs(["Checklist", "Cards", "BaaS (Bloqueio Cautelar)"])

with tab1:
    st.subheader("Checklist")

    if not res:
        st.info("Envie os arquivos e clique em Processar.")
    else:
        df_check = res["df_check"]
        idx_baas = res.get("idx_baas") or {}

        if df_check is None or df_check.empty:
            st.warning("Checklist vazio (verifique filtro/CSV).")
        else:
            # adiciona coluna com cadeado (🔒/🔓) por empresa
            if "accounts_list" in df_check.columns:
                def _has_block_for_company(accounts_list: str) -> bool:
                    for acc in extract_accounts_list(accounts_list):
                        if acc in idx_baas and idx_baas[acc].get("blocked"):
                            return True
                    return False

                df_view = df_check.copy()
                df_view["bloqueio"] = df_view["accounts_list"].apply(lambda s: lock_icon(_has_block_for_company(s)))
            else:
                df_view = df_check.copy()
                df_view["bloqueio"] = "—"

            # seleção de colunas (mais “limpo”)
            cols = []
            if "status" in df_view.columns: cols.append("status")
            cols.append("bloqueio")
            if "company_name" in df_view.columns: cols.append("company_name")
            if "today_total_i" in df_view.columns: cols.append("today_total_i")
            if "avg_7d_i" in df_view.columns: cols.append("avg_7d_i")
            if "avg_15d_i" in df_view.columns: cols.append("avg_15d_i")
            if "avg_30d_i" in df_view.columns: cols.append("avg_30d_i")
            if "var_30d_pct" in df_view.columns: cols.append("var_30d_pct")
            if "accounts_list" in df_view.columns: cols.append("accounts_list")
            if "obs" in df_view.columns: cols.append("obs")

            df_show = df_view[cols].copy()

            # nomes amigáveis
            rename = {
                "company_name": "empresa",
                "today_total_i": "hoje",
                "avg_7d_i": "média_7d",
                "avg_15d_i": "média_15d",
                "avg_30d_i": "média_30d",
                "var_30d_pct": "var_30d",
                "accounts_list": "contas_ativas",
                "obs": "motivo",
            }
            df_show.rename(columns=rename, inplace=True)

            # formatos básicos
            if "var_30d" in df_show.columns:
                df_show["var_30d"] = df_show["var_30d"].apply(fmt_pct_int)
            for k in ["hoje", "média_7d", "média_15d", "média_30d"]:
                if k in df_show.columns:
                    df_show[k] = df_show[k].apply(fmt_int_pt)

            # filtros
            colf1, colf2, colf3 = st.columns([2, 1, 1])
            with colf1:
                q = st.text_input("Buscar empresa", value="", placeholder="Digite parte do nome...")
            with colf2:
                st_opt = ["Todos", "Escalar (queda)", "Investigar", "Gerenciar (aumento)", "Normal"]
                st_sel = st.selectbox("Status", st_opt, index=0)
            with colf3:
                page_size = st.selectbox("Paginação", [10, 15, 20, 30], index=1)

            df_f = df_show.copy()
            if q.strip():
                df_f = df_f[df_f["empresa"].astype(str).str.upper().str.contains(q.strip().upper(), na=False)]
            if st_sel != "Todos" and "status" in df_f.columns:
                df_f = df_f[df_f["status"] == st_sel]

            paginate_df(df_f, page_size=int(page_size), key_prefix="checklist")

with tab2:
    st.subheader("Cards de Alertas")

    if not res:
        st.info("Envie os arquivos e clique em Processar.")
    else:
        df_cards = res["df_cards"]
        df_check = res["df_check"]
        idx_baas = res.get("idx_baas") or {}

        if df_cards is None or df_cards.empty:
            st.info("Nenhum alerta (cards) encontrado para o processamento atual.")
        else:
            # filtro por status
            statuses = ["Todos"] + sorted([s for s in df_cards["status"].dropna().unique().tolist()])
            sel = st.selectbox("Filtrar cards por status", statuses, index=0)
            dfc = df_cards if sel == "Todos" else df_cards[df_cards["status"] == sel].copy()

            for _, row in dfc.iterrows():
                empresa = row.get("empresa", "")
                status = row.get("status", "")
                msg = row.get("mensagem_discord", "")

                # detecta bloqueio (se houver) por empresa via checklist
                has_block = False
                if df_check is not None and not df_check.empty:
                    m = df_check[df_check["company_name"] == empresa]
                    if not m.empty and "accounts_list" in m.columns:
                        for acc in extract_accounts_list(m.iloc[0]["accounts_list"]):
                            if acc in idx_baas and idx_baas[acc].get("blocked"):
                                has_block = True
                                break

                with st.expander(f"{status} {lock_icon(has_block)}  —  {empresa}", expanded=False):
                    st.code(msg, language="text")

with tab3:
    st.subheader("Checklist Bloqueio Cautelar (BaaS)")

    if not res:
        st.info("Envie os arquivos e clique em Processar.")
    else:
        df_baas = res["df_baas"]
        df_check = res["df_check"]

        if df_baas is None or df_baas.empty:
            st.info("Envie o CSV BaaS (opcional) e processe para ver esta tabela.")
        elif df_check is None or df_check.empty or "accounts_list" not in df_check.columns:
            st.info("Primeiro gere o checklist de empresas (CSV transações) para filtrar o BaaS.")
        else:
            # pega TODAS as contas ativas do checklist (todas as empresas filtradas)
            accounts_set = set()
            for s in df_check["accounts_list"].astype(str).tolist():
                for acc in extract_accounts_list(s):
                    accounts_set.add(acc)

            show = df_baas.copy()
            show["conta"] = show["conta"].astype(str).str.strip()
            show = show[show["conta"].isin(accounts_set)].copy()

            if show.empty:
                st.warning("CSV BaaS foi lido, mas nenhuma conta dele bateu com as contas do checklist.")
            else:
                show_view = show[["posicao", "conta", "agencia", "nome", "saldo_bloqueado"]].copy()
                show_view["status"] = show_view["saldo_bloqueado"].apply(
                    lambda x: "BLOQUEIO" if float(x or 0) > 0 else "SEM BLOQUEIO"
                )
                show_view["saldo_bloqueado"] = show_view["saldo_bloqueado"].apply(fmt_money_ptbr)

                page_size = st.selectbox("Paginação (BaaS)", [10, 15, 20, 30], index=1)
                paginate_df(show_view, page_size=int(page_size), key_prefix="baas")
