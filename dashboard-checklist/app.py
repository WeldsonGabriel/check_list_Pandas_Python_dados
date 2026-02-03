# app.py
# Streamlit dashboard — Checklist (CSV Transações) + (opcional) CSV BaaS (bloqueio cautelar)
# Ajustes:
# - Só processa depois de clicar em "Processar"
# - Filtro de empresas obrigatório (baseado no input)
# - BaaS casa por CONTA e só mostra o que estiver no filtro
# - Gráficos coloridos
# - Paginação: checklist 15/15, baas 10/10
# - Paginação sem botões externos (só number_input)
# - KPIs renomeados
# - Dropdown Top N (8/10/15) no Top variação 30d
# - Busca por empresa (Checklist e BaaS)

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from unidecode import unidecode
except Exception:
    unidecode = None

try:
    import plotly.express as px
except Exception:
    px = None


# =========================
# Config
# =========================
st.set_page_config(page_title="Checklist + BaaS", layout="wide")

STATUS_ORDER = ["Escalar (queda)", "Investigar", "Gerenciar (aumento)", "Normal"]

STATUS_COLOR = {
    "Escalar (queda)": "#e74c3c",
    "Investigar": "#f1c40f",
    "Gerenciar (aumento)": "#3498db",
    "Normal": "#2ecc71",
    "Desconhecido": "#aab4c8",
}

LOCK_ICON = "🔒"
UNLOCK_ICON = "🔓"
UNKNOWN_ICON = "⚪"


# =========================
# Thresholds
# =========================
@dataclass(frozen=True)
class Thresholds:
    queda_critica: float = -0.60
    aumento_relevante: float = 0.80
    investigar_abs: float = 0.30


# =========================
# Utils: normalize
# =========================
SUFFIXES = [
    " LTDA", " LTDA.", " SA", " S.A", " S.A.", " ME", " EPP", " EIRELI", " EI",
    " MEI", " S/S", " SS", " S C", " S/C", " SOCIEDADE ANONIMA"
]

PUNCT_RE = re.compile(r"[^A-Z0-9 ]+")


def _strip_accents(s: str) -> str:
    if not s:
        return ""
    if unidecode:
        return unidecode(s)
    return s


def normalize_company(name: str) -> str:
    s = str(name or "").strip()
    s = _strip_accents(s)
    s = s.upper()
    s = s.replace("\t", " ").replace("\n", " ")
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()

    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_companies_input(text: str) -> List[str]:
    raw = re.split(r"[\n,]+", text or "")
    out = []
    seen = set()
    for x in raw:
        x = x.strip()
        if not x:
            continue
        k = normalize_company(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


# =========================
# Utils: rounding / metrics
# =========================
def round_int_nearest(x: float) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    if v >= 0:
        return int(math.floor(v + 0.5))
    return int(math.ceil(v - 0.5))


def pct_int_ceil_ratio(var_ratio: float) -> int:
    try:
        pct = float(var_ratio) * 100.0
    except Exception:
        return 0
    return int(math.ceil(pct))


def safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return float(numer) / float(denom)


def calc_var(today: float, avg: float) -> float:
    try:
        t = float(today)
        a = float(avg)
    except Exception:
        return 0.0
    if a == 0:
        return 0.0
    return (t - a) / a


def severity_rank(status: str) -> int:
    s = str(status or "")
    if s == "Escalar (queda)":
        return 0
    if s == "Investigar":
        return 1
    if s == "Gerenciar (aumento)":
        return 2
    if s == "Normal":
        return 3
    return 9


def calc_status(var30: float, *, queda_critica: float, aumento_relevante: float, investigar_abs: float) -> str:
    if var30 <= queda_critica:
        return "Escalar (queda)"
    if abs(var30) >= investigar_abs:
        return "Investigar"
    if var30 >= aumento_relevante:
        return "Gerenciar (aumento)"
    return "Normal"


def calc_obs(status: str) -> str:
    if status == "Escalar (queda)":
        return "Queda crítica vs média histórica"
    if status == "Investigar":
        return "Variação relevante vs média histórica"
    if status == "Gerenciar (aumento)":
        return "Aumento relevante vs média histórica"
    return "Dentro do esperado"


# =========================
# CSV parsers (robustos)
# =========================
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for key in candidates:
        for lc, orig in low.items():
            if key in lc:
                return orig
    return None


def parse_transactions_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    content = uploaded_file.getvalue()
    if not content:
        return pd.DataFrame()

    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep=";", engine="python")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",", engine="python")

    if df is None or df.empty:
        return pd.DataFrame()

    date_col = _pick_col(df, ["data", "date", "transac", "day"])
    person_col = _pick_col(df, ["person"])
    credit_col = _pick_col(df, ["credit"])
    debit_col = _pick_col(df, ["debit"])

    missing = [x for x in [date_col, person_col, credit_col, debit_col] if x is None]
    if missing:
        cols = list(df.columns)
        if len(cols) >= 5:
            date_col = date_col or cols[0]
            person_col = person_col or cols[2]
            credit_col = credit_col or cols[3]
            debit_col = debit_col or cols[4]

    if not all([date_col, person_col, credit_col, debit_col]):
        raise ValueError(
            f"CSV transações: não consegui identificar colunas. "
            f"Encontrei date={date_col}, person={person_col}, credit={credit_col}, debit={debit_col}."
        )

    work = df[[date_col, person_col, credit_col, debit_col]].copy()

    work["date"] = pd.to_datetime(work[date_col], errors="coerce", dayfirst=True).dt.date

    person = work[person_col].astype(str).fillna("")
    parts = person.str.split(" - ", n=1, expand=True)
    account = parts[0].fillna("").astype(str).str.strip()
    company = (parts[1] if parts.shape[1] > 1 else "").fillna("").astype(str).str.strip()
    company = np.where(company == "", person, company)

    account = account.str.replace(r"\D+", "", regex=True)
    account = account.replace("", np.nan)

    work["account_id"] = account.astype("string")
    work["company_name"] = pd.Series(company).astype(str)

    credit = pd.to_numeric(work[credit_col], errors="coerce").fillna(0.0)
    debit = pd.to_numeric(work[debit_col], errors="coerce").fillna(0.0)
    work["total"] = (credit + debit).astype(float)

    work = work.dropna(subset=["date", "account_id"])
    work["company_key"] = work["company_name"].apply(normalize_company)

    return work[["date", "account_id", "company_name", "company_key", "total"]].copy()


def parse_baas_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    content = uploaded_file.getvalue()
    if not content:
        return pd.DataFrame()

    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep=";", engine="python")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",", engine="python")

    if df is None or df.empty:
        return pd.DataFrame()

    pos_col = _pick_col(df, ["posição", "posicao", "pos"])
    conta_col = _pick_col(df, ["conta"])
    ag_col = _pick_col(df, ["agência", "agencia"])
    nome_col = _pick_col(df, ["nome"])
    plano_col = _pick_col(df, ["plano"])
    saldo_col = _pick_col(df, ["saldo"])
    bloqueado_col = _pick_col(df, ["bloquead"])

    if saldo_col and "bloquead" in saldo_col.lower():
        saldo_col = None

    if not conta_col or not bloqueado_col or not nome_col:
        cols = list(df.columns)
        if len(cols) >= 7:
            pos_col = pos_col or cols[0]
            conta_col = conta_col or cols[1]
            ag_col = ag_col or cols[2]
            nome_col = nome_col or cols[3]
            plano_col = plano_col or cols[4]
            saldo_col = saldo_col or cols[5]
            bloqueado_col = bloqueado_col or cols[6]

    if not conta_col or not bloqueado_col or not nome_col:
        raise ValueError(
            f"CSV BaaS: não consegui identificar colunas. "
            f"Encontrei conta={conta_col}, bloqueado={bloqueado_col}, nome={nome_col}."
        )

    keep = [c for c in [pos_col, conta_col, ag_col, nome_col, plano_col, saldo_col, bloqueado_col] if c]
    work = df[keep].copy()

    work = work.rename(columns={
        pos_col: "posicao" if pos_col else pos_col,
        conta_col: "conta",
        ag_col: "agencia" if ag_col else ag_col,
        nome_col: "nome",
        plano_col: "plano" if plano_col else plano_col,
        saldo_col: "saldo" if saldo_col else saldo_col,
        bloqueado_col: "saldo_bloqueado",
    })

    work["conta"] = work["conta"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    work["saldo_bloqueado"] = pd.to_numeric(work["saldo_bloqueado"], errors="coerce").fillna(0.0).astype(float)
    work["nome"] = work["nome"].astype(str).fillna("")
    work["nome_key"] = work["nome"].apply(normalize_company)

    if "posicao" in work.columns:
        work["posicao"] = pd.to_numeric(work["posicao"], errors="coerce").fillna(0).astype(int)

    if "agencia" in work.columns:
        work["agencia"] = work["agencia"].astype(str).str.strip()

    return work


# =========================
# Core checklist builder
# =========================
def accounts_activity_for_company(
    base_acc: pd.DataFrame,
    company_key: str,
    start,
    end,
) -> Tuple[List[str], List[str]]:
    mask = (
        (base_acc["company_key"] == company_key)
        & (base_acc["date"] >= start)
        & (base_acc["date"] <= end)
    )
    df = base_acc.loc[mask, ["account_id", "total"]].copy()
    if df.empty:
        return [], []

    sums = df.groupby("account_id", as_index=False)["total"].sum()
    sums["account_id"] = sums["account_id"].astype(str)

    active = sums.loc[sums["total"] > 0, "account_id"].tolist()
    zero = sums.loc[sums["total"] == 0, "account_id"].tolist()

    active = sorted(set(active))
    zero = sorted(set(zero))
    return active, zero


def build_checklist(
    facts: pd.DataFrame,
    companies_keys: List[str],
    thresholds: Thresholds,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    if facts.empty:
        return "", pd.DataFrame(), pd.DataFrame()

    if companies_keys:
        facts = facts[facts["company_key"].isin(set(companies_keys))].copy()

    if facts.empty:
        return "", pd.DataFrame(), pd.DataFrame()

    day_ref = max(facts["date"])
    day_ref = pd.to_datetime(day_ref)
    d1 = (day_ref - timedelta(days=1)).date()

    base_acc = (
        facts.groupby(["account_id", "company_key", "company_name", "date"], as_index=False)["total"]
        .sum()
    )

    base_comp = (
        base_acc.groupby(["company_key", "company_name", "date"], as_index=False)["total"]
        .sum()
    )

    active_start = (day_ref - timedelta(days=30)).date()
    active_end = d1

    def sum_window_company(company_key: str, days: int) -> float:
        start = (day_ref - timedelta(days=days)).date()
        mask = (
            (base_comp["company_key"] == company_key)
            & (base_comp["date"] >= start)
            & (base_comp["date"] <= d1)
        )
        return float(base_comp.loc[mask, "total"].sum())

    def today_total_company(company_key: str) -> float:
        mask = (base_comp["company_key"] == company_key) & (base_comp["date"] == day_ref.date())
        return float(base_comp.loc[mask, "total"].sum())

    rows: List[Dict] = []
    for company_key in sorted(base_comp["company_key"].unique()):
        names = base_comp.loc[base_comp["company_key"] == company_key, "company_name"].dropna()
        company_name = str(names.iloc[-1]) if not names.empty else company_key

        active_accounts, zero_accounts = accounts_activity_for_company(
            base_acc=base_acc,
            company_key=company_key,
            start=active_start,
            end=active_end,
        )
        if not active_accounts:
            continue

        today_total = today_total_company(company_key)
        sum7 = sum_window_company(company_key, 7)
        sum15 = sum_window_company(company_key, 15)
        sum30 = sum_window_company(company_key, 30)

        avg7 = safe_div(sum7, 7)
        avg15 = safe_div(sum15, 15)
        avg30 = safe_div(sum30, 30)

        var7 = calc_var(today_total, avg7)
        var15 = calc_var(today_total, avg15)
        var30 = calc_var(today_total, avg30)

        status = calc_status(
            var30,
            queda_critica=thresholds.queda_critica,
            aumento_relevante=thresholds.aumento_relevante,
            investigar_abs=thresholds.investigar_abs,
        )
        obs = calc_obs(status)

        rows.append({
            "company_key": company_key,
            "company_name": company_name,
            "today_total": float(today_total),
            "avg_7d": float(avg7),
            "avg_15d": float(avg15),
            "avg_30d": float(avg30),
            "var_7d": float(var7),
            "var_15d": float(var15),
            "var_30d": float(var30),
            "today_total_i": round_int_nearest(today_total),
            "avg_7d_i": round_int_nearest(avg7),
            "avg_15d_i": round_int_nearest(avg15),
            "avg_30d_i": round_int_nearest(avg30),
            "var_7d_pct": pct_int_ceil_ratio(var7),
            "var_15d_pct": pct_int_ceil_ratio(var15),
            "var_30d_pct": pct_int_ceil_ratio(var30),
            "accounts_list": ", ".join(active_accounts),
            "accounts_zero_count": len(zero_accounts),
            "accounts_zero_list": ", ".join(zero_accounts),
            "status": status,
            "obs": obs,
            "day_ref": day_ref.date().isoformat(),
            "severity": severity_rank(status),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return day_ref.date().isoformat(), df, pd.DataFrame()

    df = df.sort_values(["severity", "company_name"], ascending=[True, True]).reset_index(drop=True)
    df_cards = df[df["status"] != "Normal"].copy()
    return day_ref.date().isoformat(), df, df_cards


# =========================
# BaaS enrichment (CASA POR CONTA)
# =========================
def enrich_with_baas(
    df_checklist: pd.DataFrame,
    df_baas: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if df_checklist is None or df_checklist.empty:
        return df_checklist, pd.DataFrame(), 0

    if df_baas is None or df_baas.empty:
        out = df_checklist.copy()
        out["has_block"] = False
        out["blocked_total"] = 0.0
        out["blocked_accounts"] = ""
        out["lock_icon"] = UNKNOWN_ICON
        return out, pd.DataFrame(), 0

    if "conta" not in df_baas.columns or "saldo_bloqueado" not in df_baas.columns:
        raise ValueError("CSV BaaS: esperado colunas 'conta' e 'saldo_bloqueado' após parse_baas_csv().")

    baas = df_baas.copy()
    baas["conta"] = baas["conta"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    baas["saldo_bloqueado"] = pd.to_numeric(baas["saldo_bloqueado"], errors="coerce").fillna(0.0).astype(float)

    baas_agg = (
        baas.groupby("conta", as_index=False)["saldo_bloqueado"].sum()
        .sort_values("saldo_bloqueado", ascending=False)
    )

    def _split_accounts(s: str) -> List[str]:
        if not s:
            return []
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        out = []
        for p in parts:
            p2 = re.sub(r"\D+", "", p)
            if p2:
                out.append(p2)
        return out

    out = df_checklist.copy()
    blocked_rows = []
    blocked_accounts_all = set()

    has_block_list = []
    blocked_total_list = []
    blocked_accounts_list = []
    lock_icon_list = []

    baas_by_conta = baas.copy()

    for _, row in out.iterrows():
        accounts = _split_accounts(row.get("accounts_list", ""))
        if not accounts:
            has_block_list.append(False)
            blocked_total_list.append(0.0)
            blocked_accounts_list.append("")
            lock_icon_list.append(UNKNOWN_ICON)
            continue

        hit = baas_agg[baas_agg["conta"].isin(accounts) & (baas_agg["saldo_bloqueado"] > 0)].copy()

        if hit.empty:
            has_block_list.append(False)
            blocked_total_list.append(0.0)
            blocked_accounts_list.append("")
            lock_icon_list.append(UNLOCK_ICON)
            continue

        has_block_list.append(True)
        total_blocked = float(hit["saldo_bloqueado"].sum())
        blocked_total_list.append(total_blocked)

        contas_hit = hit["conta"].astype(str).tolist()
        for c in contas_hit:
            blocked_accounts_all.add(c)

        blocked_accounts_list.append(", ".join(contas_hit))
        lock_icon_list.append(LOCK_ICON)

        det = baas_by_conta[baas_by_conta["conta"].isin(contas_hit)].copy()
        if not det.empty:
            det["company_key"] = row["company_key"]
            det["company_name"] = row["company_name"]
            blocked_rows.append(det)

    out["has_block"] = has_block_list
    out["blocked_total"] = blocked_total_list
    out["blocked_accounts"] = blocked_accounts_list
    out["lock_icon"] = lock_icon_list

    if blocked_rows:
        df_baas_table = pd.concat(blocked_rows, ignore_index=True)
        df_baas_table = df_baas_table[df_baas_table["saldo_bloqueado"] > 0].copy()
        df_baas_table = df_baas_table.sort_values("saldo_bloqueado", ascending=False).reset_index(drop=True)
    else:
        df_baas_table = pd.DataFrame()

    kpi_blocked_accounts = len(blocked_accounts_all)
    return out, df_baas_table, kpi_blocked_accounts


# =========================
# UI helpers
# =========================
def fmt_int_pt(n: int) -> str:
    try:
        v = int(n)
    except Exception:
        v = 0
    return f"{v:,}".replace(",", ".")


def fmt_money_pt(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def fmt_pct_int(x: int) -> str:
    try:
        v = int(x)
    except Exception:
        v = 0
    return f"{v}%"


def paginate_df(df: pd.DataFrame, page_size: int, key: str) -> Tuple[pd.DataFrame, int, int]:
    if df is None or df.empty:
        st.session_state[key] = 1
        return df, 1, 1

    total = len(df)
    pages = max(1, int(math.ceil(total / page_size)))

    if key not in st.session_state:
        st.session_state[key] = 1

    st.session_state[key] = max(1, min(int(st.session_state[key]), pages))
    page = int(st.session_state[key])

    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].copy(), page, pages


def render_pagination_controls(pages: int, state_key: str, label: str):
    if pages <= 1:
        st.session_state[state_key] = 1
        return

    if state_key not in st.session_state:
        st.session_state[state_key] = 1

    st.session_state[state_key] = max(1, min(int(st.session_state[state_key]), pages))

    st.number_input(
        label,
        min_value=1,
        max_value=pages,
        step=1,
        key=state_key,
    )


def filter_by_company_name(df: pd.DataFrame, query: str, col: str) -> pd.DataFrame:
    """
    Filtro case-insensitive simples por substring.
    - df[col] deve existir
    """
    q = (query or "").strip()
    if not q:
        return df
    try:
        mask = df[col].astype(str).str.contains(q, case=False, na=False)
        return df.loc[mask].copy()
    except Exception:
        return df


# =========================
# Layout
# =========================
st.title("Alertas (CSV → Checklist + Cards) + BaaS (opcional)")

st.caption(
    "Fluxo: cole as empresas → suba CSV de transações → (opcional) suba CSV BaaS → clique em **Processar**. "
    "Nada é calculado antes do botão."
)

with st.sidebar:
    st.header("Parâmetros")
    queda_critica = st.number_input("Queda crítica", value=-0.60, step=0.01, format="%.2f")
    aumento_relevante = st.number_input("Aumento relevante", value=0.80, step=0.01, format="%.2f")
    investigar_abs = st.number_input("Investigar abs", value=0.30, step=0.01, format="%.2f")

    st.divider()
    st.caption("Paginação")
    page_size_checklist = st.selectbox("Checklist (por página)", [15, 20, 30], index=0)
    page_size_baas = st.selectbox("BaaS (por página)", [10, 15, 20], index=0)

thresholds = Thresholds(
    queda_critica=float(queda_critica),
    aumento_relevante=float(aumento_relevante),
    investigar_abs=float(investigar_abs),
)

col_left, col_right = st.columns([1.2, 1.0], gap="large")

with col_left:
    companies_text = st.text_area(
        "Empresas (obrigatório) — vírgula ou uma por linha",
        height=140,
        placeholder="Ex:\nDOM DIGITAL\nPIX NA HORA LTDA",
        key="companies_text",
    )

with col_right:
    trans_file = st.file_uploader("CSV Transações (obrigatório)", type=["csv"], key="trans_csv")
    baas_file = st.file_uploader("CSV BaaS (opcional)", type=["csv"], key="baas_csv")

process = st.button("Processar", type="primary", use_container_width=True)

# =========================
# Processing (only on click)
# =========================
if process:
    companies_keys = split_companies_input(companies_text)
    if not companies_keys:
        st.error("Você precisa informar pelo menos 1 empresa no input antes de processar.")
        st.stop()
    if trans_file is None:
        st.error("Você precisa subir o CSV de transações antes de processar.")
        st.stop()

    with st.spinner("Lendo CSVs e calculando checklist..."):
        facts = parse_transactions_csv(trans_file)
        if facts.empty:
            st.error("O CSV de transações foi lido, mas ficou vazio após normalização (datas/contas inválidas).")
            st.stop()

        day_ref, df_checklist, df_cards = build_checklist(facts, companies_keys, thresholds)

        if df_checklist.empty:
            st.warning("Nenhuma empresa do input apareceu no CSV (após normalização), ou não há contas ativas no período.")
            st.session_state["day_ref"] = day_ref
            st.session_state["df_checklist"] = df_checklist
            st.session_state["df_cards"] = df_cards
            st.session_state["df_baas_table"] = pd.DataFrame()
            st.session_state["kpi_blocked_accounts"] = 0
        else:
            if baas_file is not None:
                df_baas = parse_baas_csv(baas_file)
            else:
                df_baas = pd.DataFrame()

            df_checklist2, df_baas_table, kpi_blocked_accounts = enrich_with_baas(df_checklist, df_baas)

            st.session_state["day_ref"] = day_ref
            st.session_state["df_checklist"] = df_checklist2
            st.session_state["df_cards"] = df_cards
            st.session_state["df_baas_table"] = df_baas_table
            st.session_state["kpi_blocked_accounts"] = kpi_blocked_accounts

            st.session_state["page_checklist"] = 1
            st.session_state["page_baas"] = 1


# =========================
# Render (only after process)
# =========================
if "df_checklist" in st.session_state:
    df_checklist = st.session_state.get("df_checklist", pd.DataFrame())
    df_cards = st.session_state.get("df_cards", pd.DataFrame())
    df_baas_table = st.session_state.get("df_baas_table", pd.DataFrame())
    day_ref = st.session_state.get("day_ref", "—")
    kpi_blocked_accounts = int(st.session_state.get("kpi_blocked_accounts", 0))

    # KPIs (renomeados)
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    with k1:
        st.metric("Dia de referência", day_ref)

    with k2:
        st.metric("Empresas filtradas", fmt_int_pt(len(df_checklist)))

    with k3:
        alerts_count = int((df_checklist["status"] != "Normal").sum()) if not df_checklist.empty else 0
        st.metric("Alertas", fmt_int_pt(alerts_count))

    with k4:
        critical = int((df_checklist["status"] == "Escalar (queda)").sum()) if not df_checklist.empty else 0
        st.metric("Alertas críticos", fmt_int_pt(critical))

    with k5:
        st.metric("Contas com bloqueio", fmt_int_pt(kpi_blocked_accounts))

    st.divider()

    # Charts
    st.subheader("Visão Geral")

    if px is None:
        st.warning("plotly não está disponível. Instale 'plotly' para ver gráficos.")
    else:
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.caption("Alertas por status")

            if df_checklist.empty:
                st.info("Sem dados.")
            else:
                sc = (
                    df_checklist.groupby("status", as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                )
                sc["order"] = sc["status"].apply(lambda s: STATUS_ORDER.index(s) if s in STATUS_ORDER else 999)
                sc = sc.sort_values("order").drop(columns=["order"])

                fig1 = px.bar(
                    sc,
                    x="status",
                    y="count",
                    color="status",
                    color_discrete_map=STATUS_COLOR,
                )
                fig1.update_layout(
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title=None,
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                fig1.update_traces(marker_line_width=0)
                st.plotly_chart(fig1, use_container_width=True)

        with c2:
            # dropdown top N (8/10/15)
            topn = st.selectbox("Top N", [8, 10, 15], index=0, key="topn_var30")

            st.caption("Top variação 30d (piores / melhores)")

            if df_checklist.empty:
                st.info("Sem dados.")
            else:
                tmp = df_checklist[["company_name", "var_30d"]].copy()
                tmp["var_30d"] = pd.to_numeric(tmp["var_30d"], errors="coerce").fillna(0.0)

                worst = tmp.sort_values("var_30d", ascending=True).head(int(topn))
                best = tmp.sort_values("var_30d", ascending=False).head(int(topn))
                merged = pd.concat([worst, best], ignore_index=True)

                merged["sign"] = np.where(merged["var_30d"] >= 0, "pos", "neg")

                fig2 = px.bar(
                    merged,
                    x="company_name",
                    y="var_30d",
                    color="sign",
                    color_discrete_map={"pos": "#2ecc71", "neg": "#e74c3c"},
                )
                fig2.update_layout(
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title=None,
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                fig2.update_traces(marker_line_width=0)
                fig2.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Checklist
    st.subheader("Checklist")

    if df_checklist.empty:
        st.info("Sem empresas para mostrar (verifique o filtro e os CSVs).")
    else:
        # busca por empresa (Checklist)
        q_check = st.text_input("Buscar empresa (Checklist)", placeholder="Digite parte do nome...", key="q_checklist")

        view = df_checklist.copy()
        view = filter_by_company_name(view, q_check, "company_name")

        view["status_view"] = view["status"].astype(str) + " " + view.get("lock_icon", UNKNOWN_ICON).astype(str)

        show_cols = [
            "status_view",
            "company_name",
            "today_total_i",
            "avg_7d_i",
            "avg_15d_i",
            "avg_30d_i",
            "var_30d_pct",
            "accounts_list",
            "obs",
        ]
        view_show = view[show_cols].rename(columns={
            "status_view": "Status",
            "company_name": "Empresa",
            "today_total_i": "Hoje",
            "avg_7d_i": "Média 7d",
            "avg_15d_i": "Média 15d",
            "avg_30d_i": "Média 30d",
            "var_30d_pct": "Var 30d",
            "accounts_list": "Contas",
            "obs": "Motivo",
        })

        view_show["Var 30d"] = view_show["Var 30d"].apply(fmt_pct_int)

        # se o filtro mudou e deixou a página inválida, a paginação já clampa
        st.session_state.setdefault("page_checklist", 1)
        page_df, page, pages = paginate_df(view_show, page_size=int(page_size_checklist), key="page_checklist")

        render_pagination_controls(pages, "page_checklist", f"Página Checklist (1–{pages})")
        st.dataframe(page_df, use_container_width=True, hide_index=True)

    st.divider()

    # BaaS Checklist
    st.subheader("Checklist Bloqueio Cautelar (BaaS) — por conta")

    if df_baas_table is None or df_baas_table.empty:
        st.caption("Sem dados de BaaS (ou nenhum bloqueio encontrado para as empresas filtradas).")
    else:
        # busca por empresa (BaaS)
        q_baas = st.text_input("Buscar empresa (BaaS)", placeholder="Digite parte do nome...", key="q_baas")

        t = df_baas_table.copy()
        t = filter_by_company_name(t, q_baas, "company_name")

        cols = []
        if "posicao" in t.columns:
            cols.append("posicao")
        cols += ["conta"]
        if "agencia" in t.columns:
            cols.append("agencia")
        cols += ["company_name", "saldo_bloqueado"]

        t = t[cols].copy()

        rename = {
            "posicao": "Posição",
            "conta": "Conta",
            "agencia": "Agência",
            "company_name": "Empresa",
            "saldo_bloqueado": "Saldo Bloqueado",
        }
        t = t.rename(columns=rename)

        t["Saldo Bloqueado"] = t["Saldo Bloqueado"].apply(fmt_money_pt)

        st.session_state.setdefault("page_baas", 1)
        page_df, page, pages = paginate_df(t, page_size=int(page_size_baas), key="page_baas")

        render_pagination_controls(pages, "page_baas", f"Página BaaS (1–{pages})")
        st.dataframe(page_df, use_container_width=True, hide_index=True)

    st.divider()

    # Cards
    st.subheader("Cards (para colar no Discord)")

    if df_cards is None or df_cards.empty:
        st.caption("Nenhum alerta (status != Normal).")
    else:
        for _, r in df_cards.iterrows():
            lock_icon = UNKNOWN_ICON
            if "lock_icon" in df_checklist.columns:
                match = df_checklist[df_checklist["company_key"] == r["company_key"]]
                if not match.empty:
                    lock_icon = str(match.iloc[0].get("lock_icon", UNKNOWN_ICON))

            title = f"{r['status']} {lock_icon} — {r['company_name']}"
            with st.expander(title, expanded=False):
                msg = (
                    f"ALERTA: {r['status']}\n"
                    f"Empresa: {r['company_name']}\n"
                    f"Data: {r['day_ref']}\n"
                    f"Motivo: {r['obs']}\n"
                    f"Total(D): {r['today_total_i']}\n"
                    f"Médias: 7d={r['avg_7d_i']} | 15d={r['avg_15d_i']} | 30d={r['avg_30d_i']}\n"
                    f"Variação: vs30={r['var_30d_pct']}% | vs15={r['var_15d_pct']}% | vs7={r['var_7d_pct']}%\n"
                    f"Contas (ativas): {r['accounts_list']}\n"
                    f"Contas zeradas: {r['accounts_zero_count']} ({r['accounts_zero_list']})"
                )
                st.code(msg, language="text")

else:
    st.info("Aguardando você preencher as empresas + subir o CSV e clicar em **Processar**.")