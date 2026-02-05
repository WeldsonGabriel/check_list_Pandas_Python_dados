# app.py
# Streamlit dashboard — Checklist (CSV Transações) + CSV BaaS (bloqueio cautelar) embutido no checklist
#
# Correção desta versão:
# ✅ Corrige StreamlitAPIException de paginação (não modifica session_state após widget instanciado)

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
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


# =========================
# Config
# =========================
st.set_page_config(page_title="Checklist + Bloqueio (BaaS)", layout="wide")

STATUS_ORDER = ["Alerta (queda/ zerada)", "Investigar", "Gerenciar (aumento)", "Normal"]

STATUS_COLOR = {
    "Alerta (queda/ zerada)": "#e74c3c",       # vermelho
    "Investigar": "#f1c40f",            # amarelo
    "Gerenciar (aumento)": "#3498db",   # azul
    "Normal": "#2ecc71",                # verde
    "Desconhecido": "#aab4c8",
}

LOCK_ICON = "🔒"
UNLOCK_ICON = "🔓"
UNKNOWN_ICON = "⚪"

st.markdown(
    """
<style>
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 12px 14px;
  border-radius: 14px;
}

.mini-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 10px;
}
.mini-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 12px 14px;
}
.mini-title { font-size: 13px; opacity: .75; margin-bottom: 6px; }
.mini-value { font-size: 28px; font-weight: 700; line-height: 1.1; }

.status-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.status-box {
  border-radius: 14px;
  padding: 14px;
  text-align: center;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}
.status-num { font-size: 28px; font-weight: 800; line-height: 1.1; }
.status-lab { margin-top: 6px; font-size: 13px; opacity: .85; }

.sb-red { border-color: rgba(231,76,60,0.45); background: linear-gradient(180deg, rgba(231,76,60,0.12), rgba(255,255,255,0.02)); }
.sb-yellow { border-color: rgba(241,196,15,0.45); background: linear-gradient(180deg, rgba(241,196,15,0.12), rgba(255,255,255,0.02)); }
.sb-blue { border-color: rgba(52,152,219,0.45); background: linear-gradient(180deg, rgba(52,152,219,0.12), rgba(255,255,255,0.02)); }
.sb-green { border-color: rgba(46,204,113,0.45); background: linear-gradient(180deg, rgba(46,204,113,0.10), rgba(255,255,255,0.02)); }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Thresholds
# =========================
@dataclass(frozen=True)
class Thresholds:
    queda_critica: float = -0.60
    aumento_relevante: float = 0.80
    investigar_abs: float = 0.30


# =========================
# Normalize helpers
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
# Metrics helpers
# =========================
def safe_div(n: float, d: float) -> float:
    if not d:
        return 0.0
    return float(n) / float(d)


def calc_var(today: float, avg: float) -> float:
    try:
        t = float(today)
        a = float(avg)
    except Exception:
        return 0.0
    if a == 0:
        return 0.0
    return (t - a) / a


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


def calc_status(var30: float, *, queda_critica: float, aumento_relevante: float, investigar_abs: float) -> str:
    if var30 <= queda_critica:
        return "Alerta (queda/ zerada)"
    if abs(var30) >= investigar_abs:
        return "Investigar"
    if var30 >= aumento_relevante:
        return "Gerenciar (aumento)"
    return "Normal"


def severity_rank(status: str) -> int:
    if status == "Alerta (queda/ zerada)":
        return 0
    if status == "Investigar":
        return 1
    if status == "Gerenciar (aumento)":
        return 2
    if status == "Normal":
        return 3
    return 9


def calc_obs(status: str) -> str:
    if status == "Alerta (queda/ zerada)":
        return "Queda crítica vs média histórica"
    if status == "Investigar":
        return "Variação relevante vs média histórica"
    if status == "Gerenciar (aumento)":
        return "Aumento relevante vs média histórica"
    return "Dentro do esperado"


# =========================
# Formatting
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


def fmt_pct_int(n: int) -> str:
    try:
        v = int(n)
    except Exception:
        v = 0
    return f"{v}%"


# =========================
# CSV helpers
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

    if any(x is None for x in [date_col, person_col, credit_col, debit_col]):
        cols = list(df.columns)
        if len(cols) >= 5:
            date_col = date_col or cols[0]
            person_col = person_col or cols[2]
            credit_col = credit_col or cols[3]
            debit_col = debit_col or cols[4]

    if not all([date_col, person_col, credit_col, debit_col]):
        raise ValueError(
            f"CSV transações: não identifiquei colunas. "
            f"date={date_col}, person={person_col}, credit={credit_col}, debit={debit_col}."
        )

    work = df[[date_col, person_col, credit_col, debit_col]].copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce", dayfirst=True).dt.date

    person = work[person_col].astype(str).fillna("")
    parts = person.str.split(" - ", n=1, expand=True)
    acc = parts[0].fillna("").astype(str).str.strip()
    comp = (parts[1] if parts.shape[1] > 1 else "").fillna("").astype(str).str.strip()
    comp = np.where(comp == "", person, comp)

    acc = acc.str.replace(r"\D+", "", regex=True)
    acc = acc.replace("", np.nan)

    credit = pd.to_numeric(work[credit_col], errors="coerce").fillna(0.0)
    debit = pd.to_numeric(work[debit_col], errors="coerce").fillna(0.0)

    out = pd.DataFrame({
        "date": work["date"],
        "account_id": acc.astype("string"),
        "company_name": pd.Series(comp).astype(str),
        "total": (credit + debit).astype(float),
    })

    out = out.dropna(subset=["date", "account_id"]).copy()
    out["company_key"] = out["company_name"].apply(normalize_company)
    return out[["date", "account_id", "company_name", "company_key", "total"]].copy()


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
    bloqueado_col = _pick_col(df, ["bloquead"])

    saldo_col = _pick_col(df, ["saldo"])
    if saldo_col and "bloquead" in saldo_col.lower():
        saldo_col = None

    if not conta_col or not bloqueado_col:
        cols = list(df.columns)
        if len(cols) >= 7:
            pos_col = pos_col or cols[0]
            conta_col = conta_col or cols[1]
            ag_col = ag_col or cols[2]
            nome_col = nome_col or cols[3]
            plano_col = plano_col or cols[4]
            saldo_col = saldo_col or cols[5]
            bloqueado_col = bloqueado_col or cols[6]

    if not conta_col or not bloqueado_col:
        raise ValueError(
            f"CSV BaaS: colunas não identificadas. conta={conta_col}, bloqueado={bloqueado_col}"
        )

    keep = [c for c in [pos_col, conta_col, ag_col, nome_col, plano_col, saldo_col, bloqueado_col] if c]
    work = df[keep].copy()

    rename = {conta_col: "conta", bloqueado_col: "saldo_bloqueado"}
    if pos_col:
        rename[pos_col] = "posicao"
    if ag_col:
        rename[ag_col] = "agencia"
    if nome_col:
        rename[nome_col] = "nome"
    if plano_col:
        rename[plano_col] = "plano"
    if saldo_col:
        rename[saldo_col] = "saldo"

    work = work.rename(columns=rename)

    work["conta"] = work["conta"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    work["saldo_bloqueado"] = pd.to_numeric(work["saldo_bloqueado"], errors="coerce").fillna(0.0).astype(float)

    if "posicao" in work.columns:
        work["posicao"] = pd.to_numeric(work["posicao"], errors="coerce").fillna(0).astype(int)
    if "agencia" in work.columns:
        work["agencia"] = work["agencia"].astype(str).str.strip()

    return work


# =========================
# Checklist POR CONTA
# =========================
def build_checklist_per_account(
    facts: pd.DataFrame,
    companies_keys: List[str],
    thresholds: Thresholds,
) -> Tuple[str, pd.DataFrame]:
    if facts.empty:
        return "", pd.DataFrame()

    facts = facts[facts["company_key"].isin(set(companies_keys))].copy()
    if facts.empty:
        return "", pd.DataFrame()

    day_ref = max(facts["date"])
    day_ref = pd.to_datetime(day_ref).date()
    d1 = (pd.to_datetime(day_ref) - timedelta(days=1)).date()

    base_acc = (
        facts.groupby(["account_id", "company_key", "company_name", "date"], as_index=False)["total"]
        .sum()
    )

    active_start = (pd.to_datetime(day_ref) - timedelta(days=30)).date()
    active_end = d1

    mask_active = (base_acc["date"] >= active_start) & (base_acc["date"] <= active_end)
    active_sums = (
        base_acc.loc[mask_active]
        .groupby(["account_id", "company_key"], as_index=False)["total"].sum()
        .rename(columns={"total": "sum_30d_window"})
    )

    active_keys = active_sums.loc[active_sums["sum_30d_window"] > 0, ["account_id", "company_key"]]
    if active_keys.empty:
        return day_ref.isoformat(), pd.DataFrame()

    base_acc = base_acc.merge(active_keys.assign(_keep=1), on=["account_id", "company_key"], how="inner").drop(columns=["_keep"])

    def sum_window(account_id: str, company_key: str, days: int) -> float:
        start = (pd.to_datetime(day_ref) - timedelta(days=days)).date()
        m = (
            (base_acc["account_id"] == account_id) &
            (base_acc["company_key"] == company_key) &
            (base_acc["date"] >= start) &
            (base_acc["date"] <= d1)
        )
        return float(base_acc.loc[m, "total"].sum())

    def today_total(account_id: str, company_key: str) -> float:
        m = (
            (base_acc["account_id"] == account_id) &
            (base_acc["company_key"] == company_key) &
            (base_acc["date"] == day_ref)
        )
        return float(base_acc.loc[m, "total"].sum())

    dims = base_acc[["account_id", "company_key", "company_name"]].drop_duplicates().copy()

    rows: List[Dict] = []
    for _, d in dims.iterrows():
        account_id = str(d["account_id"])
        company_key = str(d["company_key"])
        company_name = str(d["company_name"])

        tdy = today_total(account_id, company_key)
        s7 = sum_window(account_id, company_key, 7)
        s15 = sum_window(account_id, company_key, 15)
        s30 = sum_window(account_id, company_key, 30)

        a7 = safe_div(s7, 7)
        a15 = safe_div(s15, 15)
        a30 = safe_div(s30, 30)

        v7 = calc_var(tdy, a7)
        v15 = calc_var(tdy, a15)
        v30 = calc_var(tdy, a30)

        status = calc_status(
            v30,
            queda_critica=thresholds.queda_critica,
            aumento_relevante=thresholds.aumento_relevante,
            investigar_abs=thresholds.investigar_abs,
        )

        rows.append({
            "company_key": company_key,
            "company_name": company_name,
            "account_id": account_id,
            "day_ref": day_ref.isoformat(),

            "today_total": float(tdy),
            "avg_7d": float(a7),
            "avg_15d": float(a15),
            "avg_30d": float(a30),

            "var_7d": float(v7),
            "var_15d": float(v15),
            "var_30d": float(v30),

            "today_total_i": round_int_nearest(tdy),
            "avg_7d_i": round_int_nearest(a7),
            "avg_15d_i": round_int_nearest(a15),
            "avg_30d_i": round_int_nearest(a30),

            "var_7d_pct": pct_int_ceil_ratio(v7),
            "var_15d_pct": pct_int_ceil_ratio(v15),
            "var_30d_pct": pct_int_ceil_ratio(v30),

            "status": status,
            "obs": calc_obs(status),
            "severity": severity_rank(status),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return day_ref.isoformat(), df

    df = df.sort_values(["severity", "company_name", "account_id"], ascending=[True, True, True]).reset_index(drop=True)
    return day_ref.isoformat(), df


# =========================
# Enrich BaaS (por conta)
# =========================
def enrich_with_baas_accounts(
    df_checklist: pd.DataFrame,
    df_baas: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, float]:
    out = df_checklist.copy()
    out["account_id"] = out["account_id"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)

    if df_baas is None or df_baas.empty or out.empty:
        out["saldo_bloqueado_total"] = 0.0
        out["has_block"] = False
        out["lock_icon"] = UNLOCK_ICON if not out.empty else UNKNOWN_ICON
        return out, 0, 0.0

    baas = df_baas.copy()
    baas["conta"] = baas["conta"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    baas["saldo_bloqueado"] = pd.to_numeric(baas["saldo_bloqueado"], errors="coerce").fillna(0.0).astype(float)

    baas_agg = (
        baas.groupby("conta", as_index=False)["saldo_bloqueado"].sum()
        .rename(columns={"conta": "account_id", "saldo_bloqueado": "saldo_bloqueado_total"})
    )

    merged = out.merge(baas_agg, on="account_id", how="left")
    merged["saldo_bloqueado_total"] = pd.to_numeric(merged["saldo_bloqueado_total"], errors="coerce").fillna(0.0).astype(float)

    merged["has_block"] = merged["saldo_bloqueado_total"] > 0
    merged["lock_icon"] = np.where(merged["has_block"], LOCK_ICON, UNLOCK_ICON)

    qtd = int(merged["has_block"].sum())
    soma = float(merged.loc[merged["has_block"], "saldo_bloqueado_total"].sum())
    return merged, qtd, soma


# =========================
# Pagination (FIX)
# =========================
def get_pages(total_rows: int, page_size: int) -> int:
    return max(1, int(math.ceil(total_rows / max(1, page_size))))


def slice_page(df: pd.DataFrame, page: int, page_size: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    page = max(1, int(page))
    start = (page - 1) * int(page_size)
    end = start + int(page_size)
    return df.iloc[start:end].copy()


# =========================
# Gauge + status boxes (mesmo visual que você aprovou)
# =========================
def render_status_boxes(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("Sem dados.")
        return

    counts = {s: int((df["status"] == s).sum()) for s in STATUS_ORDER}

    st.markdown(
        f"""
<div class="status-row">
  <div class="status-box sb-red">
    <div class="status-num">{counts["Alerta (queda/ zerada)"]}</div>
    <div class="status-lab">Alerta</div>
  </div>
  <div class="status-box sb-yellow">
    <div class="status-num">{counts["Investigar"]}</div>
    <div class="status-lab">Investigar</div>
  </div>
  <div class="status-box sb-blue">
    <div class="status-num">{counts["Gerenciar (aumento)"]}</div>
    <div class="status-lab">Gerenciar</div>
  </div>
  <div class="status-box sb-green">
    <div class="status-num">{counts["Normal"]}</div>
    <div class="status-lab">Normal</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_gauge_panel(
    title: str,
    today_total: float,
    avg7: float,
    avg15: float,
    avg30: float,
    var7: float,
    var15: float,
    var30: float,
    height: int = 480,
):
    if go is None:
        st.warning("plotly não está disponível (instale plotly).")
        return

    max_ref = max(today_total, avg7, avg15, avg30, 1.0)
    gauge_max = max_ref * 1.20

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(today_total),
            number={"valueformat": ".0f"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": STATUS_COLOR["Gerenciar (aumento)"]},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [0, gauge_max], "color": "rgba(255,255,255,0.08)"}],
            },
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.90)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
<div class="mini-grid">
  <div class="mini-card">
    <div class="mini-title">Média 7d</div>
    <div class="mini-value">{fmt_int_pt(round_int_nearest(avg7))}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Variação 7d %</div>
    <div class="mini-value">{fmt_pct_int(pct_int_ceil_ratio(var7))}</div>
  </div>

  <div class="mini-card">
    <div class="mini-title">Média 15d</div>
    <div class="mini-value">{fmt_int_pt(round_int_nearest(avg15))}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Variação 15d %</div>
    <div class="mini-value">{fmt_pct_int(pct_int_ceil_ratio(var15))}</div>
  </div>

  <div class="mini-card">
    <div class="mini-title">Média 30d</div>
    <div class="mini-value">{fmt_int_pt(round_int_nearest(avg30))}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Variação 30d %</div>
    <div class="mini-value">{fmt_pct_int(pct_int_ceil_ratio(var30))}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================
# UI
# =========================
st.title("Alertas (Checklist por conta) + Bloqueio (BaaS)")

with st.sidebar:
    st.header("Parâmetros")
    queda_critica = st.number_input("Queda crítica", value=-0.60, step=0.01, format="%.2f")
    aumento_relevante = st.number_input("Aumento relevante", value=0.80, step=0.01, format="%.2f")
    investigar_abs = st.number_input("Investigar abs", value=0.30, step=0.01, format="%.2f")

    st.divider()
    st.header("Paginação")
    page_size_checklist = st.selectbox("Checklist (por página)", [15, 20, 30, 50], index=0)

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

if process:
    companies_keys = split_companies_input(companies_text)
    if not companies_keys:
        st.error("Você precisa informar pelo menos 1 empresa no input antes de processar.")
        st.stop()
    if trans_file is None:
        st.error("Você precisa subir o CSV de transações antes de processar.")
        st.stop()

    with st.spinner("Lendo CSVs e calculando checklist por conta..."):
        facts = parse_transactions_csv(trans_file)
        if facts.empty:
            st.error("O CSV de transações ficou vazio após normalização (datas/contas inválidas).")
            st.stop()

        day_ref, df_checklist = build_checklist_per_account(facts, companies_keys, thresholds)

        if df_checklist.empty:
            st.warning("Nenhuma conta ativa encontrada para as empresas filtradas na janela de 30d.")
            st.session_state["day_ref"] = day_ref
            st.session_state["df_checklist"] = df_checklist
        else:
            df_baas = parse_baas_csv(baas_file) if baas_file is not None else pd.DataFrame()
            df_checklist2, kpi_blocked_accounts, kpi_blocked_sum = enrich_with_baas_accounts(df_checklist, df_baas)

            st.session_state["day_ref"] = day_ref
            st.session_state["df_checklist"] = df_checklist2
            st.session_state["kpi_blocked_accounts"] = kpi_blocked_accounts
            st.session_state["kpi_blocked_sum"] = kpi_blocked_sum

            st.session_state["page_checklist"] = 1

# Render
if "df_checklist" not in st.session_state:
    st.info("Aguardando você preencher as empresas + subir o CSV e clicar em **Processar**.")
    st.stop()

df_checklist = st.session_state.get("df_checklist", pd.DataFrame())
day_ref = st.session_state.get("day_ref", "—")
kpi_blocked_accounts = int(st.session_state.get("kpi_blocked_accounts", 0))

# KPIs topo
k1, k2, k3, k4, k5 = st.columns(5, gap="small")
with k1:
    st.metric("Dia de referência", day_ref)
with k2:
    uniq_companies = int(df_checklist["company_key"].nunique()) if not df_checklist.empty else 0
    st.metric("Empresas filtradas", fmt_int_pt(uniq_companies))
with k3:
    alerts_count = int((df_checklist["status"] != "Normal").sum()) if not df_checklist.empty else 0
    st.metric("Alertas", fmt_int_pt(alerts_count))
with k4:
    critical = int((df_checklist["status"] == "Alerta (queda/ zerada)").sum()) if not df_checklist.empty else 0
    st.metric("Alertas críticos", fmt_int_pt(critical))
with k5:
    st.metric("Contas com bloqueio", fmt_int_pt(kpi_blocked_accounts))

st.divider()

st.subheader("Visão Geral")
render_status_boxes(df_checklist)
st.divider()

# Gauge + Top30
if px is not None and go is not None and not df_checklist.empty:
    left, right = st.columns([1.1, 1.0], gap="large")

    companies_for_dropdown = sorted(df_checklist["company_name"].dropna().astype(str).unique().tolist())
    options = ["TOTAL GERAL"] + companies_for_dropdown

    with left:
        st.caption("Painel (Total geral ou por empresa)")
        selected = st.selectbox("Selecionar", options, index=0, key="gauge_company_select")

        if selected == "TOTAL GERAL":
            df_scope = df_checklist.copy()
            title = "TOTAL HOJE"
        else:
            df_scope = df_checklist[df_checklist["company_name"].astype(str) == str(selected)].copy()
            title = str(selected)

        today_total = float(df_scope["today_total"].sum())
        avg7 = float(df_scope["avg_7d"].sum())
        avg15 = float(df_scope["avg_15d"].sum())
        avg30 = float(df_scope["avg_30d"].sum())

        var7 = calc_var(today_total, avg7)
        var15 = calc_var(today_total, avg15)
        var30 = calc_var(today_total, avg30)

        render_gauge_panel(title, today_total, avg7, avg15, avg30, var7, var15, var30, height=200)

    with right:
        st.caption("Top variação 30d (piores / melhores)")
        topn = st.selectbox("Top", [8, 10, 15], index=0, key="topn_select")

        g = (
            df_checklist.groupby(["company_name"], as_index=False)
            .agg(today_total=("today_total", "sum"), avg_30d=("avg_30d", "sum"))
        )
        g["var_30d"] = g.apply(lambda r: calc_var(r["today_total"], r["avg_30d"]), axis=1)

        worst = g.sort_values("var_30d", ascending=True).head(int(topn))
        best = g.sort_values("var_30d", ascending=False).head(int(topn))
        merged = pd.concat([worst, best], ignore_index=True)
        merged["sign"] = np.where(merged["var_30d"] >= 0, "pos", "neg")

        fig = px.bar(
            merged,
            x="company_name",
            y="var_30d",
            color="sign",
            color_discrete_map={"pos": STATUS_COLOR["Normal"], "neg": STATUS_COLOR["Alerta (queda/ zerada)"]},
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title=None,
            yaxis_title=None,
            height=480,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_traces(marker_line_width=0)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# Checklist + busca
st.subheader("Checklist (por conta)")

if df_checklist.empty:
    st.info("Sem dados para mostrar.")
    st.stop()

f1, f2, f3 = st.columns([1.2, 1.0, 0.8], gap="small")
with f1:
    q_empresa = st.text_input("Buscar empresa", value="", key="q_empresa")
with f2:
    q_conta = st.text_input("Buscar conta", value="", key="q_conta")
with f3:
    status_filter = st.selectbox("Status", ["Todos"] + STATUS_ORDER, index=0, key="status_filter")

view = df_checklist.copy()

if q_empresa.strip():
    q = q_empresa.strip().lower()
    view = view[view["company_name"].astype(str).str.lower().str.contains(q, na=False)]

if q_conta.strip():
    q = re.sub(r"\D+", "", q_conta.strip())
    if q:
        view = view[view["account_id"].astype(str).str.contains(q, na=False)]

if status_filter != "Todos":
    view = view[view["status"] == status_filter]

# tabela
view["Status"] = view["status"].astype(str) + " " + view["lock_icon"].astype(str)
view["Conta"] = view["account_id"].astype(str)
view["Empresa"] = view["company_name"].astype(str)
view["Hoje"] = view["today_total_i"].apply(fmt_int_pt)
view["Média 7d"] = view["avg_7d_i"].apply(fmt_int_pt)
view["Média 15d"] = view["avg_15d_i"].apply(fmt_int_pt)
view["Média 30d"] = view["avg_30d_i"].apply(fmt_int_pt)
view["Var 30d"] = view["var_30d_pct"].apply(fmt_pct_int)
view["Saldo Bloqueado"] = view["saldo_bloqueado_total"].apply(fmt_money_pt)
view["Motivo"] = view["obs"].astype(str)

show = view[["Status", "Conta", "Empresa", "Hoje", "Média 7d", "Média 15d", "Média 30d", "Var 30d", "Saldo Bloqueado", "Motivo"]].copy()

# === PAGINAÇÃO FIXA (sem modificar session_state depois do widget)
total_rows = int(len(show))
pages = get_pages(total_rows, int(page_size_checklist))

st.session_state.setdefault("page_checklist", 1)
# clamp ANTES do widget
st.session_state["page_checklist"] = max(1, min(pages, int(st.session_state["page_checklist"])))

st.number_input(
    f"Página Checklist (1–{pages})",
    min_value=1,
    max_value=pages,
    value=int(st.session_state["page_checklist"]),
    step=1,
    key="page_checklist",
)

# só lê e recorta
page_now = max(1, min(pages, int(st.session_state["page_checklist"])))
page_df = slice_page(show, page_now, int(page_size_checklist))

st.dataframe(page_df, use_container_width=True, hide_index=True)

st.divider()

# Cards (Discord)
st.subheader("Cards (para colar no Discord)")

alerts = df_checklist[df_checklist["status"] != "Normal"].copy()
if alerts.empty:
    st.caption("Nenhum alerta (status != Normal).")
else:
    for _, r in alerts.iterrows():
        title = f"{r['status']} {r['lock_icon']} — {r['company_name']} | Conta {r['account_id']}"
        with st.expander(title, expanded=False):
            msg = (
                f"ALERTA: {r['status']}\n"
                f"Empresa: {r['company_name']}\n"
                f"Conta: {r['account_id']}\n"
                f"Data: {r['day_ref']}\n"
                f"Motivo: {r['obs']}\n"
                f"Total(D): {r['today_total_i']}\n"
                f"Médias: 7d={r['avg_7d_i']} | 15d={r['avg_15d_i']} | 30d={r['avg_30d_i']}\n"
                f"Variação: vs30={r['var_30d_pct']}% | vs15={r['var_15d_pct']}% | vs7={r['var_7d_pct']}%\n"
                f"Bloqueio: {fmt_money_pt(r.get('saldo_bloqueado_total', 0.0))} {r['lock_icon']}"
            )
            st.code(msg, language="text")