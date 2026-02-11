# app.py
# Streamlit dashboard — Checklist (por CONTA) + Bloqueio (BaaS)
# Versão: semanas W1..W4 (últimas 4 semanas) + gauge "TOTAL (4 SEMANAS)"
#
# Correções desta versão:
# ✅ Ground dropdown por CONTA (Empresa + Conta) igual checklist
# ✅ Conta NÃO abrevia: sempre pega do "Person Name" (prefixo antes do " - ") e mantém como string
# ✅ BaaS merge por chave numérica (digits-only) para encontrar bloqueios com robustez
# ✅ KPI label: "Contas com bloqueio"
# ✅ Sidebar: seletor de altura do Ground (220 / 340 / 480)

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
    "Investigar": "#f1c40f",                  # amarelo
    "Gerenciar (aumento)": "#3498db",         # azul
    "Normal": "#2ecc71",                      # verde
    "Desconhecido": "#aab4c8",
}

LOCK_ICON = "🔒"
UNLOCK_ICON = "🔓"
UNKNOWN_ICON = "⚪"

STATUS_DOT = {
    "Alerta (queda/ zerada)": "🔴",
    "Investigar": "🟡",
    "Gerenciar (aumento)": "🔵",
    "Normal": "🟢",
    "Desconhecido": "⚪",
}

st.markdown(
    """
<style>
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 12px 14px;
  border-radius: 14px;
}

.mini-grid-6 {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
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
.mini-value { font-size: 26px; font-weight: 800; line-height: 1.1; }
.mini-sub  { font-size: 12px; opacity: .65; margin-top: 6px; }

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
    baixo_periodo_limite: int = 1


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


def digits_only(s: str) -> str:
    return re.sub(r"\D+", "", str(s or ""))


# =========================
# Math / status helpers
# =========================
def safe_div(n: float, d: float) -> float:
    if not d:
        return 0.0
    return float(n) / float(d)


def calc_var(curr: float, ref: float) -> float:
    try:
        c = float(curr)
        r = float(ref)
    except Exception:
        return 0.0
    if r == 0:
        return 0.0
    return (c - r) / r


def pct_int_ceil_ratio(var_ratio: float) -> int:
    try:
        pct = float(var_ratio) * 100.0
    except Exception:
        return 0
    return int(math.ceil(pct))


def calc_status(var: float, *, queda_critica: float, aumento_relevante: float, investigar_abs: float) -> str:
    if var <= queda_critica:
        return "Alerta (queda/ zerada)"
    if var >= aumento_relevante:
        return "Gerenciar (aumento)"
    if abs(var) >= investigar_abs:
        return "Investigar"
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


def calc_obs_week(status: str, total_4w: int, baixo_periodo_limite: int) -> str:
    if total_4w <= 0:
        return "Conta zerada no período (4 semanas)"
    if total_4w <= int(baixo_periodo_limite):
        return "Volume muito baixo no período (4 semanas)"
    if status == "Alerta (queda/ zerada)":
        return "Queda crítica vs média das 3 semanas anteriores (W4 vs W1–W3)"
    if status == "Gerenciar (aumento)":
        return "Aumento relevante vs média das 3 semanas anteriores (W4 vs W1–W3)"
    if status == "Investigar":
        return "Variação relevante vs média das 3 semanas anteriores (W4 vs W1–W3)"
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


def fmt_range(d0, d1) -> str:
    try:
        a = pd.to_datetime(d0).date()
        b = pd.to_datetime(d1).date()
        return f"{a.strftime('%d/%m')}-{b.strftime('%d/%m')}"
    except Exception:
        return "—"


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
    """
    CREDIT/DEBIT = quantidade (int)
    account_id = SEMPRE vem do "Person Name" (prefixo antes do ' - ')
    """
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
            f"CSV transações: não identifiquei colunas mínimas. "
            f"date={date_col}, person={person_col}, credit={credit_col}, debit={debit_col}."
        )

    work = df[[date_col, person_col, credit_col, debit_col]].copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce", dayfirst=True).dt.date

    person = work[person_col].astype(str).fillna("")
    parts = person.str.split(" - ", n=1, expand=True)

    # conta COMPLETA vem do prefixo do person name (ex: "11058 - ...")
    acc_raw = parts[0].fillna("").astype(str).str.strip()
    acc_full = acc_raw.map(digits_only)  # mantém como string, preserva dígitos

    comp = (parts[1] if parts.shape[1] > 1 else "").fillna("").astype(str).str.strip()
    comp = np.where(comp == "", person, comp)

    credit = pd.to_numeric(work[credit_col], errors="coerce").fillna(0).astype(int)
    debit = pd.to_numeric(work[debit_col], errors="coerce").fillna(0).astype(int)

    out = pd.DataFrame({
        "date": work["date"],
        "account_id": acc_full.astype("string"),
        "company_name": pd.Series(comp).astype(str),
        "credit_un": credit.astype(int),
        "debit_un": debit.astype(int),
    })

    out = out.dropna(subset=["date"]).copy()
    out = out[out["account_id"].astype(str).str.len() > 0].copy()  # precisa ter conta completa

    out["company_key"] = out["company_name"].apply(normalize_company)
    out["total_un"] = (out["credit_un"] + out["debit_un"]).astype(int)

    # chave para join BaaS
    out["account_key"] = out["account_id"].astype(str).map(digits_only)

    return out[["date", "account_id", "account_key", "company_name", "company_key", "credit_un", "debit_un", "total_un"]].copy()


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

    conta_col = _pick_col(df, ["conta", "account"])
    bloqueado_col = _pick_col(df, ["bloquead", "saldo bloqueado", "saldo_bloqueado", "blocked"])

    if not conta_col or not bloqueado_col:
        cols = list(df.columns)
        if len(cols) >= 7:
            conta_col = conta_col or cols[1]
            bloqueado_col = bloqueado_col or cols[6]

    if not conta_col or not bloqueado_col:
        raise ValueError(
            f"CSV BaaS: colunas não identificadas. conta={conta_col}, bloqueado={bloqueado_col}"
        )

    work = df[[conta_col, bloqueado_col]].copy()
    work = work.rename(columns={conta_col: "conta", bloqueado_col: "saldo_bloqueado"})
    work["conta"] = work["conta"].astype(str).str.strip()
    work["conta_key"] = work["conta"].map(digits_only)
    work["saldo_bloqueado"] = pd.to_numeric(work["saldo_bloqueado"], errors="coerce").fillna(0.0).astype(float)
    return work[["conta", "conta_key", "saldo_bloqueado"]]


# =========================
# Week windows
# =========================
def week_windows(day_ref) -> List[Tuple[str, object, object]]:
    dr = pd.to_datetime(day_ref).date()
    out = []
    for i in range(4, 0, -1):
        start = dr - timedelta(days=(i * 7) - 1)
        end = dr - timedelta(days=((i - 1) * 7))
        wname = f"W{5 - i}"
        out.append((wname, start, end))
    return out  # W1..W4


def sum_in_range(df: pd.DataFrame, start, end, col: str) -> int:
    m = (df["date"] >= start) & (df["date"] <= end)
    if df.loc[m].empty:
        return 0
    return int(df.loc[m, col].sum())


# =========================
# Build checklist per account (weeks)
# =========================
def build_checklist_per_account_weeks(
    facts: pd.DataFrame,
    companies_keys: List[str],
    thresholds: Thresholds,
) -> Tuple[str, pd.DataFrame, Dict]:
    if facts.empty:
        return "", pd.DataFrame(), {}

    facts = facts[facts["company_key"].isin(set(companies_keys))].copy()
    if facts.empty:
        return "", pd.DataFrame(), {}

    day_ref = max(facts["date"])
    day_ref = pd.to_datetime(day_ref).date()

    base = (
        facts.groupby(["account_id", "account_key", "company_key", "company_name", "date"], as_index=False)
        .agg(credit_un=("credit_un", "sum"), debit_un=("debit_un", "sum"), total_un=("total_un", "sum"))
    )

    wins = week_windows(day_ref)
    w_meta = {w: (a, b) for (w, a, b) in wins}

    dims = base[["account_id", "account_key", "company_key", "company_name"]].drop_duplicates().copy()
    rows: List[Dict] = []

    for _, d in dims.iterrows():
        account_id = str(d["account_id"])
        account_key = str(d["account_key"])
        company_key = str(d["company_key"])
        company_name = str(d["company_name"])

        scope = base[(base["account_key"].astype(str) == account_key) & (base["company_key"] == company_key)].copy()
        if scope.empty:
            continue

        w_totals = {}
        for (w, a, b) in wins:
            w_totals[w] = {
                "total": sum_in_range(scope, a, b, "total_un"),
                "credit": sum_in_range(scope, a, b, "credit_un"),
                "debit": sum_in_range(scope, a, b, "debit_un"),
                "range": fmt_range(a, b),
            }

        w1 = w_totals["W1"]["total"]
        w2 = w_totals["W2"]["total"]
        w3 = w_totals["W3"]["total"]
        w4 = w_totals["W4"]["total"]

        total_4w = int(w1 + w2 + w3 + w4)
        credit_4w = int(w_totals["W1"]["credit"] + w_totals["W2"]["credit"] + w_totals["W3"]["credit"] + w_totals["W4"]["credit"])
        debit_4w = int(w_totals["W1"]["debit"] + w_totals["W2"]["debit"] + w_totals["W3"]["debit"] + w_totals["W4"]["debit"])

        ref_mean = safe_div((w1 + w2 + w3), 3)
        var_w4 = calc_var(w4, ref_mean)
        var_w4_pct = pct_int_ceil_ratio(var_w4)

        status = calc_status(
            var_w4,
            queda_critica=thresholds.queda_critica,
            aumento_relevante=thresholds.aumento_relevante,
            investigar_abs=thresholds.investigar_abs,
        )
        obs = calc_obs_week(status, total_4w=total_4w, baixo_periodo_limite=thresholds.baixo_periodo_limite)

        rows.append({
            "company_key": company_key,
            "company_name": company_name,
            "account_id": account_id,       # EXIBIÇÃO (completo)
            "account_key": account_key,     # JOIN BaaS
            "day_ref": day_ref.isoformat(),

            "w1": int(w1), "w2": int(w2), "w3": int(w3), "w4": int(w4),
            "w1_range": w_totals["W1"]["range"],
            "w2_range": w_totals["W2"]["range"],
            "w3_range": w_totals["W3"]["range"],
            "w4_range": w_totals["W4"]["range"],

            "credit_4w": int(credit_4w),
            "debit_4w": int(debit_4w),
            "total_4w": int(total_4w),

            "ref_mean_w1w3": float(ref_mean),
            "var_w4": float(var_w4),
            "var_w4_pct": int(var_w4_pct),

            "status": status,
            "obs": obs,
            "severity": severity_rank(status),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return day_ref.isoformat(), df, w_meta

    df = df.sort_values(["severity", "company_name", "account_id"], ascending=[True, True, True]).reset_index(drop=True)
    return day_ref.isoformat(), df, w_meta


# =========================
# Enrich BaaS (por conta) — JOIN POR account_key
# =========================
def enrich_with_baas_accounts(
    df_checklist: pd.DataFrame,
    df_baas: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, float]:
    out = df_checklist.copy()
    out["account_id"] = out["account_id"].astype(str).str.strip()
    out["account_key"] = out["account_key"].astype(str).map(digits_only)

    if df_baas is None or df_baas.empty or out.empty:
        out["saldo_bloqueado_total"] = 0.0
        out["has_block"] = False
        out["lock_icon"] = UNLOCK_ICON if not out.empty else UNKNOWN_ICON
        return out, 0, 0.0

    baas = df_baas.copy()
    baas["conta_key"] = baas["conta_key"].astype(str).map(digits_only)
    baas["saldo_bloqueado"] = pd.to_numeric(baas["saldo_bloqueado"], errors="coerce").fillna(0.0).astype(float)

    baas_agg = (
        baas.groupby("conta_key", as_index=False)["saldo_bloqueado"].sum()
        .rename(columns={"conta_key": "account_key", "saldo_bloqueado": "saldo_bloqueado_total"})
    )

    merged = out.merge(baas_agg, on="account_key", how="left")
    merged["saldo_bloqueado_total"] = pd.to_numeric(merged["saldo_bloqueado_total"], errors="coerce").fillna(0.0).astype(float)

    merged["has_block"] = merged["saldo_bloqueado_total"] > 0
    merged["lock_icon"] = np.where(merged["has_block"], LOCK_ICON, UNLOCK_ICON)

    qtd = int(merged["has_block"].sum())
    soma = float(merged.loc[merged["has_block"], "saldo_bloqueado_total"].sum())
    return merged, qtd, soma


# =========================
# Pagination
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
# UI blocks
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


def render_gauge_and_week_cards(
    title: str,
    w1: int, w2: int, w3: int, w4: int,
    w1_range: str, w2_range: str, w3_range: str, w4_range: str,
    var_w4_pct: int,
    total_4w: int,
    height: int,
):
    if go is None:
        st.warning("plotly não está disponível (instale plotly).")
        return

    gauge_max = max(total_4w * 1.15, 1.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(total_4w),
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
        height=int(height),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.90)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"W1 {w1_range} | W2 {w2_range} | W3 {w3_range} | W4 {w4_range}")

    st.markdown(
        f"""
<div class="mini-grid-6">
  <div class="mini-card">
    <div class="mini-title">Semana 1</div>
    <div class="mini-value">{fmt_int_pt(w1)}</div>
    <div class="mini-sub">{w1_range}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Semana 2</div>
    <div class="mini-value">{fmt_int_pt(w2)}</div>
    <div class="mini-sub">{w2_range}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Semana 3</div>
    <div class="mini-value">{fmt_int_pt(w3)}</div>
    <div class="mini-sub">{w3_range}</div>
  </div>

  <div class="mini-card">
    <div class="mini-title">Semana 4</div>
    <div class="mini-value">{fmt_int_pt(w4)}</div>
    <div class="mini-sub">{w4_range}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Variação (W4 vs média W1–W3)</div>
    <div class="mini-value">{fmt_pct_int(var_w4_pct)}</div>
    <div class="mini-sub">comparação de volume</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Total (4 semanas)</div>
    <div class="mini-value">{fmt_int_pt(total_4w)}</div>
    <div class="mini-sub">W1+W2+W3+W4</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================
# APP UI
# =========================
st.title("Alertas (Checklist por conta) + Bloqueio (BaaS) — Semanas W1..W4")

with st.sidebar:
    st.header("Parâmetros (status)")
    queda_critica = st.number_input("Queda crítica (W4 vs média W1–W3)", value=-0.60, step=0.01, format="%.2f")
    aumento_relevante = st.number_input("Aumento relevante (W4 vs média W1–W3)", value=0.80, step=0.01, format="%.2f")
    investigar_abs = st.number_input("Investigar abs (>= 30%)", value=0.30, step=0.01, format="%.2f")

    st.divider()
    st.header("Baixo volume no período")
    baixo_periodo_limite = st.number_input("Volume muito baixo (<=)", value=1, step=1, min_value=0)

    st.divider()
    st.header("Ground (altura)")
    ground_height = st.select_slider("Altura do Ground", options=[220, 340, 480], value=340)

    st.divider()
    st.header("Paginação")
    page_size_checklist = st.selectbox("Checklist (por página)", [15, 20, 30, 50], index=0)

thresholds = Thresholds(
    queda_critica=float(queda_critica),
    aumento_relevante=float(aumento_relevante),
    investigar_abs=float(investigar_abs),
    baixo_periodo_limite=int(baixo_periodo_limite),
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

    with st.spinner("Lendo CSVs e calculando checklist por conta (W1..W4)..."):
        facts = parse_transactions_csv(trans_file)
        if facts.empty:
            st.error("O CSV de transações ficou vazio após normalização (datas/contas inválidas).")
            st.stop()

        day_ref, df_checklist, w_meta = build_checklist_per_account_weeks(facts, companies_keys, thresholds)

        if df_checklist.empty:
            st.warning("Nenhuma conta encontrada para as empresas filtradas.")
            st.session_state["day_ref"] = day_ref
            st.session_state["df_checklist"] = df_checklist
            st.session_state["w_meta"] = w_meta
        else:
            df_baas = parse_baas_csv(baas_file) if baas_file is not None else pd.DataFrame()
            df_checklist2, kpi_blocked_accounts, kpi_blocked_sum = enrich_with_baas_accounts(df_checklist, df_baas)

            st.session_state["day_ref"] = day_ref
            st.session_state["df_checklist"] = df_checklist2
            st.session_state["w_meta"] = w_meta
            st.session_state["kpi_blocked_accounts"] = kpi_blocked_accounts
            st.session_state["kpi_blocked_sum"] = kpi_blocked_sum
            st.session_state["page_checklist"] = 1

# Render
if "df_checklist" not in st.session_state:
    st.info("Aguardando você preencher as empresas + subir o CSV e clicar em **Processar**.")
    st.stop()

df_checklist = st.session_state.get("df_checklist", pd.DataFrame())
day_ref = st.session_state.get("day_ref", "—")
w_meta = st.session_state.get("w_meta", {})
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

    def _rng(w):
        a, b = w_meta.get(w, (None, None))
        return fmt_range(a, b)

    with left:
        st.caption("Painel (Total geral ou por conta)")

        # ✅ dropdown por CONTA (igual checklist)
        df_checklist["_gkey"] = df_checklist["company_name"].astype(str) + " | Conta " + df_checklist["account_id"].astype(str)
        account_options = sorted(df_checklist["_gkey"].unique().tolist())
        options = ["TOTAL GERAL"] + account_options

        selected = st.selectbox("Selecionar", options, index=0, key="gauge_account_select")

        if selected == "TOTAL GERAL":
            df_scope = df_checklist.copy()
            title = "TOTAL (4 SEMANAS)"
        else:
            df_scope = df_checklist[df_checklist["_gkey"] == selected].copy()
            title = selected

        w1 = int(df_scope["w1"].sum())
        w2 = int(df_scope["w2"].sum())
        w3 = int(df_scope["w3"].sum())
        w4 = int(df_scope["w4"].sum())
        total_4w = int(df_scope["total_4w"].sum())

        ref_mean = safe_div((w1 + w2 + w3), 3)
        var_w4 = calc_var(w4, ref_mean)
        var_w4_pct = pct_int_ceil_ratio(var_w4)

        render_gauge_and_week_cards(
            title=title,
            w1=w1, w2=w2, w3=w3, w4=w4,
            w1_range=_rng("W1"), w2_range=_rng("W2"), w3_range=_rng("W3"), w4_range=_rng("W4"),
            var_w4_pct=var_w4_pct,
            total_4w=total_4w,
            height=int(ground_height),  # ✅ controle de altura
        )

    with right:
        st.caption("Top variação (W4 vs média W1–W3) — piores / melhores")
        topn = st.selectbox("Top", [8, 10, 15], index=0, key="topn_select")

        g = (
            df_checklist.groupby(["company_name"], as_index=False)
            .agg(w1=("w1", "sum"), w2=("w2", "sum"), w3=("w3", "sum"), w4=("w4", "sum"))
        )
        g["ref_mean"] = (g["w1"] + g["w2"] + g["w3"]) / 3.0
        g["var"] = g.apply(lambda r: calc_var(r["w4"], r["ref_mean"]), axis=1)

        worst = g.sort_values("var", ascending=True).head(int(topn))
        best = g.sort_values("var", ascending=False).head(int(topn))
        merged = pd.concat([worst, best], ignore_index=True)
        merged["sign"] = np.where(merged["var"] >= 0, "pos", "neg")

        fig = px.bar(
            merged,
            x="company_name",
            y="var",
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
    q = q_conta.strip()
    view = view[view["account_id"].astype(str).str.contains(q, na=False)]

if status_filter != "Todos":
    view = view[view["status"] == status_filter]

# ✅ Força Conta como STRING (nunca numérica) para não abreviar no Streamlit
view["Conta"] = view["account_id"].astype(str).map(lambda s: s)  # explicitamente string
view["Empresa"] = view["company_name"].astype(str)
view["Status"] = view["status"].map(lambda s: f"{STATUS_DOT.get(s,'⚪')} {s}") + " " + view["lock_icon"].astype(str)

view["Semana 1"] = view["w1"].apply(lambda x: fmt_int_pt(int(x)))
view["Semana 2"] = view["w2"].apply(lambda x: fmt_int_pt(int(x)))
view["Semana 3"] = view["w3"].apply(lambda x: fmt_int_pt(int(x)))
view["Semana 4"] = view["w4"].apply(lambda x: fmt_int_pt(int(x)))

view["Crédito"] = view["credit_4w"].apply(lambda x: fmt_int_pt(int(x)))
view["Débito"] = view["debit_4w"].apply(lambda x: fmt_int_pt(int(x)))
view["Total (4S)"] = view["total_4w"].apply(lambda x: fmt_int_pt(int(x)))
view["Var (W4 vs W1–W3)"] = view["var_w4_pct"].apply(fmt_pct_int)
view["Saldo Bloqueado"] = view["saldo_bloqueado_total"].apply(fmt_money_pt)
view["Motivo"] = view["obs"].astype(str)

show = view[
    ["Status", "Conta", "Empresa", "Semana 1", "Semana 2", "Semana 3", "Semana 4",
     "Crédito", "Débito", "Total (4S)", "Var (W4 vs W1–W3)", "Saldo Bloqueado", "Motivo"]
].copy()

# Paginação
total_rows = int(len(show))
pages = get_pages(total_rows, int(page_size_checklist))

st.session_state.setdefault("page_checklist", 1)
st.session_state["page_checklist"] = max(1, min(pages, int(st.session_state["page_checklist"])))

st.number_input(
    f"Página Checklist (1–{pages})",
    min_value=1,
    max_value=pages,
    value=int(st.session_state["page_checklist"]),
    step=1,
    key="page_checklist",
)

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
                f"W1={r['w1']} | W2={r['w2']} | W3={r['w3']} | W4={r['w4']}\n"
                f"Variação (W4 vs média W1–W3): {r['var_w4_pct']}%\n"
                f"Crédito(4S): {r['credit_4w']} | Débito(4S): {r['debit_4w']} | Total(4S): {r['total_4w']}\n"
                f"Bloqueio: {fmt_money_pt(r.get('saldo_bloqueado_total', 0.0))} {r['lock_icon']}"
            )
            st.code(msg, language="text")