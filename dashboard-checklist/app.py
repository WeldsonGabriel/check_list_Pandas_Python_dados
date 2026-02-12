# app.py
# Streamlit Dashboard — Checklist (por conta) com Semanas 1–4 (quantidade), BaaS embutido, Ground + Cards,
# e 3 abas: Principal | Bolhas (Aquário) | Analytics
#
# Ajustes desta versão:
# ✅ Corrige parsing de CREDIT UN / DEBIT UN quando vem com separador de milhar tipo "448.326"
#    (antes virava 448; agora vira 448326)
# ✅ Mantém todo o resto do app exatamente como estava

from __future__ import annotations

import io
import re
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
# Page config
# =========================
st.set_page_config(page_title="Checklist Semanas + Bloqueio (BaaS)", layout="wide")


# =========================
# Status / cores
# =========================
STATUS_ORDER = ["Alerta (queda/ zerada)", "Investigar", "Gerenciar (aumento)", "Normal"]

STATUS_COLOR = {
    "Alerta (queda/ zerada)": "#e74c3c",  # vermelho
    "Investigar": "#f1c40f",              # amarelo
    "Gerenciar (aumento)": "#3498db",     # azul
    "Normal": "#2ecc71",                  # verde
    "Desconhecido": "#aab4c8",
}

STATUS_DOT = {
    "Alerta (queda/ zerada)": "🔴",
    "Investigar": "🟡",
    "Gerenciar (aumento)": "🔵",
    "Normal": "🟢",
}

LOCK_ICON = "🔒"
UNLOCK_ICON = "🔓"
UNKNOWN_ICON = "⚪"


# =========================
# CSS
# =========================
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
.mini-value { font-size: 22px; font-weight: 800; line-height: 1.1; }

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
# Normalização
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
    s = _strip_accents(s).upper()
    s = s.replace("\t", " ").replace("\n", " ")
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    return re.sub(r"\s+", " ", s).strip()


def split_companies_input(text: str) -> List[str]:
    raw = re.split(r"[\n,]+", text or "")
    out: List[str] = []
    seen: set[str] = set()
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
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


# =========================
# CSV parsing (Transações)
# =========================
def _guess_sep(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    if sample.count(";") >= sample.count(","):
        return ";"
    return ","


def _parse_person(person: str) -> Tuple[str, str]:
    """
    "11717 - PIX NA HORA LTDA" -> ("11717", "PIX NA HORA LTDA")
    """
    s = str(person or "").strip()
    if not s:
        return "", ""
    if " - " in s:
        a, b = s.split(" - ", 1)
        return a.strip(), b.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return a.strip(), b.strip()
    return "", s.strip()


# ✅ BUGFIX: parse correto de quantidade quando vem "448.326" (milhar) / "448,326" etc.
def parse_count(x) -> int:
    if x is None:
        return 0
    s = str(x).strip()
    if not s:
        return 0

    # remove espaços
    s = s.replace(" ", "")

    # padrão do seu CSV: ponto como separador de milhar (sem vírgula)
    if "." in s and "," not in s:
        s = s.replace(".", "")

    # se vier com vírgula por algum motivo, remove também
    s = s.replace(",", "")

    # remove lixo
    s = re.sub(r"[^\d\-]", "", s)

    try:
        return int(s)
    except Exception:
        return 0


def parse_transactions_csv(uploaded_file) -> pd.DataFrame:
    """
    Interpretação:
      - credit_raw e debit_raw representam QUANTIDADE de transações (não valor).
    Saída:
      date (date), account_id (str), company_name (str), company_key (str),
      credit_cnt (int), debit_cnt (int), total_cnt (int)
    """
    if uploaded_file is None:
        return pd.DataFrame()

    b = uploaded_file.getvalue()
    if not b:
        return pd.DataFrame()

    text = b.decode("utf-8", errors="replace")
    sep = _guess_sep(text[:4000])

    df_raw = pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str, header=0)

    if df_raw.shape[1] < 5:
        df_raw = pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str, header=None)
        if df_raw.shape[1] < 5:
            return pd.DataFrame()
        col_date, col_person, col_credit, col_debit = 0, 2, 3, 4
        date_raw = df_raw.iloc[:, col_date].astype(str)
        person_raw = df_raw.iloc[:, col_person].astype(str)
        credit_raw = df_raw.iloc[:, col_credit]
        debit_raw = df_raw.iloc[:, col_debit]
    else:
        date_raw = df_raw.iloc[:, 0].astype(str)
        person_raw = df_raw.iloc[:, 2].astype(str)
        credit_raw = df_raw.iloc[:, 3]
        debit_raw = df_raw.iloc[:, 4]

    dates = pd.to_datetime(date_raw, errors="coerce", dayfirst=True)
    d = dates.dt.date

    acc_emp = person_raw.apply(_parse_person)
    account_id = acc_emp.apply(lambda t: t[0]).astype(str).str.strip()
    company_name = acc_emp.apply(lambda t: t[1]).astype(str).str.strip()
    company_key = company_name.map(normalize_company)

    # ✅ FIX AQUI (antes era to_numeric + astype(int))
    credit_cnt = credit_raw.apply(parse_count)
    debit_cnt = debit_raw.apply(parse_count)

    out = pd.DataFrame({
        "date": d,
        "account_id": account_id,
        "company_name": company_name,
        "company_key": company_key,
        "credit_cnt": credit_cnt,
        "debit_cnt": debit_cnt,
    })
    out = out.dropna(subset=["date"]).copy()
    out = out[out["company_key"].astype(str).str.len() > 0].copy()
    out["account_id"] = out["account_id"].astype(str)
    out = out[out["account_id"].str.len() > 0].copy()

    out["total_cnt"] = out["credit_cnt"] + out["debit_cnt"]
    return out


# =========================
# CSV parsing (BaaS)
# =========================
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for key in candidates:
        for lc, orig in low.items():
            if key in lc:
                return orig
    return None


def parse_baas_csv(uploaded_file) -> pd.DataFrame:
    """
    Saída: conta (str), saldo_bloqueado (float)
    """
    if uploaded_file is None:
        return pd.DataFrame()

    b = uploaded_file.getvalue()
    if not b:
        return pd.DataFrame()

    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python", dtype=str)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=";", engine="python", dtype=str)

    if df is None or df.empty:
        return pd.DataFrame()

    conta_col = _pick_col(df, ["conta", "account"])
    bloq_col = _pick_col(df, ["saldo bloque", "saldo_bloque", "bloquead", "blocked"])

    if not conta_col or not bloq_col:
        if df.shape[1] >= 7:
            conta_col = conta_col or df.columns[1]
            bloq_col = bloq_col or df.columns[6]
        else:
            return pd.DataFrame()

    out = pd.DataFrame()
    out["conta"] = df[conta_col].fillna("").astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    out["saldo_bloqueado"] = pd.to_numeric(df[bloq_col], errors="coerce").fillna(0.0).astype(float)
    out = out[out["conta"].str.len() > 0].copy()
    out = out.groupby("conta", as_index=False)["saldo_bloqueado"].sum()
    return out


# =========================
# Janela de semanas (28d terminando em D-1)
# =========================
def compute_week_ranges(day_ref: pd.Timestamp) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    d1 = (pd.to_datetime(day_ref) - timedelta(days=1)).normalize()
    start = d1 - timedelta(days=27)
    w1 = (start, start + timedelta(days=6))
    w2 = (start + timedelta(days=7), start + timedelta(days=13))
    w3 = (start + timedelta(days=14), start + timedelta(days=20))
    w4 = (start + timedelta(days=21), d1)
    return {"w1": w1, "w2": w2, "w3": w3, "w4": w4}


# =========================
# Status por volume semanal
# =========================
def calc_status_volume(
    week4: int,
    avg_prev: float,
    total_4w: int,
    *,
    thresholds: Thresholds,
    low_volume_threshold: int,
) -> Tuple[str, str, float]:
    if total_4w <= 0:
        return "Alerta (queda/ zerada)", "Conta zerada no período", 0.0

    if total_4w <= int(low_volume_threshold):
        return "Investigar", "Volume muito baixo no período", 0.0

    if avg_prev <= 0:
        return "Normal", "Dentro do esperado", 0.0

    var = (float(week4) - float(avg_prev)) / float(avg_prev)

    if var <= thresholds.queda_critica:
        return "Alerta (queda/ zerada)", "Queda crítica vs período anterior", var

    if var >= thresholds.investigar_abs:
        return "Gerenciar (aumento)", "Aumento relevante vs período anterior", var

    if var <= -thresholds.investigar_abs:
        return "Investigar", "Queda relevante vs período anterior", var

    return "Normal", "Dentro do esperado", var


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


# =========================
# Build checklist (por conta)
# =========================
def build_checklist_weeks(
    facts: pd.DataFrame,
    companies_keys: List[str],
    thresholds: Thresholds,
    low_volume_threshold: int,
) -> Tuple[str, pd.DataFrame, Dict[str, str]]:
    if facts is None or facts.empty:
        return "", pd.DataFrame(), {}

    facts = facts[facts["company_key"].isin(set(companies_keys))].copy()
    if facts.empty:
        return "", pd.DataFrame(), {}

    day_ref = pd.to_datetime(facts["date"].max())
    ranges = compute_week_ranges(day_ref)

    labels = {
        "w1": f"{ranges['w1'][0].date().isoformat()} → {ranges['w1'][1].date().isoformat()}",
        "w2": f"{ranges['w2'][0].date().isoformat()} → {ranges['w2'][1].date().isoformat()}",
        "w3": f"{ranges['w3'][0].date().isoformat()} → {ranges['w3'][1].date().isoformat()}",
        "w4": f"{ranges['w4'][0].date().isoformat()} → {ranges['w4'][1].date().isoformat()}",
    }

    facts["date_dt"] = pd.to_datetime(facts["date"]).dt.normalize()

    def sum_week(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        m = (df["date_dt"] >= start) & (df["date_dt"] <= end)
        return df.loc[m].groupby(["account_id", "company_key"], as_index=False)["total_cnt"].sum().set_index(["account_id", "company_key"])["total_cnt"]

    dims = facts[["account_id", "company_key", "company_name"]].drop_duplicates().copy()
    dims["account_id"] = dims["account_id"].astype(str)

    s_w1 = sum_week(facts, ranges["w1"][0], ranges["w1"][1])
    s_w2 = sum_week(facts, ranges["w2"][0], ranges["w2"][1])
    s_w3 = sum_week(facts, ranges["w3"][0], ranges["w3"][1])
    s_w4 = sum_week(facts, ranges["w4"][0], ranges["w4"][1])

    start_all = ranges["w1"][0]
    end_all = ranges["w4"][1]
    m_all = (facts["date_dt"] >= start_all) & (facts["date_dt"] <= end_all)
    period = facts.loc[m_all].groupby(["account_id", "company_key"], as_index=False).agg(
        credit=("credit_cnt", "sum"),
        debit=("debit_cnt", "sum"),
        total_4w=("total_cnt", "sum"),
    ).set_index(["account_id", "company_key"])

    rows: List[Dict] = []
    for _, r in dims.iterrows():
        acc = str(r["account_id"])
        ck = str(r["company_key"])
        name = str(r["company_name"])

        key = (acc, ck)
        w1 = int(s_w1.get(key, 0))
        w2 = int(s_w2.get(key, 0))
        w3 = int(s_w3.get(key, 0))
        w4 = int(s_w4.get(key, 0))

        credit = int(period.loc[key, "credit"]) if key in period.index else 0
        debit = int(period.loc[key, "debit"]) if key in period.index else 0
        total_4w = int(period.loc[key, "total_4w"]) if key in period.index else (w1 + w2 + w3 + w4)

        avg_prev = (w1 + w2 + w3) / 3.0
        status, motivo, var = calc_status_volume(
            week4=w4,
            avg_prev=avg_prev,
            total_4w=total_4w,
            thresholds=thresholds,
            low_volume_threshold=low_volume_threshold,
        )

        rows.append({
            "company_key": ck,
            "company_name": name,
            "account_id": acc,
            "day_ref": day_ref.date().isoformat(),

            "week1": w1,
            "week2": w2,
            "week3": w3,
            "week4": w4,

            "credit": credit,
            "debit": debit,
            "total_4w": total_4w,

            "avg_prev_weeks": float(avg_prev),
            "var_week4_vs_prev": float(var),

            "status": status,
            "motivo": motivo,
            "severity": severity_rank(status),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return day_ref.date().isoformat(), df, labels

    df = df[df["total_4w"] > 0].copy()
    df = df.sort_values(["severity", "company_name", "account_id"], ascending=[True, True, True]).reset_index(drop=True)
    return day_ref.date().isoformat(), df, labels


# =========================
# Enrich BaaS (por conta)
# =========================
def enrich_baas(df_checklist: pd.DataFrame, df_baas: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    out = df_checklist.copy()
    if out.empty:
        return out, 0, 0.0

    out["account_id"] = out["account_id"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)

    if df_baas is None or df_baas.empty:
        out["saldo_bloqueado_total"] = 0.0
        out["has_block"] = False
        out["lock_icon"] = UNLOCK_ICON
        return out, 0, 0.0

    baas = df_baas.copy()
    baas["conta"] = baas["conta"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    baas["saldo_bloqueado"] = pd.to_numeric(baas["saldo_bloqueado"], errors="coerce").fillna(0.0).astype(float)
    baas = baas.rename(columns={"conta": "account_id", "saldo_bloqueado": "saldo_bloqueado_total"})

    merged = out.merge(baas, on="account_id", how="left")
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
# UI helpers
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


def render_ground_panel(
    title: str,
    w1: int, w2: int, w3: int, w4: int,
    credit: int, debit: int, total_4w: int,
    var: float,
    height: int,
    labels: Dict[str, str],
):
    if go is None:
        st.warning("plotly não está disponível (instale plotly).")
        return

    max_ref = max(total_4w, w1, w2, w3, w4, 1)
    gauge_max = max_ref * 1.25

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(total_4w),
            number={"valueformat": ".0f"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": "rgba(255,255,255,0.85)"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [0, gauge_max], "color": "rgba(255,255,255,0.08)"}],
            },
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.90)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    var_pct = int(math.ceil(var * 100.0)) if np.isfinite(var) else 0
    st.markdown(
        f"""
<div class="mini-grid">
  <div class="mini-card">
    <div class="mini-title">Semana 1 ({labels.get("w1","")})</div>
    <div class="mini-value">{fmt_int_pt(w1)}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Semana 2 ({labels.get("w2","")})</div>
    <div class="mini-value">{fmt_int_pt(w2)}</div>
  </div>

  <div class="mini-card">
    <div class="mini-title">Semana 3 ({labels.get("w3","")})</div>
    <div class="mini-value">{fmt_int_pt(w3)}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Semana 4 ({labels.get("w4","")})</div>
    <div class="mini-value">{fmt_int_pt(w4)}</div>
  </div>

  <div class="mini-card">
    <div class="mini-title">Crédito (Qtd)</div>
    <div class="mini-value">{fmt_int_pt(credit)}</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Débito (Qtd)</div>
    <div class="mini-value">{fmt_int_pt(debit)}</div>
  </div>

  <div class="mini-card">
    <div class="mini-title">Variação (S4 vs média S1–S3)</div>
    <div class="mini-value">{var_pct}%</div>
  </div>
  <div class="mini-card">
    <div class="mini-title">Total 4 Semanas</div>
    <div class="mini-value">{fmt_int_pt(total_4w)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================
# Bubble aquarium (D3) — agora com "bubble_scale"
# =========================
def render_bubble_aquarium(df_nodes: pd.DataFrame, height: int = 700, bubble_scale: float = 1.6):
    if df_nodes is None or df_nodes.empty:
        st.info("Sem dados para bolhas.")
        return

    df_nodes = df_nodes.copy()

    # evita travar com muitos nós
    if len(df_nodes) > 600:
        df_nodes = df_nodes.nlargest(600, "value")

    df_nodes["value"] = pd.to_numeric(df_nodes["value"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if df_nodes["value"].max() <= 0:
        df_nodes["value"] = 1.0

    nodes = df_nodes.to_dict(orient="records")
    nodes_json = json.dumps(nodes, ensure_ascii=False)

    colors_json = json.dumps(STATUS_COLOR, ensure_ascii=False)
    bubble_scale = float(bubble_scale)

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  html, body {{ margin:0; padding:0; background:transparent; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }}
  #wrap {{
    width: 100%;
    height: {height}px;
    position: relative;
    border-radius: 16px;
    overflow: hidden;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08);
  }}
  .tooltip {{
    position: absolute;
    pointer-events: none;
    opacity: 0;
    background: rgba(15, 15, 18, 0.96);
    color: #fff;
    padding: 10px 12px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 10px 30px rgba(0,0,0,.35);
    font-size: 12px;
    max-width: 420px;
    line-height: 1.35;
    z-index: 9999;
  }}
  .t-title {{ font-weight: 800; font-size: 13px; margin-bottom: 6px; }}
  .t-row {{ opacity: .92; }}
  .legend {{
    position:absolute;
    left: 12px;
    bottom: 12px;
    display:flex;
    gap: 10px;
    flex-wrap:wrap;
    background: rgba(0,0,0,0.20);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 8px 10px;
    border-radius: 14px;
    backdrop-filter: blur(6px);
  }}
  .leg-item {{ display:flex; align-items:center; gap:6px; font-size: 12px; opacity:.92; }}
  .dot {{ width:10px; height:10px; border-radius:999px; }}
  .hint {{
    position:absolute;
    right:12px;
    top:12px;
    background: rgba(0,0,0,0.20);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 8px 10px;
    border-radius: 14px;
    font-size: 12px;
    opacity:.92;
    backdrop-filter: blur(6px);
  }}
</style>
</head>
<body>
<div id="wrap"></div>
<div class="tooltip" id="tooltip"></div>

<script>
  const nodes = {nodes_json};
  const COLORS = {colors_json};
  const SCALE = {bubble_scale};

  const wrap = document.getElementById("wrap");
  const tooltip = document.getElementById("tooltip");

  const W = wrap.clientWidth;
  const H = wrap.clientHeight;
  const minSide = Math.max(240, Math.min(W, H));

  const svg = d3.select("#wrap").append("svg")
    .attr("width", W)
    .attr("height", H);

  // Raio: baseado no tamanho do aquário -> bolhas maiores e ocupando melhor
  const valueExtent = d3.extent(nodes, d => +d.value);

  const baseMin = Math.max(14, minSide * 0.035);
  const baseMax = Math.max(46, minSide * 0.16);

  const rMin = Math.min(40, baseMin * SCALE);
  const rMax = Math.min(120, baseMax * SCALE);

  const rScale = d3.scaleSqrt()
    .domain(valueExtent[0] === valueExtent[1] ? [0, valueExtent[1] || 1] : valueExtent)
    .range([rMin, rMax]);

  const padding = 2;

  nodes.forEach(d => {{
    d.r = rScale(+d.value || 0);
    d.x = W/2 + (Math.random() - 0.5) * 80;
    d.y = H/2 + (Math.random() - 0.5) * 80;
  }});

  const circles = svg.selectAll("circle")
    .data(nodes, d => d.id)
    .join("circle")
    .attr("r", d => d.r)
    .attr("cx", d => d.x)
    .attr("cy", d => d.y)
    .attr("fill", d => COLORS[d.status] || "#aab4c8")
    .attr("fill-opacity", 0.80)
    .attr("stroke", d => d.blocked ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.25)")
    .attr("stroke-width", d => d.blocked ? 2.2 : 1.0);

  circles
    .on("mousemove", (event, d) => {{
      const b = d.blocked ? "🔒 BLOQUEIO" : "🔓 sem bloqueio";
      tooltip.innerHTML = `
        <div class="t-title">${{d.company}} | Conta ${{d.label}}</div>
        <div class="t-row">Status: <b>${{d.status}}</b></div>
        <div class="t-row">Total 4 semanas: <b>${{d.value}}</b></div>
        <div class="t-row">Semana 1: <b>${{d.w1}}</b> | Semana 2: <b>${{d.w2}}</b></div>
        <div class="t-row">Semana 3: <b>${{d.w3}}</b> | Semana 4: <b>${{d.w4}}</b></div>
        <div class="t-row">Bloqueio: <b>${{b}}</b> (${{ d.blocked_value != null ? d.blocked_value : 0 }})</div>
      `;
      tooltip.style.opacity = 1;
      tooltip.style.left = (event.pageX + 14) + "px";
      tooltip.style.top = (event.pageY + 14) + "px";

      d3.select(event.currentTarget)
        .attr("fill-opacity", 0.95)
        .attr("stroke", "rgba(255,255,255,0.95)")
        .attr("stroke-width", 2.6);
    }})
    .on("mouseleave", (event, d) => {{
      tooltip.style.opacity = 0;
      d3.select(event.currentTarget)
        .attr("fill-opacity", 0.80)
        .attr("stroke", d.blocked ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.25)")
        .attr("stroke-width", d.blocked ? 2.2 : 1.0);
    }});

  // Forças ajustadas: menos repulsão + colisão forte => “preenche” melhor o aquário
  const sim = d3.forceSimulation(nodes)
    .velocityDecay(0.18)
    .force("charge", d3.forceManyBody().strength(-6))
    .force("center", d3.forceCenter(W/2, H/2))
    .force("collision", d3.forceCollide().radius(d => d.r + 3).iterations(3))
    .force("x", d3.forceX(W/2).strength(0.06))
    .force("y", d3.forceY(H/2).strength(0.06));

  // Clamp para garantir que ficam "certinho dentro do aquário"
  sim.on("tick", () => {{
    nodes.forEach(d => {{
      d.x = Math.max(d.r + padding, Math.min(W - d.r - padding, d.x));
      d.y = Math.max(d.r + padding, Math.min(H - d.r - padding, d.y));
    }});
    circles.attr("cx", d => d.x).attr("cy", d => d.y);
  }});

  const legend = d3.select("#wrap").append("div").attr("class","legend");
  const items = [
    ["Alerta (queda/ zerada)", COLORS["Alerta (queda/ zerada)"]],
    ["Investigar", COLORS["Investigar"]],
    ["Gerenciar (aumento)", COLORS["Gerenciar (aumento)"]],
    ["Normal", COLORS["Normal"]],
  ];
  items.forEach(([name, c]) => {{
    const it = legend.append("div").attr("class","leg-item");
    it.append("div").attr("class","dot").style("background", c);
    it.append("div").text(name);
  }});

  d3.select("#wrap").append("div")
    .attr("class","hint")
    .html("Passe o mouse nas bolhas • Tamanho = Total 4 semanas • Borda branca = bloqueio");
</script>
</body>
</html>
"""
    components.html(html, height=height + 12, scrolling=False)


# =========================
# App
# =========================
st.title("Checklist (Semanas 1–4) + Bloqueio (BaaS)")

with st.sidebar:
    st.header("Parâmetros (Status)")
    queda_critica = st.number_input("Queda crítica (S4 vs média S1–S3)", value=-0.60, step=0.01, format="%.2f")
    aumento_relevante = st.number_input("Aumento relevante (não essencial)", value=0.80, step=0.01, format="%.2f")
    investigar_abs = st.number_input("Faixa normal (±)", value=0.30, step=0.01, format="%.2f")

    st.divider()
    st.header("Volume muito baixo")
    low_volume_threshold = st.number_input("Limite (Total 4 semanas)", value=5, step=1, min_value=0)

    st.divider()
    st.header("Ground (tamanho)")
    ground_height = st.slider("Altura do Ground", min_value=220, max_value=480, value=340, step=20)

    st.divider()
    st.header("Paginação")
    page_size = st.selectbox("Checklist (por página)", [15, 20, 30, 50], index=0)

thresholds = Thresholds(
    queda_critica=float(queda_critica),
    aumento_relevante=float(aumento_relevante),
    investigar_abs=float(investigar_abs),
)

tab_main, tab_bubbles, tab_analytics = st.tabs(["Principal", "Bolhas (Aquário)", "Analytics"])

# =========================
# PRINCIPAL
# =========================
with tab_main:
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

        with st.spinner("Processando..."):
            facts = parse_transactions_csv(trans_file)
            if facts.empty:
                st.error("CSV de transações ficou vazio após leitura/normalização.")
                st.stop()

            day_ref, df_checklist, week_labels = build_checklist_weeks(
                facts=facts,
                companies_keys=companies_keys,
                thresholds=thresholds,
                low_volume_threshold=int(low_volume_threshold),
            )

            df_baas = parse_baas_csv(baas_file) if baas_file is not None else pd.DataFrame()
            df_final, kpi_blocked_accounts, kpi_blocked_sum = enrich_baas(df_checklist, df_baas)

            st.session_state["day_ref"] = day_ref
            st.session_state["week_labels"] = week_labels
            st.session_state["df_final"] = df_final
            st.session_state["kpi_blocked_accounts"] = int(kpi_blocked_accounts)
            st.session_state["kpi_blocked_sum"] = float(kpi_blocked_sum)
            st.session_state["page"] = 1

    if "df_final" not in st.session_state:
        st.info("Preencha empresas + suba CSV + clique em **Processar**.")
        st.stop()

    df_final = st.session_state.get("df_final", pd.DataFrame())
    week_labels = st.session_state.get("week_labels", {})
    day_ref = st.session_state.get("day_ref", "—")
    kpi_blocked_accounts = int(st.session_state.get("kpi_blocked_accounts", 0))

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    with k1:
        st.metric("Dia de referência", day_ref)
    with k2:
        st.metric("Empresas filtradas", fmt_int_pt(int(df_final["company_key"].nunique()) if not df_final.empty else 0))
    with k3:
        st.metric("Alertas", fmt_int_pt(int((df_final["status"] != "Normal").sum()) if not df_final.empty else 0))
    with k4:
        st.metric("Alertas críticos", fmt_int_pt(int((df_final["status"] == "Alerta (queda/ zerada)").sum()) if not df_final.empty else 0))
    with k5:
        st.metric("Contas com bloqueio", fmt_int_pt(kpi_blocked_accounts))

    st.divider()

    st.subheader("Visão Geral")
    render_status_boxes(df_final)
    st.divider()

    if px is None or go is None:
        st.warning("Instale plotly para ver o Ground e gráficos.")
    elif df_final.empty:
        st.info("Sem dados.")
    else:
        left, right = st.columns([1.1, 1.0], gap="large")

        df_final["select_key"] = df_final["company_name"].astype(str) + " | " + df_final["account_id"].astype(str)
        options = ["TOTAL GERAL"] + sorted(df_final["select_key"].unique().tolist())

        with left:
            st.caption("Ground (Total geral ou por Conta)")
            selected = st.selectbox("Selecionar", options, index=0, key="ground_select")

            if selected == "TOTAL GERAL":
                scope = df_final.copy()
                title = "TOTAL 4 SEMANAS (GERAL)"
                w1 = int(scope["week1"].sum())
                w2 = int(scope["week2"].sum())
                w3 = int(scope["week3"].sum())
                w4 = int(scope["week4"].sum())
                credit = int(scope["credit"].sum())
                debit = int(scope["debit"].sum())
                total_4w = int(scope["total_4w"].sum())
                avg_prev = (w1 + w2 + w3) / 3.0 if (w1 + w2 + w3) > 0 else 0.0
                var = (w4 - avg_prev) / avg_prev if avg_prev > 0 else 0.0
            else:
                scope = df_final[df_final["select_key"] == selected].copy()
                r = scope.iloc[0]
                title = selected
                w1, w2, w3, w4 = int(r["week1"]), int(r["week2"]), int(r["week3"]), int(r["week4"])
                credit, debit = int(r["credit"]), int(r["debit"])
                total_4w = int(r["total_4w"])
                var = float(r.get("var_week4_vs_prev", 0.0))

            render_ground_panel(
                title=title,
                w1=w1, w2=w2, w3=w3, w4=w4,
                credit=credit, debit=debit, total_4w=total_4w,
                var=var,
                height=int(ground_height),
                labels=week_labels,
            )

        with right:
            st.caption("Top variação (S4 vs média S1–S3) — piores / melhores")
            topn = st.selectbox("Top", [8, 10, 15], index=0, key="topn_select")

            g = df_final.groupby(["company_name"], as_index=False).agg(
                w1=("week1", "sum"),
                w2=("week2", "sum"),
                w3=("week3", "sum"),
                w4=("week4", "sum"),
            )
            g["avg_prev"] = (g["w1"] + g["w2"] + g["w3"]) / 3.0
            g["var"] = np.where(g["avg_prev"] > 0, (g["w4"] - g["avg_prev"]) / g["avg_prev"], 0.0)

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

    st.subheader("Checklist (por conta)")

    if df_final.empty:
        st.info("Sem dados para mostrar.")
        st.stop()

    f1, f2, f3 = st.columns([1.2, 1.0, 0.8], gap="small")
    with f1:
        q_empresa = st.text_input("Buscar empresa", value="", key="q_empresa")
    with f2:
        q_conta = st.text_input("Buscar conta", value="", key="q_conta")
    with f3:
        status_filter = st.selectbox("Status", ["Todos"] + STATUS_ORDER, index=0, key="status_filter")

    view = df_final.copy()

    if q_empresa.strip():
        q = q_empresa.strip().lower()
        view = view[view["company_name"].astype(str).str.lower().str.contains(q, na=False)]

    if q_conta.strip():
        q = re.sub(r"\D+", "", q_conta.strip())
        if q:
            view = view[view["account_id"].astype(str).str.contains(q, na=False)]

    if status_filter != "Todos":
        view = view[view["status"] == status_filter].copy()

    view["Status"] = view["status"].map(lambda s: STATUS_DOT.get(str(s), "⚪")) + " " + view["status"].astype(str) + " " + view["lock_icon"].astype(str)

    show = pd.DataFrame({
        "Status": view["Status"],
        "Conta": view["account_id"].astype(str),
        "Empresa": view["company_name"].astype(str),
        "Semana 1": view["week1"].astype(int).map(fmt_int_pt),
        "Semana 2": view["week2"].astype(int).map(fmt_int_pt),
        "Semana 3": view["week3"].astype(int).map(fmt_int_pt),
        "Semana 4": view["week4"].astype(int).map(fmt_int_pt),
        "Crédito (Qtd)": view["credit"].astype(int).map(fmt_int_pt),
        "Débito (Qtd)": view["debit"].astype(int).map(fmt_int_pt),
        "Total 4 Semanas": view["total_4w"].astype(int).map(fmt_int_pt),
        "Saldo Bloqueado": view["saldo_bloqueado_total"].map(fmt_money_pt),
        "Motivo": view["motivo"].astype(str),
    })

    st.caption(
        f"Semana 1: {week_labels.get('w1','')} | Semana 2: {week_labels.get('w2','')} | "
        f"Semana 3: {week_labels.get('w3','')} | Semana 4: {week_labels.get('w4','')}"
    )

    total_rows = int(len(show))
    pages = get_pages(total_rows, int(page_size))

    st.session_state.setdefault("page", 1)
    st.session_state["page"] = clamp_int(int(st.session_state["page"]), 1, pages)

    st.number_input(
        f"Página Checklist (1–{pages})",
        min_value=1,
        max_value=pages,
        value=int(st.session_state["page"]),
        step=1,
        key="page",
    )

    page_now = clamp_int(int(st.session_state["page"]), 1, pages)
    page_df = slice_page(show, page_now, int(page_size))
    st.dataframe(page_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Cards (para colar no Discord)")
    alerts = df_final[df_final["status"] != "Normal"].copy()

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
                    f"Motivo: {r['motivo']}\n"
                    f"Semanas: S1={r['week1']} | S2={r['week2']} | S3={r['week3']} | S4={r['week4']}\n"
                    f"Crédito(Qtd): {r['credit']} | Débito(Qtd): {r['debit']}\n"
                    f"Total 4 semanas: {r['total_4w']}\n"
                    f"Bloqueio: {fmt_money_pt(r.get('saldo_bloqueado_total', 0.0))} {r['lock_icon']}"
                )
                st.code(msg, language="text")


# =========================
# BOLHAS
# =========================
with tab_bubbles:
    if "df_final" not in st.session_state:
        st.info("Você precisa processar na aba Principal primeiro.")
        st.stop()

    df_final = st.session_state.get("df_final", pd.DataFrame())
    if df_final.empty:
        st.info("Sem dados para bolhas.")
        st.stop()

    st.subheader("Bolhas (Aquário)")
    st.caption("Tamanho = Total 4 semanas | Cores = status | Borda branca = bloqueio | Hover mostra detalhes")

    df_nodes = pd.DataFrame({
        "id": df_final["account_id"].astype(str),
        "label": df_final["account_id"].astype(str),
        "company": df_final["company_name"].astype(str),
        "status": df_final["status"].astype(str),
        "value": df_final["total_4w"].astype(float),
        "blocked": df_final["has_block"].fillna(False).astype(bool),
        "blocked_value": df_final["saldo_bloqueado_total"].fillna(0.0).astype(float),
        "w1": df_final["week1"].astype(int),
        "w2": df_final["week2"].astype(int),
        "w3": df_final["week3"].astype(int),
        "w4": df_final["week4"].astype(int),
    })

    colA, colB = st.columns([1, 1], gap="small")
    with colA:
        bubble_height = st.slider("Altura do aquário", 520, 900, 700, 20)
    with colB:
        bubble_scale = st.slider("Tamanho das bolhas", 20.0, 41.5, 5.6, 0.1)

    render_bubble_aquarium(df_nodes, height=int(bubble_height), bubble_scale=float(bubble_scale))


# =========================
# ANALYTICS
# =========================
with tab_analytics:
    if "df_final" not in st.session_state:
        st.info("Você precisa processar na aba Principal primeiro.")
        st.stop()

    df_final = st.session_state.get("df_final", pd.DataFrame())
    if df_final.empty or px is None:
        st.info("Sem dados (ou plotly não instalado).")
        st.stop()

    st.subheader("Analytics")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.caption("Distribuição por status (contas)")
        sc = df_final.groupby("status", as_index=False).size().rename(columns={"size": "count"})
        sc["ord"] = sc["status"].apply(lambda s: STATUS_ORDER.index(s) if s in STATUS_ORDER else 999)
        sc = sc.sort_values("ord")
        fig = px.bar(sc, x="status", y="count", color="status", color_discrete_map=STATUS_COLOR)
        fig.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.caption("Top Empresas por Volume (Total 4 semanas) — cor = pior status da empresa")
        top_emp = st.slider("Top empresas (por volume total)", 5, 40, 15, 1, key="top_emp_vol")

        def worst_status(series: pd.Series) -> str:
            ranks = series.map(severity_rank)
            i = int(ranks.min()) if len(ranks) else 9
            for s in STATUS_ORDER:
                if severity_rank(s) == i:
                    return s
            return "Desconhecido"

        ge = df_final.groupby("company_name", as_index=False).agg(
            total=("total_4w", "sum"),
            worst_status=("status", worst_status),
            blocked_accounts=("has_block", "sum"),
        ).sort_values("total", ascending=False).head(int(top_emp))

        fig = px.bar(
            ge.sort_values("total", ascending=True),
            x="total",
            y="company_name",
            orientation="h",
            color="worst_status",
            color_discrete_map=STATUS_COLOR,
            hover_data={"blocked_accounts": True, "total": True, "worst_status": True},
        )
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Total (4 semanas) — Qtd transações",
            yaxis_title=None,
            legend_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.caption("Evolução semanal (soma por empresa)")
    top_companies = st.slider("Top empresas (por volume)", 5, 40, 15, 1, key="top_emp_line")

    g = df_final.groupby("company_name", as_index=False).agg(
        w1=("week1", "sum"), w2=("week2", "sum"), w3=("week3", "sum"), w4=("week4", "sum"),
        total=("total_4w", "sum"),
    ).sort_values("total", ascending=False).head(int(top_companies))

    melt = g.melt(id_vars=["company_name"], value_vars=["w1", "w2", "w3", "w4"], var_name="week", value_name="qty")
    week_name = {"w1": "Semana 1", "w2": "Semana 2", "w3": "Semana 3", "w4": "Semana 4"}
    melt["week"] = melt["week"].map(week_name)

    fig = px.line(melt, x="week", y="qty", color="company_name", markers=True)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title="Qtd transações")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.caption("Motivos (top)")
    mc = df_final.groupby("motivo", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
    fig = px.bar(mc.head(12), x="motivo", y="count")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)