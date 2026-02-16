# app.py
# Streamlit Dashboard — Checklist (por conta) com Semanas 1–4 (quantidade), BaaS embutido, Ground + Cards
#
# ✅ Mantém toda a base/estilo do app
# ✅ Corrige parsing de CREDIT UN / DEBIT UN quando vem com separador de milhar tipo "448.326"
# ✅ Remove D-1: semanas terminam no ÚLTIMO DIA real do CSV (day_ref)
#
# ✅ NOVO (sem LLM / “IA grátis” via regras + estatística):
# - Remove abas "Bolhas" e "Analytics" (menos poluição)
# - Mantém "Principal"
# - Adiciona 2 abas novas:
#   1) "Alertas" (cards/farol + filtros, do pior pro melhor)
#   2) "Análise (Diário)" (tabela dia 1..N + “assistente” de explicação por conta/empresa)
# - BaaS: além de "Saldo Bloqueado", também lê "Saldo" (se existir no CSV)
#
# Observação:
# - O “assistente” aqui NÃO usa LLM: ele só calcula métricas e gera explicação determinística.

from __future__ import annotations

import io
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
# Page config
# =========================
st.set_page_config(page_title="Checklist Semanas + BaaS", layout="wide")


# =========================
# Status / cores (Checklist semanal)
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

.badges { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
.badge {
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.03);
  opacity: .95;
}
.badge strong { font-weight: 800; }
.b-red { border-color: rgba(231,76,60,0.45); background: rgba(231,76,60,0.10); }
.b-orange { border-color: rgba(230,126,34,0.50); background: rgba(230,126,34,0.12); }
.b-yellow { border-color: rgba(241,196,15,0.45); background: rgba(241,196,15,0.12); }
.b-green { border-color: rgba(46,204,113,0.45); background: rgba(46,204,113,0.10); }
.b-gray { border-color: rgba(170,180,200,0.35); background: rgba(170,180,200,0.08); }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Thresholds (Checklist semanal)
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


def parse_count(x) -> int:
    """BUGFIX: parse correto de quantidade quando vem "448.326" (milhar) / "448,326" etc."""
    if x is None:
        return 0
    s = str(x).strip()
    if not s:
        return 0
    s = s.replace(" ", "")
    if "." in s and "," not in s:
        s = s.replace(".", "")
    s = s.replace(",", "")
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
# CSV parsing (BaaS / Saldos)
# =========================
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for key in candidates:
        for lc, orig in low.items():
            if key in lc:
                return orig
    return None


def _parse_money_pt(x) -> float:
    """
    Aceita:
      - "370264,91"
      - "370.264,91"
      - "370264.91"
      - "370264"
    """
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    s = s.replace("R$", "").replace(" ", "")
    # se tiver vírgula, assume decimal pt-BR
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    # senão, só remove milhares " " etc
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0


def parse_baas_csv(uploaded_file) -> pd.DataFrame:
    """
    Saída: account_id (str), saldo_total (float), saldo_bloqueado_total (float)

    Compatível com o seu header (print):
      Conta | ... | Saldo | Saldo Bloqueado | ...
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
    saldo_col = _pick_col(df, ["saldo"])
    bloq_col = _pick_col(df, ["saldo bloque", "saldo_bloque", "bloquead", "blocked"])

    # fallback por posição (se vier diferente)
    if not conta_col and df.shape[1] >= 2:
        conta_col = df.columns[1]
    if not saldo_col and df.shape[1] >= 6:
        # no seu print, "Saldo" é coluna E (index 4) mas pode variar; tenta achar por proximidade
        saldo_col = df.columns[min(4, df.shape[1] - 1)]
    if not bloq_col and df.shape[1] >= 7:
        bloq_col = df.columns[min(5, df.shape[1] - 1)]

    if not conta_col:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["account_id"] = df[conta_col].fillna("").astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    out["saldo_total"] = df[saldo_col].apply(_parse_money_pt) if saldo_col in df.columns else 0.0
    out["saldo_bloqueado_total"] = df[bloq_col].apply(_parse_money_pt) if bloq_col in df.columns else 0.0

    out = out[out["account_id"].str.len() > 0].copy()
    out["saldo_total"] = pd.to_numeric(out["saldo_total"], errors="coerce").fillna(0.0).astype(float)
    out["saldo_bloqueado_total"] = pd.to_numeric(out["saldo_bloqueado_total"], errors="coerce").fillna(0.0).astype(float)

    out = out.groupby("account_id", as_index=False).agg(
        saldo_total=("saldo_total", "sum"),
        saldo_bloqueado_total=("saldo_bloqueado_total", "sum"),
    )
    return out


# =========================
# Janela de semanas (28d terminando no último dia do CSV)  ✅ (SEM D-1)
# =========================
def compute_week_ranges(day_ref: pd.Timestamp) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    end = pd.to_datetime(day_ref).normalize()
    start = end - timedelta(days=27)

    w1 = (start, start + timedelta(days=6))
    w2 = (start + timedelta(days=7), start + timedelta(days=13))
    w3 = (start + timedelta(days=14), start + timedelta(days=20))
    w4 = (start + timedelta(days=21), end)

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
        return (
            df.loc[m]
            .groupby(["account_id", "company_key"], as_index=False)["total_cnt"]
            .sum()
            .set_index(["account_id", "company_key"])["total_cnt"]
        )

    dims = facts[["account_id", "company_key", "company_name"]].drop_duplicates().copy()
    dims["account_id"] = dims["account_id"].astype(str)

    s_w1 = sum_week(facts, ranges["w1"][0], ranges["w1"][1])
    s_w2 = sum_week(facts, ranges["w2"][0], ranges["w2"][1])
    s_w3 = sum_week(facts, ranges["w3"][0], ranges["w3"][1])
    s_w4 = sum_week(facts, ranges["w4"][0], ranges["w4"][1])

    start_all = ranges["w1"][0]
    end_all = ranges["w4"][1]
    m_all = (facts["date_dt"] >= start_all) & (facts["date_dt"] <= end_all)
    period = (
        facts.loc[m_all]
        .groupby(["account_id", "company_key"], as_index=False)
        .agg(
            credit=("credit_cnt", "sum"),
            debit=("debit_cnt", "sum"),
            total_4w=("total_cnt", "sum"),
        )
        .set_index(["account_id", "company_key"])
    )

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
# Enrich BaaS (por conta) — agora com saldo_total também
# =========================
def enrich_baas(df_checklist: pd.DataFrame, df_baas: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    out = df_checklist.copy()
    if out.empty:
        return out, 0, 0.0

    out["account_id"] = out["account_id"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)

    if df_baas is None or df_baas.empty:
        out["saldo_total"] = 0.0
        out["saldo_bloqueado_total"] = 0.0
        out["has_block"] = False
        out["lock_icon"] = UNLOCK_ICON
        return out, 0, 0.0

    baas = df_baas.copy()
    baas["account_id"] = baas["account_id"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    baas["saldo_total"] = pd.to_numeric(baas.get("saldo_total", 0.0), errors="coerce").fillna(0.0).astype(float)
    baas["saldo_bloqueado_total"] = pd.to_numeric(baas.get("saldo_bloqueado_total", 0.0), errors="coerce").fillna(0.0).astype(float)

    merged = out.merge(baas, on="account_id", how="left")
    merged["saldo_total"] = pd.to_numeric(merged["saldo_total"], errors="coerce").fillna(0.0).astype(float)
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


def render_badge(label: str, value: str, tone: str):
    cls = {
        "red": "b-red",
        "orange": "b-orange",
        "yellow": "b-yellow",
        "green": "b-green",
        "gray": "b-gray",
    }.get(tone, "b-gray")
    st.markdown(
        f"""<span class="badge {cls}">{label}: <strong>{value}</strong></span>""",
        unsafe_allow_html=True,
    )


# =========================
# Daily matrix (Análise por dia)
# =========================
def build_daily_matrix(
    facts: pd.DataFrame,
    companies_keys: List[str],
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    """
    Tabela por dia (do 1º ao último dia do CSV), por empresa e conta.
    Saída:
      start_day, end_day, table
    table:
      Empresa | Conta | (colunas de dia Timestamp) | Total
    """
    if facts is None or facts.empty:
        return pd.NaT, pd.NaT, pd.DataFrame()

    base = facts.copy()
    if companies_keys:
        base = base[base["company_key"].isin(set(companies_keys))].copy()
    if base.empty:
        return pd.NaT, pd.NaT, pd.DataFrame()

    base["date_dt"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    base = base.dropna(subset=["date_dt"]).copy()

    start_day = base["date_dt"].min()
    end_day = base["date_dt"].max()
    all_days = pd.date_range(start=start_day, end=end_day, freq="D")

    daily = (
        base.groupby(["company_name", "account_id", "date_dt"], as_index=False)["total_cnt"]
        .sum()
        .rename(columns={"total_cnt": "qty"})
    )

    pivot = (
        daily.pivot_table(
            index=["company_name", "account_id"],
            columns="date_dt",
            values="qty",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=all_days, fill_value=0)
    )

    pivot["Total"] = pivot.sum(axis=1)

    out = pivot.reset_index().rename(columns={"company_name": "Empresa", "account_id": "Conta"})
    out["Empresa_sort"] = out["Empresa"].astype(str).str.lower()
    out["Conta_sort"] = out["Conta"].astype(str)
    out = (
        out.sort_values(["Empresa_sort", "Conta_sort"])
        .drop(columns=["Empresa_sort", "Conta_sort"])
        .reset_index(drop=True)
    )

    return start_day, end_day, out


# =========================
# Alert engine (sem LLM) — regras + métricas
# =========================
@dataclass(frozen=True)
class AlertConfig:
    # streaks
    zero_yellow: int = 5
    zero_orange: int = 10
    zero_red: int = 15

    down_yellow: int = 5
    down_orange: int = 10
    down_red: int = 15

    # bloqueio (R$)
    block_yellow: float = 10_000.0
    block_orange: float = 50_000.0
    block_red: float = 100_000.0

    # frequência (% dias com tx)
    freq_yellow: float = 80.0
    freq_orange: float = 60.0
    freq_red: float = 40.0

    # janela da “média” para detectar queda
    baseline_days: int = 7
    # queda = abaixo de X% da baseline
    drop_ratio: float = 0.70


def farol_by_streak(x: int, y: int, o: int, r: int) -> str:
    if x >= r:
        return "red"
    if x >= o:
        return "orange"
    if x >= y:
        return "yellow"
    if x > 0:
        return "gray"
    return "green"


def farol_by_money(x: float, y: float, o: float, r: float) -> str:
    if x >= r:
        return "red"
    if x >= o:
        return "orange"
    if x >= y:
        return "yellow"
    if x > 0:
        return "gray"
    return "green"


def farol_by_freq(pct: float, y: float, o: float, r: float) -> str:
    # quanto menor, pior
    if pct <= r:
        return "red"
    if pct <= o:
        return "orange"
    if pct <= y:
        return "yellow"
    return "green"


def compute_streak_tail(values: np.ndarray, predicate) -> int:
    """Conta streak no FINAL (do último dia pra trás)."""
    n = 0
    for v in values[::-1]:
        if predicate(v):
            n += 1
        else:
            break
    return n


def build_alerts(
    daily_table: pd.DataFrame,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    df_final: pd.DataFrame,
    cfg: AlertConfig,
) -> pd.DataFrame:
    """
    Gera DataFrame de alertas por Conta + Empresa.
    Usa:
      - streak_zero (dias finais zerados)
      - streak_down (dias finais abaixo da baseline)
      - freq_pct (dias com tx / dias totais)
      - bloqueio (saldo_bloqueado_total)
      - saldo_total (saldo_total)
    """
    if daily_table is None or daily_table.empty or df_final is None or df_final.empty:
        return pd.DataFrame()

    day_cols = [c for c in daily_table.columns if isinstance(c, pd.Timestamp)]
    if not day_cols:
        return pd.DataFrame()

    # mapa de saldos por conta
    saldos = df_final[["company_name", "account_id", "saldo_total", "saldo_bloqueado_total", "has_block"]].copy()
    saldos["account_id"] = saldos["account_id"].astype(str).str.replace(r"\D+", "", regex=True)
    saldos = saldos.drop_duplicates(subset=["account_id"], keep="first")

    rows: List[Dict] = []
    for _, r in daily_table.iterrows():
        empresa = str(r["Empresa"])
        conta = str(r["Conta"])
        series = np.array([int(r[c]) for c in day_cols], dtype=int)

        total = int(series.sum())
        last = int(series[-1]) if len(series) else 0

        # frequência
        days_with_tx = int((series > 0).sum())
        freq_pct = (days_with_tx / max(1, len(series))) * 100.0

        # streak zero
        streak_zero = compute_streak_tail(series, lambda v: int(v) == 0)

        # streak queda (abaixo de baseline)
        # baseline = média dos últimos cfg.baseline_days anteriores ao dia atual (rolling simples no tail)
        # Para ficar robusto e rápido: calcula baseline global do período (ignorando zeros opcionais) e compara tail.
        if len(series) >= max(2, cfg.baseline_days + 1):
            baseline = float(np.mean(series[-(cfg.baseline_days + 1):-1]))
        elif len(series) >= 2:
            baseline = float(np.mean(series[:-1]))
        else:
            baseline = float(series[0]) if len(series) else 0.0

        thresh = baseline * float(cfg.drop_ratio)
        streak_down = compute_streak_tail(
            series,
            lambda v: (baseline > 0) and (int(v) < thresh) and (int(v) >= 0),
        )

        # saldos
        srow = saldos[saldos["account_id"] == re.sub(r"\D+", "", conta)]
        saldo_total = float(srow["saldo_total"].iloc[0]) if not srow.empty else 0.0
        saldo_bloq = float(srow["saldo_bloqueado_total"].iloc[0]) if not srow.empty else 0.0
        has_block = bool(srow["has_block"].iloc[0]) if not srow.empty else False

        # faróis
        tone_zero = farol_by_streak(streak_zero, cfg.zero_yellow, cfg.zero_orange, cfg.zero_red)
        tone_down = farol_by_streak(streak_down, cfg.down_yellow, cfg.down_orange, cfg.down_red)
        tone_block = farol_by_money(saldo_bloq, cfg.block_yellow, cfg.block_orange, cfg.block_red)
        tone_freq = farol_by_freq(freq_pct, cfg.freq_yellow, cfg.freq_orange, cfg.freq_red)

        # score (ordenar do pior pro melhor)
        tone_score = {"red": 4, "orange": 3, "yellow": 2, "gray": 1, "green": 0}
        score = (
            tone_score.get(tone_zero, 0) * 1000
            + tone_score.get(tone_down, 0) * 500
            + tone_score.get(tone_block, 0) * 200
            + tone_score.get(tone_freq, 0) * 100
        )

        # “motivo” curto (determinístico)
        motivos = []
        if streak_zero > 0:
            motivos.append(f"zerado há {streak_zero} dia(s)")
        if baseline > 0 and streak_down > 0:
            motivos.append(f"abaixo da baseline ({cfg.baseline_days}d) há {streak_down} dia(s)")
        if has_block and saldo_bloq > 0:
            motivos.append(f"bloqueio R$ {fmt_money_pt(saldo_bloq)}")
        if freq_pct < 100:
            motivos.append(f"freq {freq_pct:.0f}%")

        rows.append({
            "Empresa": empresa,
            "Conta": conta,
            "Total_Periodo": total,
            "Ultimo_Dia": last,
            "Freq_pct": float(freq_pct),
            "Baseline": float(baseline),
            "Thresh_drop": float(thresh),
            "Streak_zero": int(streak_zero),
            "Streak_down": int(streak_down),
            "Saldo": float(saldo_total),
            "Saldo_Bloqueado": float(saldo_bloq),
            "tone_zero": tone_zero,
            "tone_down": tone_down,
            "tone_block": tone_block,
            "tone_freq": tone_freq,
            "score": int(score),
            "motivo_curto": " | ".join(motivos) if motivos else "OK",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["score", "Empresa", "Conta"], ascending=[False, True, True]).reset_index(drop=True)
    df["Periodo"] = f"{start_day.date().isoformat()} → {end_day.date().isoformat()}"
    return df


def assistant_explain_row(row: pd.Series, cfg: AlertConfig) -> str:
    """
    Texto determinístico (sem LLM) para “explicar” a situação da conta.
    """
    empresa = row.get("Empresa", "—")
    conta = row.get("Conta", "—")

    total = int(row.get("Total_Periodo", 0))
    last = int(row.get("Ultimo_Dia", 0))
    freq = float(row.get("Freq_pct", 0.0))
    z = int(row.get("Streak_zero", 0))
    d = int(row.get("Streak_down", 0))
    baseline = float(row.get("Baseline", 0.0))
    thresh = float(row.get("Thresh_drop", 0.0))
    saldo = float(row.get("Saldo", 0.0))
    bloq = float(row.get("Saldo_Bloqueado", 0.0))

    lines = []
    lines.append(f"**Conta {conta} — {empresa}**")
    lines.append(f"- Período analisado: {row.get('Periodo','—')}")
    lines.append(f"- Total no período: **{fmt_int_pt(total)}** | Último dia: **{fmt_int_pt(last)}**")
    lines.append(f"- Frequência: **{freq:.0f}%** (dias com transação / dias do CSV)")
    if baseline > 0:
        lines.append(f"- Baseline ({cfg.baseline_days}d): **{fmt_int_pt(int(round(baseline)))}** | Limiar de queda ({int(cfg.drop_ratio*100)}%): **{fmt_int_pt(int(round(thresh)))}**")
    if z > 0:
        lines.append(f"- 🔴 Zerado no tail: **{z} dia(s)** (consecutivos no final)")
    if baseline > 0 and d > 0:
        lines.append(f"- 🟠 Queda no tail: **{d} dia(s)** abaixo do limiar (vs baseline)")
    if bloq > 0:
        lines.append(f"- 🔒 Saldo bloqueado: **R$ {fmt_money_pt(bloq)}** | Saldo total: **R$ {fmt_money_pt(saldo)}**")
    else:
        lines.append(f"- 🔓 Sem bloqueio | Saldo total: **R$ {fmt_money_pt(saldo)}**")

    # recomendação “engineer”
    actions = []
    if z >= cfg.zero_yellow:
        actions.append("Validar se houve interrupção (webhook/instabilidade/limite) e confirmar operação do cliente no período.")
    if baseline > 0 and d >= cfg.down_yellow and z == 0:
        actions.append("Checar causa de queda: sazonalidade, indisponibilidade, mudança de comportamento ou regra antifraude.")
    if bloq >= cfg.block_yellow:
        actions.append("Checar motivo do bloqueio e impacto operacional (valor elevado).")
    if freq <= cfg.freq_orange:
        actions.append("Checar se a conta opera apenas em dias úteis/fins de semana e ajustar leitura por calendário.")

    if actions:
        lines.append("")
        lines.append("**Sugestões objetivas (auto):**")
        for a in actions[:6]:
            lines.append(f"- {a}")

    return "\n".join(lines)


# =========================
# App
# =========================
st.title("Checklist (Semanas 1–4) + Saldos/Bloqueio (BaaS)")

with st.sidebar:
    st.header("Parâmetros (Status semanal)")
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

    st.divider()
    st.header("Alertas (farol / diário)")

    st.caption("Streak zerado (dias)")
    zero_y = st.number_input("Zerado ≥ (amarelo)", value=5, min_value=1, step=1)
    zero_o = st.number_input("Zerado ≥ (laranja)", value=10, min_value=2, step=1)
    zero_r = st.number_input("Zerado ≥ (vermelho)", value=15, min_value=3, step=1)

    st.caption("Streak queda (dias)")
    down_y = st.number_input("Queda ≥ (amarelo)", value=5, min_value=1, step=1)
    down_o = st.number_input("Queda ≥ (laranja)", value=10, min_value=2, step=1)
    down_r = st.number_input("Queda ≥ (vermelho)", value=15, min_value=3, step=1)

    st.caption("Bloqueio (R$)")
    blk_y = st.number_input("Bloqueio ≥ (amarelo)", value=10_000.0, step=1000.0, format="%.0f")
    blk_o = st.number_input("Bloqueio ≥ (laranja)", value=50_000.0, step=1000.0, format="%.0f")
    blk_r = st.number_input("Bloqueio ≥ (vermelho)", value=100_000.0, step=1000.0, format="%.0f")

    st.caption("Frequência (%, menor = pior)")
    fr_y = st.number_input("Freq ≤ (amarelo)", value=80.0, step=1.0, format="%.0f")
    fr_o = st.number_input("Freq ≤ (laranja)", value=60.0, step=1.0, format="%.0f")
    fr_r = st.number_input("Freq ≤ (vermelho)", value=40.0, step=1.0, format="%.0f")

    st.caption("Baseline / queda")
    baseline_days = st.number_input("Janela baseline (dias)", value=7, min_value=2, step=1)
    drop_ratio = st.slider("Queda = < X% da baseline", min_value=0.30, max_value=0.95, value=0.70, step=0.05)

thresholds = Thresholds(
    queda_critica=float(queda_critica),
    aumento_relevante=float(aumento_relevante),
    investigar_abs=float(investigar_abs),
)

alert_cfg = AlertConfig(
    zero_yellow=int(zero_y),
    zero_orange=int(zero_o),
    zero_red=int(zero_r),
    down_yellow=int(down_y),
    down_orange=int(down_o),
    down_red=int(down_r),
    block_yellow=float(blk_y),
    block_orange=float(blk_o),
    block_red=float(blk_r),
    freq_yellow=float(fr_y),
    freq_orange=float(fr_o),
    freq_red=float(fr_r),
    baseline_days=int(baseline_days),
    drop_ratio=float(drop_ratio),
)

tab_main, tab_alerts, tab_daily = st.tabs(["Principal", "Alertas", "Análise (Diário)"])


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
        baas_file = st.file_uploader("CSV BaaS (opcional) — com Saldo + Saldo Bloqueado", type=["csv"], key="baas_csv")

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

            st.session_state["facts"] = facts
            st.session_state["companies_keys"] = companies_keys

            day_ref, df_checklist, week_labels = build_checklist_weeks(
                facts=facts,
                companies_keys=companies_keys,
                thresholds=thresholds,
                low_volume_threshold=int(low_volume_threshold),
            )

            df_baas = parse_baas_csv(baas_file) if baas_file is not None else pd.DataFrame()
            df_final, kpi_blocked_accounts, kpi_blocked_sum = enrich_baas(df_checklist, df_baas)

            # Daily matrix + alerts (para as outras abas)
            start_day, end_day, daily_table = build_daily_matrix(facts=facts, companies_keys=companies_keys)
            alerts_df = build_alerts(
                daily_table=daily_table,
                start_day=start_day,
                end_day=end_day,
                df_final=df_final,
                cfg=alert_cfg,
            )

            st.session_state["day_ref"] = day_ref
            st.session_state["week_labels"] = week_labels
            st.session_state["df_final"] = df_final
            st.session_state["kpi_blocked_accounts"] = int(kpi_blocked_accounts)
            st.session_state["kpi_blocked_sum"] = float(kpi_blocked_sum)
            st.session_state["page"] = 1

            st.session_state["daily_start"] = start_day
            st.session_state["daily_end"] = end_day
            st.session_state["daily_table"] = daily_table
            st.session_state["alerts_df"] = alerts_df

    if "df_final" not in st.session_state:
        st.info("Preencha empresas + suba CSV + clique em **Processar**.")
        st.stop()

    df_final = st.session_state.get("df_final", pd.DataFrame())
    week_labels = st.session_state.get("week_labels", {})
    day_ref = st.session_state.get("day_ref", "—")
    kpi_blocked_accounts = int(st.session_state.get("kpi_blocked_accounts", 0))
    kpi_blocked_sum = float(st.session_state.get("kpi_blocked_sum", 0.0))

    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
    with k1:
        st.metric("Dia de referência", day_ref)
    with k2:
        st.metric("Empresas filtradas", fmt_int_pt(int(df_final["company_key"].nunique()) if not df_final.empty else 0))
    with k3:
        st.metric("Alertas (semanal)", fmt_int_pt(int((df_final["status"] != "Normal").sum()) if not df_final.empty else 0))
    with k4:
        st.metric("Alertas críticos", fmt_int_pt(int((df_final["status"] == "Alerta (queda/ zerada)").sum()) if not df_final.empty else 0))
    with k5:
        st.metric("Contas com bloqueio", fmt_int_pt(kpi_blocked_accounts))
    with k6:
        st.metric("Soma bloqueada (R$)", fmt_money_pt(kpi_blocked_sum))

    st.divider()

    st.subheader("Visão Geral (semanal)")
    render_status_boxes(df_final)
    st.divider()

    if px is None or go is None:
        st.warning("Instale plotly para ver o Ground.")
    elif df_final.empty:
        st.info("Sem dados.")
    else:
        left, _right = st.columns([1.1, 1.0], gap="large")

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

    st.divider()

    st.subheader("Checklist (por conta) — semanal")

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
        "Saldo (R$)": view["saldo_total"].map(fmt_money_pt) if "saldo_total" in view.columns else "0,00",
        "Saldo Bloqueado (R$)": view["saldo_bloqueado_total"].map(fmt_money_pt),
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

    st.subheader("Cards (para colar no Discord) — semanal")
    alerts_week = df_final[df_final["status"] != "Normal"].copy()

    if alerts_week.empty:
        st.caption("Nenhum alerta (status != Normal).")
    else:
        for _, r in alerts_week.iterrows():
            title = f"{r['status']} {r['lock_icon']} — {r['company_name']} | Conta {r['account_id']}"
            with st.expander(title, expanded=False):
                msg = (
                    f"ALERTA (SEMANAL): {r['status']}\n"
                    f"Empresa: {r['company_name']}\n"
                    f"Conta: {r['account_id']}\n"
                    f"Data (day_ref): {r['day_ref']}\n"
                    f"Motivo: {r['motivo']}\n"
                    f"Semanas: S1={r['week1']} | S2={r['week2']} | S3={r['week3']} | S4={r['week4']}\n"
                    f"Crédito(Qtd): {r['credit']} | Débito(Qtd): {r['debit']}\n"
                    f"Total 4 semanas: {r['total_4w']}\n"
                    f"Saldo: R$ {fmt_money_pt(r.get('saldo_total', 0.0))}\n"
                    f"Bloqueio: R$ {fmt_money_pt(r.get('saldo_bloqueado_total', 0.0))} {r['lock_icon']}"
                )
                st.code(msg, language="text")


# =========================
# ALERTAS (DIÁRIO)
# =========================
with tab_alerts:
    if "alerts_df" not in st.session_state:
        st.info("Você precisa processar na aba Principal primeiro.")
        st.stop()

    alerts_df = st.session_state.get("alerts_df", pd.DataFrame())
    if alerts_df is None or alerts_df.empty:
        st.info("Sem alertas diários para mostrar (ou sem dados suficientes).")
        st.stop()

    st.subheader("Alertas (Diário) — farol por conta")
    st.caption("Ordenado do pior → melhor. Use filtros para focar em empresa/conta.")

    companies = ["TOTAL GERAL"] + sorted(alerts_df["Empresa"].dropna().astype(str).unique().tolist())

    c1, c2, c3 = st.columns([1.2, 1.0, 0.8], gap="small")
    with c1:
        sel_emp = st.selectbox("Empresa (filtro)", companies, index=0, key="alerts_emp")
    with c2:
        q_emp = st.text_input("Buscar por nome", value="", key="alerts_q_emp")
    with c3:
        q_acc = st.text_input("Buscar por conta", value="", key="alerts_q_acc")

    v = alerts_df.copy()
    if sel_emp != "TOTAL GERAL":
        v = v[v["Empresa"] == sel_emp].copy()
    if q_emp.strip():
        q = q_emp.strip().lower()
        v = v[v["Empresa"].astype(str).str.lower().str.contains(q, na=False)]
    if q_acc.strip():
        q = re.sub(r"\D+", "", q_acc.strip())
        if q:
            v = v[v["Conta"].astype(str).str.contains(q, na=False)]

    # KPIs de topo (diário)
    day_ref = st.session_state.get("day_ref", "—")
    total_accounts = int(len(v))
    red_count = int(((v["tone_zero"] == "red") | (v["tone_down"] == "red") | (v["tone_block"] == "red") | (v["tone_freq"] == "red")).sum())
    zero_now = int((v["Streak_zero"] > 0).sum())
    block_now = int((v["Saldo_Bloqueado"] > 0).sum())

    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        st.metric("Day_ref (semanal)", day_ref)
    with k2:
        st.metric("Contas no escopo", fmt_int_pt(total_accounts))
    with k3:
        st.metric("Contas com streak zerado", fmt_int_pt(zero_now))
    with k4:
        st.metric("Contas com bloqueio", fmt_int_pt(block_now))

    st.divider()

    if v.empty:
        st.info("Sem resultados para os filtros.")
        st.stop()

    # Render cards
    for _, r in v.iterrows():
        empresa = r["Empresa"]
        conta = r["Conta"]

        title = f"{empresa} | Conta {conta}"
        subtitle = str(r.get("motivo_curto", "OK"))

        with st.expander(f"{title} — {subtitle}", expanded=False):
            st.markdown('<div class="badges">', unsafe_allow_html=True)

            render_badge("Zerado (tail)", f"{int(r['Streak_zero'])}d", str(r["tone_zero"]))
            render_badge("Queda (tail)", f"{int(r['Streak_down'])}d", str(r["tone_down"]))
            render_badge("Bloqueio", f"R$ {fmt_money_pt(float(r['Saldo_Bloqueado']))}", str(r["tone_block"]))
            render_badge("Frequência", f"{float(r['Freq_pct']):.0f}%", str(r["tone_freq"]))

            st.markdown("</div>", unsafe_allow_html=True)

            # detalhes rápidos
            d1, d2, d3, d4 = st.columns(4, gap="small")
            with d1:
                st.metric("Total período", fmt_int_pt(int(r["Total_Periodo"])))
            with d2:
                st.metric("Último dia", fmt_int_pt(int(r["Ultimo_Dia"])))
            with d3:
                st.metric("Saldo (R$)", fmt_money_pt(float(r["Saldo"])))
            with d4:
                st.metric("Baseline (7d)", fmt_int_pt(int(round(float(r["Baseline"])))))

            st.markdown("**Resumo determinístico (assistente):**")
            st.markdown(assistant_explain_row(r, alert_cfg))


# =========================
# ANÁLISE (DIÁRIO)
# =========================
with tab_daily:
    if "daily_table" not in st.session_state or "df_final" not in st.session_state:
        st.info("Você precisa processar na aba Principal primeiro.")
        st.stop()

    daily_table = st.session_state.get("daily_table", pd.DataFrame())
    start_day = st.session_state.get("daily_start", pd.NaT)
    end_day = st.session_state.get("daily_end", pd.NaT)
    df_final = st.session_state.get("df_final", pd.DataFrame())

    if daily_table is None or daily_table.empty or pd.isna(start_day) or pd.isna(end_day):
        st.info("Sem dados suficientes para montar a análise diária.")
        st.stop()

    st.subheader("Análise de Transações por Dia")
    st.caption("Tabela por dia (do 1º ao último dia do CSV) — por empresa e conta. Total no fim da linha.")

    companies_names = ["TOTAL GERAL"] + sorted(daily_table["Empresa"].dropna().astype(str).unique().tolist())

    cA, cB, cC = st.columns([1.2, 1.0, 1.0], gap="small")
    with cA:
        selected_company = st.selectbox("Empresa (filtro)", options=companies_names, index=0, key="daily_company_select")
    with cB:
        q_nome = st.text_input("Buscar por nome", value="", key="daily_q_nome")
    with cC:
        q_conta = st.text_input("Buscar por conta", value="", key="daily_q_conta")

    view = daily_table.copy()
    if selected_company != "TOTAL GERAL":
        view = view[view["Empresa"] == selected_company].copy()
    if q_nome.strip():
        q = q_nome.strip().lower()
        view = view[view["Empresa"].astype(str).str.lower().str.contains(q, na=False)]
    if q_conta.strip():
        q = re.sub(r"\D+", "", q_conta.strip())
        if q:
            view = view[view["Conta"].astype(str).str.contains(q, na=False)]

    day_cols = [c for c in view.columns if isinstance(c, pd.Timestamp)]

    empresas_filtradas = int(view["Empresa"].nunique()) if not view.empty else 0
    total_periodo = int(view["Total"].sum()) if not view.empty else 0

    last_day_total = int(view[end_day].sum()) if (not view.empty and end_day in view.columns) else 0

    freq_pct = 0.0
    if not view.empty and len(day_cols) > 0:
        day_sums = view[day_cols].sum(axis=0)
        days_with_tx = int((day_sums > 0).sum())
        freq_pct = (days_with_tx / max(1, len(day_cols))) * 100.0

    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        st.metric("Empresas filtradas (escopo)", fmt_int_pt(empresas_filtradas))
    with k2:
        st.metric("Total transações (período)", fmt_int_pt(total_periodo))
    with k3:
        st.metric("Total no último dia do CSV", fmt_int_pt(last_day_total))
    with k4:
        st.metric("Frequência (dias com TX)", f"{freq_pct:.0f}%")

    st.divider()

    # Assistente (sem LLM): selecionar uma conta e explicar
    st.subheader("Assistente (sem LLM) — explicar uma conta")
    st.caption("Escolha uma conta (empresa+conta) para ver métricas e explicação determinística.")

    # monta lista de opções de conta no escopo atual (view)
    if view.empty:
        st.info("Sem linhas no escopo atual.")
    else:
        view_keys = view[["Empresa", "Conta"]].astype(str)
        options = (view_keys["Empresa"] + " | " + view_keys["Conta"]).tolist()
        sel = st.selectbox("Conta (escopo atual)", options=options, index=0, key="daily_assistant_sel")

        emp_sel, acc_sel = sel.split(" | ", 1)
        row_daily = view[(view["Empresa"] == emp_sel) & (view["Conta"] == acc_sel)].iloc[0].copy()

        # gera “row alerta” on-the-fly para essa conta (reusa engine)
        # (para ter streaks, baseline, faróis, etc.)
        tmp_daily = pd.DataFrame([row_daily])
        tmp_alerts = build_alerts(tmp_daily, start_day, end_day, df_final, alert_cfg)
        if tmp_alerts is not None and not tmp_alerts.empty:
            ar = tmp_alerts.iloc[0]

            st.markdown('<div class="badges">', unsafe_allow_html=True)
            render_badge("Zerado (tail)", f"{int(ar['Streak_zero'])}d", str(ar["tone_zero"]))
            render_badge("Queda (tail)", f"{int(ar['Streak_down'])}d", str(ar["tone_down"]))
            render_badge("Bloqueio", f"R$ {fmt_money_pt(float(ar['Saldo_Bloqueado']))}", str(ar["tone_block"]))
            render_badge("Frequência", f"{float(ar['Freq_pct']):.0f}%", str(ar["tone_freq"]))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(assistant_explain_row(ar, alert_cfg))
        else:
            st.info("Não foi possível calcular explicação para esta conta (dados insuficientes).")

    st.divider()

    # Tabela diária (colunas como dia do mês)
    pretty = view.copy()
    rename_map = {c: c.strftime("%d") for c in pretty.columns if isinstance(c, pd.Timestamp)}
    pretty = pretty.rename(columns=rename_map)

    for col in list(rename_map.values()) + ["Total"]:
        if col in pretty.columns:
            pretty[col] = pretty[col].astype(int).map(fmt_int_pt)

    st.caption(f"Período do CSV: {start_day.date().isoformat()} → {end_day.date().isoformat()}")
    st.dataframe(pretty, use_container_width=True, hide_index=True)
