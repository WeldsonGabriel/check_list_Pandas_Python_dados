# app.py
# Streamlit dashboard — Checklist (CSV Transações) + (opcional) CSV BaaS (bloqueio cautelar)
# Versão ajustada:
# - Só processa após clicar em "Processar"
# - Filtro de empresas obrigatório (input)
# - BaaS casa por CONTA (accounts_list)
# - Gráficos com cores (status; var_30d: verde>0 / vermelho<0)
# - Paginação SEM setas externas (apenas number_input com +/-)
# - Busca por empresa no Checklist e no BaaS
# - Gauge + KPIs com largura fixa 480 e dropdown (TOTAL + empresa)

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
st.set_page_config(page_title="Checklist + BaaS", layout="wide")

STATUS_ORDER = ["Escalar (queda)", "Investigar", "Gerenciar (aumento)", "Normal"]

STATUS_COLOR = {
    "Escalar (queda)": "#e74c3c",       # vermelho
    "Investigar": "#f1c40f",            # amarelo
    "Gerenciar (aumento)": "#3498db",   # azul
    "Normal": "#2ecc71",                # verde
    "Desconhecido": "#aab4c8",
}

LOCK_ICON = "🔒"
UNLOCK_ICON = "🔓"
UNKNOWN_ICON = "⚪"


# =========================
# CSS (Gauge + KPI grid 480)
# =========================
GAUGE_KPI_CSS = """
<style>
/* força melhor espaçamento do título do gauge */
.gauge-title{
  margin: 0 0 10px 0;
  padding: 0;
}

/* container do bloco (gauge + KPIs) */
.gauge-wrap{
  width: 100%;
  max-width: 480px;      /* ✅ pedido: 480 */
  margin: 0 auto;        /* centraliza */
}

/* grid dos 6 cards */
.kpi-grid{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 10px;
}

/* card */
.kpi-card{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  min-height: 74px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

/* label e valor */
.kpi-label{
  font-size: 12px;
  color: rgba(231,236,246,0.75);
  line-height: 1.1;
  margin-bottom: 6px;
}
.kpi-value{
  font-size: 26px;
  font-weight: 800;
  color: rgba(231,236,246,0.98);
  line-height: 1;
}

/* variação: cor por sinal */
.kpi-value.pos{ color: #2ecc71; }
.kpi-value.neg{ color: #e74c3c; }

/* cards de status (quadradinhos) */
.status-row{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 8px;
}
.status-box{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  min-height: 92px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: center;
}
.status-num{
  font-size: 30px;
  font-weight: 900;
  line-height: 1.0;
  margin-bottom: 6px;
}
.status-label{
  font-size: 12px;
  color: rgba(231,236,246,0.75);
}
.status-box.red   { border-color: rgba(231,76,60,0.35);  box-shadow: 0 12px 30px rgba(231,76,60,0.07); }
.status-box.yellow{ border-color: rgba(241,196,15,0.35); box-shadow: 0 12px 30px rgba(241,196,15,0.06); }
.status-box.blue  { border-color: rgba(52,152,219,0.35); box-shadow: 0 12px 30px rgba(52,152,219,0.06); }
.status-box.green { border-color: rgba(46,204,113,0.35); box-shadow: 0 12px 30px rgba(46,204,113,0.06); }

/* responsivo */
@media (max-width: 1100px){
  .status-row{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 700px){
  .status-row{ grid-template-columns: 1fr; }
  .gauge-wrap{ max-width: 100%; }
  .kpi-value{ font-size: 24px; }
}
</style>
"""
st.markdown(GAUGE_KPI_CSS, unsafe_allow_html=True)


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
    s = _strip_accents(s).upper()
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
    out, seen = [], set()
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
    if status == "Escalar (queda)":
        return 0
    if status == "Investigar":
        return 1
    if status == "Gerenciar (aumento)":
        return 2
    if status == "Normal":
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
# CSV parsers
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
            f"CSV transações: colunas não identificadas. "
            f"date={date_col}, person={person_col}, credit={credit_col}, debit={debit_col}"
        )

    work = df[[date_col, person_col, credit_col, debit_col]].copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce", dayfirst=True).dt.date

    person = work[person_col].astype(str).fillna("")
    parts = person.str.split(" - ", n=1, expand=True)
    account = parts[0].fillna("").astype(str).str.strip()
    company = (parts[1] if parts.shape[1] > 1 else "").fillna("").astype(str).str.strip()
    company = np.where(company == "", person, company)

    account = account.str.replace(r"\D+", "", regex=True).replace("", np.nan)

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
    bloqueado_col = _pick_col(df, ["bloquead"])

    # saldo (cuidado para não pegar "saldo bloqueado")
    saldo_col = _pick_col(df, ["saldo"])
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
            f"CSV BaaS: colunas não identificadas. conta={conta_col}, bloqueado={bloqueado_col}, nome={nome_col}"
        )

    keep = [c for c in [pos_col, conta_col, ag_col, nome_col, plano_col, saldo_col, bloqueado_col] if c]
    work = df[keep].copy()

    ren = {}
    if pos_col: ren[pos_col] = "posicao"
    ren[conta_col] = "conta"
    if ag_col: ren[ag_col] = "agencia"
    ren[nome_col] = "nome"
    if plano_col: ren[plano_col] = "plano"
    if saldo_col: ren[saldo_col] = "saldo"
    ren[bloqueado_col] = "saldo_bloqueado"
    work = work.rename(columns=ren)

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
# Checklist builder
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

    return sorted(set(active)), sorted(set(zero))

def build_checklist(
    facts: pd.DataFrame,
    companies_keys: List[str],
    thresholds: Thresholds,
) -> Tuple[str, pd.DataFrame]:
    if facts.empty:
        return "", pd.DataFrame()

    if companies_keys:
        facts = facts[facts["company_key"].isin(set(companies_keys))].copy()
    if facts.empty:
        return "", pd.DataFrame()

    day_ref = pd.to_datetime(max(facts["date"]))
    d1 = (day_ref - timedelta(days=1)).date()

    base_acc = facts.groupby(["account_id", "company_key", "company_name", "date"], as_index=False)["total"].sum()
    base_comp = base_acc.groupby(["company_key", "company_name", "date"], as_index=False)["total"].sum()

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

        rows.append({
            "company_key": company_key,
            "company_name": company_name,
            "day_ref": day_ref.date().isoformat(),
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
            "obs": calc_obs(status),
            "severity": severity_rank(status),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return day_ref.date().isoformat(), df

    df = df.sort_values(["severity", "company_name"], ascending=[True, True]).reset_index(drop=True)
    return day_ref.date().isoformat(), df


# =========================
# BaaS enrichment (match by account)
# =========================
def enrich_with_baas(df_checklist: pd.DataFrame, df_baas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if df_checklist is None or df_checklist.empty:
        return df_checklist, pd.DataFrame(), 0

    out = df_checklist.copy()

    if df_baas is None or df_baas.empty:
        out["has_block"] = False
        out["blocked_total"] = 0.0
        out["blocked_accounts"] = ""
        out["lock_icon"] = UNKNOWN_ICON
        return out, pd.DataFrame(), 0

    if "conta" not in df_baas.columns or "saldo_bloqueado" not in df_baas.columns:
        raise ValueError("CSV BaaS inválido: esperado colunas 'conta' e 'saldo_bloqueado'.")

    baas = df_baas.copy()
    baas["conta"] = baas["conta"].astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    baas["saldo_bloqueado"] = pd.to_numeric(baas["saldo_bloqueado"], errors="coerce").fillna(0.0).astype(float)

    baas_agg = baas.groupby("conta", as_index=False)["saldo_bloqueado"].sum()
    baas_agg = baas_agg.sort_values("saldo_bloqueado", ascending=False)

    def split_accounts(s: str) -> List[str]:
        if not s:
            return []
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        out2 = []
        for p in parts:
            p2 = re.sub(r"\D+", "", p)
            if p2:
                out2.append(p2)
        return out2

    has_block_list = []
    blocked_total_list = []
    blocked_accounts_list = []
    lock_icon_list = []

    blocked_rows = []
    blocked_accounts_all = set()

    for _, row in out.iterrows():
        accounts = split_accounts(row.get("accounts_list", ""))
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

        contas_hit = hit["conta"].astype(str).tolist()
        total_blocked = float(hit["saldo_bloqueado"].sum())

        has_block_list.append(True)
        blocked_total_list.append(total_blocked)
        blocked_accounts_list.append(", ".join(contas_hit))
        lock_icon_list.append(LOCK_ICON)

        for c in contas_hit:
            blocked_accounts_all.add(c)

        det = baas[baas["conta"].isin(contas_hit)].copy()
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

    return out, df_baas_table, len(blocked_accounts_all)


# =========================
# Format helpers
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

def paginate_df(df: pd.DataFrame, page_size: int, key: str) -> Tuple[pd.DataFrame, int, int]:
    if df is None or df.empty:
        return df, 1, 1
    total = len(df)
    pages = max(1, int(math.ceil(total / page_size)))
    page = st.session_state.get(key, 1)
    page = max(1, min(pages, int(page)))
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].copy(), page, pages

def render_page_input(pages: int, state_key: str, label: str):
    if pages <= 1:
        return
    st.number_input(
        label,
        min_value=1,
        max_value=pages,
        value=int(st.session_state.get(state_key, 1)),
        step=1,
        key=state_key,
    )

def kpi_card(label: str, value: str, cls: str = "") -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {cls}">{value}</div>
    </div>
    """


# =========================
# Gauge renderer (TOTAL or COMPANY)
# =========================
def render_gauge_block(title: str, today_i: int, avg7: float, avg15: float, avg30: float,
                      var7: float, var15: float, var30: float):
    if go is None:
        st.warning("plotly não está disponível (instale plotly).")
        return

    # card values
    avg7_i = round_int_nearest(avg7)
    avg15_i = round_int_nearest(avg15)
    avg30_i = round_int_nearest(avg30)

    v7 = pct_int_ceil_ratio(var7)
    v15 = pct_int_ceil_ratio(var15)
    v30 = pct_int_ceil_ratio(var30)

    v7_cls = "pos" if v7 >= 0 else "neg"
    v15_cls = "pos" if v15 >= 0 else "neg"
    v30_cls = "pos" if v30 >= 0 else "neg"

    # gauge max: usa max entre (today e avg30*1.8) para não ficar "esticado"
    maxv = max(1, int(max(today_i, round_int_nearest(avg30 * 1.8))))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=today_i,
        number={"font": {"size": 56}},
        gauge={
            "axis": {"range": [0, maxv]},
            "bar": {"color": "#3498db"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, maxv], "color": "rgba(255,255,255,0.07)"},
            ],
        },
        title={"text": ""},  # título fora, para não cortar
    ))
    fig.update_layout(
        height=310,
        margin=dict(l=20, r=20, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(231,236,246,0.98)"),
    )

    st.markdown(f"<h3 class='gauge-title'>{title}</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    html = f"""
    <div class="gauge-wrap">
      <div class="kpi-grid">
        {kpi_card("Média 7d", fmt_int_pt(avg7_i))}
        {kpi_card("Variação 7d %", f"{v7}%", v7_cls)}
        {kpi_card("Média 15d", fmt_int_pt(avg15_i))}
        {kpi_card("Variação 15d %", f"{v15}%", v15_cls)}
        {kpi_card("Média 30d", fmt_int_pt(avg30_i))}
        {kpi_card("Variação 30d %", f"{v30}%", v30_cls)}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================
# UI
# =========================
st.title("Alertas (CSV → Checklist + Cards) + BaaS (opcional)")
st.caption("Fluxo: cole as empresas → suba CSV transações → (opcional) CSV BaaS → clique em **Processar**.")

with st.sidebar:
    st.header("Parâmetros")
    queda_critica = st.number_input("Queda crítica", value=-0.60, step=0.01, format="%.2f")
    aumento_relevante = st.number_input("Aumento relevante", value=0.80, step=0.01, format="%.2f")
    investigar_abs = st.number_input("Investigar abs", value=0.30, step=0.01, format="%.2f")

    st.divider()
    st.subheader("Paginação")
    page_size_checklist = st.selectbox("Checklist (por página)", [15, 20, 30], index=0)
    page_size_baas = st.selectbox("BaaS (por página)", [10, 15, 20], index=0)

    st.divider()
    topn = st.selectbox("Top var 30d", [8, 10, 15], index=0)

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
# Processing
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

        day_ref, df_checklist = build_checklist(facts, companies_keys, thresholds)
        if df_checklist.empty:
            st.warning("Nenhuma empresa do input apareceu no CSV (após normalização), ou não há contas ativas no período.")
            st.session_state["day_ref"] = day_ref
            st.session_state["df_checklist"] = df_checklist
            st.session_state["df_baas_table"] = pd.DataFrame()
            st.session_state["kpi_blocked_accounts"] = 0
        else:
            df_baas = parse_baas_csv(baas_file) if baas_file is not None else pd.DataFrame()
            df_checklist2, df_baas_table, kpi_blocked_accounts = enrich_with_baas(df_checklist, df_baas)

            st.session_state["day_ref"] = day_ref
            st.session_state["df_checklist"] = df_checklist2
            st.session_state["df_baas_table"] = df_baas_table
            st.session_state["kpi_blocked_accounts"] = kpi_blocked_accounts

        st.session_state["page_checklist"] = 1
        st.session_state["page_baas"] = 1

# =========================
# Render
# =========================
if "df_checklist" not in st.session_state:
    st.info("Aguardando você preencher as empresas + subir o CSV e clicar em **Processar**.")
    st.stop()

df_checklist = st.session_state.get("df_checklist", pd.DataFrame())
df_baas_table = st.session_state.get("df_baas_table", pd.DataFrame())
day_ref = st.session_state.get("day_ref", "—")
kpi_blocked_accounts = int(st.session_state.get("kpi_blocked_accounts", 0))

# KPIs topo
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
    # ✅ garante o label correto (evita “proteção”)
    st.metric("Contas com bloqueio", fmt_int_pt(kpi_blocked_accounts))

# quadradinhos status (sem filtro, sem clique)
st.markdown("### Status")
if df_checklist.empty:
    st.caption("Sem dados.")
else:
    n_escalar = int((df_checklist["status"] == "Escalar (queda)").sum())
    n_invest = int((df_checklist["status"] == "Investigar").sum())
    n_ger = int((df_checklist["status"] == "Gerenciar (aumento)").sum())
    n_norm = int((df_checklist["status"] == "Normal").sum())

    st.markdown(
        f"""
        <div class="status-row">
          <div class="status-box red">
            <div class="status-num" style="color:{STATUS_COLOR['Escalar (queda)']}">{n_escalar}</div>
            <div class="status-label">Escalar</div>
          </div>
          <div class="status-box yellow">
            <div class="status-num" style="color:{STATUS_COLOR['Investigar']}">{n_invest}</div>
            <div class="status-label">Investigar</div>
          </div>
          <div class="status-box blue">
            <div class="status-num" style="color:{STATUS_COLOR['Gerenciar (aumento)']}">{n_ger}</div>
            <div class="status-label">Gerenciar</div>
          </div>
          <div class="status-box green">
            <div class="status-num" style="color:{STATUS_COLOR['Normal']}">{n_norm}</div>
            <div class="status-label">Normal</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# Visão geral: Gauge (esquerda) + Top var (direita)
st.subheader("Visão Geral")

if px is None or go is None:
    st.warning("plotly não está disponível. Instale 'plotly' para ver gráficos e gauge.")
else:
    c_left, c_right = st.columns([1.0, 1.1], gap="large")

    with c_left:
        if df_checklist.empty:
            st.info("Sem dados.")
        else:
            # dropdown TOTAL + empresas
            options = ["TOTAL GERAL"] + df_checklist["company_name"].astype(str).tolist()
            sel = st.selectbox("Selecionar (gauge)", options, index=0, key="gauge_select")

            if sel == "TOTAL GERAL":
                today_i = int(df_checklist["today_total_i"].sum())
                avg7 = float(df_checklist["avg_7d"].sum())
                avg15 = float(df_checklist["avg_15d"].sum())
                avg30 = float(df_checklist["avg_30d"].sum())
                var7 = calc_var(float(df_checklist["today_total"].sum()), safe_div(float(df_checklist["avg_7d"].sum()), 1))
                var15 = calc_var(float(df_checklist["today_total"].sum()), safe_div(float(df_checklist["avg_15d"].sum()), 1))
                var30 = calc_var(float(df_checklist["today_total"].sum()), safe_div(float(df_checklist["avg_30d"].sum()), 1))
                render_gauge_block("Total geral — Hoje", today_i, avg7, avg15, avg30, var7, var15, var30)
            else:
                row = df_checklist[df_checklist["company_name"].astype(str) == sel].iloc[0]
                render_gauge_block(
                    f"{sel} — Hoje",
                    int(row["today_total_i"]),
                    float(row["avg_7d"]),
                    float(row["avg_15d"]),
                    float(row["avg_30d"]),
                    float(row["var_7d"]),
                    float(row["var_15d"]),
                    float(row["var_30d"]),
                )

    with c_right:
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
            fig = px.bar(
                merged,
                x="company_name",
                y="var_30d",
                color="sign",
                color_discrete_map={"pos": "#2ecc71", "neg": "#e74c3c"},
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title=None,
                yaxis_title=None,
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig.update_traces(marker_line_width=0)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# Checklist
st.subheader("Checklist")

if df_checklist.empty:
    st.info("Sem empresas para mostrar (verifique o filtro e os CSVs).")
else:
    search_company = st.text_input("Buscar empresa (Checklist)", value="", key="search_checklist")
    view = df_checklist.copy()

    if search_company.strip():
        s = normalize_company(search_company)
        view = view[view["company_name"].astype(str).apply(normalize_company).str.contains(s, na=False)].copy()

    # status com cadeado (ícone apenas)
    view["status_view"] = view["status"].astype(str) + " " + view.get("lock_icon", UNKNOWN_ICON).astype(str)

    show = view[[
        "status_view", "company_name",
        "today_total_i", "avg_7d_i", "avg_15d_i", "avg_30d_i",
        "var_30d_pct", "accounts_list", "obs"
    ]].rename(columns={
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

    show["Var 30d"] = show["Var 30d"].apply(lambda x: f"{int(x)}%")

    st.session_state.setdefault("page_checklist", 1)
    page_df, page, pages = paginate_df(show, page_size=int(page_size_checklist), key="page_checklist")
    render_page_input(pages, "page_checklist", f"Página Checklist (1–{pages})")
    st.dataframe(page_df, use_container_width=True, hide_index=True)

st.divider()

# BaaS
st.subheader("Checklist Bloqueio Cautelar (BaaS) — por conta")

if df_baas_table is None or df_baas_table.empty:
    st.caption("Sem dados de BaaS (ou nenhum bloqueio encontrado para as empresas filtradas).")
else:
    search_baas = st.text_input("Buscar empresa (BaaS)", value="", key="search_baas")
    t = df_baas_table.copy()

    if search_baas.strip():
        s = normalize_company(search_baas)
        t = t[t["company_name"].astype(str).apply(normalize_company).str.contains(s, na=False)].copy()

    cols = []
    if "posicao" in t.columns: cols.append("posicao")
    cols += ["conta"]
    if "agencia" in t.columns: cols.append("agencia")
    cols += ["company_name", "saldo_bloqueado"]

    t = t[cols].copy()
    t = t.rename(columns={
        "posicao": "Posição",
        "conta": "Conta",
        "agencia": "Agência",
        "company_name": "Empresa",
        "saldo_bloqueado": "Saldo Bloqueado",
    })
    t["Saldo Bloqueado"] = t["Saldo Bloqueado"].apply(fmt_money_pt)

    st.session_state.setdefault("page_baas", 1)
    page_df, page, pages = paginate_df(t, page_size=int(page_size_baas), key="page_baas")
    render_page_input(pages, "page_baas", f"Página BaaS (1–{pages})")
    st.dataframe(page_df, use_container_width=True, hide_index=True)

st.divider()

# Cards (Discord)
st.subheader("Cards (para colar no Discord)")

if df_checklist is None or df_checklist.empty:
    st.caption("Sem dados.")
else:
    df_cards = df_checklist[df_checklist["status"] != "Normal"].copy()
    if df_cards.empty:
        st.caption("Nenhum alerta (status != Normal).")
    else:
        for _, r in df_cards.iterrows():
            title = f"{r['status']} {r.get('lock_icon','⚪')} — {r['company_name']}"
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