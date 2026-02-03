# app.py
# Streamlit dashboard: CSV Transações -> Checklist + Cards + (opcional) CSV BaaS (bloqueio cautelar)
# - Filtra empresas por input
# - Normaliza "Person Name" = "CONTA - EMPRESA"
# - Calcula médias 7/15/30 com janela [day_ref - N, day_ref-1]
# - Anti falso-positivo multi-conta: empresa só entra se tiver >=1 conta com total>0 no período 30d até ontem
# - Inteiros no checklist (arredondamento tradicional, .5 para cima)
# - Variação % inteira (ceil, negativo aproxima para zero)
# - CSV BaaS (opcional): marca bloqueio por conta (Saldo Bloqueado > 0)
# - Checklist BaaS paginado (10 por página)
# - Cards com indicador 🔒 (sem texto "bloqueio/sem bloqueio" no status)

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(page_title="Alertas (CSV → Checklist + Cards)", layout="wide")

PAGE_SIZE_CHECKLIST = 15
PAGE_SIZE_BAAS = 10

# =========================
# Domain rules (JS/Python parity)
# =========================

@dataclass(frozen=True)
class Thresholds:
    queda_critica: float = -0.60
    aumento_relevante: float = 0.80
    investigar_abs: float = 0.30


def _safe_div(n: float, d: float) -> float:
    if not d:
        return 0.0
    return float(n) / float(d)


def _round_int_nearest(x: float) -> int:
    """Inteiro SEMPRE, regra clássica: 0.5 arredonda para cima."""
    try:
        v = float(x)
    except Exception:
        return 0
    if v >= 0:
        return int(math.floor(v + 0.5))
    return int(math.ceil(v - 0.5))


def _pct_int_ceil_ratio(var_ratio: float) -> int:
    """
    Razão -> % inteiro com ceil.
    - Positivo: ceil(80.1)=81
    - Negativo: ceil(-60.1)=-60 (aproxima para zero), alinhado ao teu exemplo.
    """
    try:
        pct = float(var_ratio) * 100.0
    except Exception:
        return 0
    return int(math.ceil(pct))


def calc_var(today_total: float, avg: float) -> float:
    """Razão de variação: (today-avg)/avg"""
    if avg == 0:
        return 0.0
    return (float(today_total) - float(avg)) / float(avg)


def calc_status(var30: float, *, thresholds: Thresholds) -> str:
    if var30 <= thresholds.queda_critica:
        return "Escalar (queda)"
    if var30 >= thresholds.aumento_relevante:
        return "Gerenciar (aumento)"
    if abs(var30) >= thresholds.investigar_abs:
        return "Investigar"
    return "Normal"


def calc_obs(status: str) -> str:
    if status == "Escalar (queda)":
        return "Queda crítica vs média histórica"
    if status == "Gerenciar (aumento)":
        return "Aumento relevante vs média histórica"
    if status == "Investigar":
        return "Variação relevante (investigar)"
    return "Sem observações"


def severity_rank(status: str) -> int:
    # menor = mais severo
    if status == "Escalar (queda)":
        return 0
    if status == "Investigar":
        return 1
    if status == "Gerenciar (aumento)":
        return 2
    if status == "Normal":
        return 3
    return 9


# =========================
# Normalizers
# =========================

_SUFFIXES = {"LTDA", "Ltda", "ME", "EPP", "S/A", "SA", "S A", "S.A", "S.A.", "EIRELI"}


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalize_company_name(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return ""
    s = _strip_accents(s)
    s = s.upper()
    # remove pontuação
    s = re.sub(r"[^\w\s]", " ", s)
    # remove múltiplos espaços
    s = re.sub(r"\s+", " ", s).strip()
    # remove sufixos comuns no final (pode repetir)
    parts = s.split()
    while parts and parts[-1] in {x.upper() for x in _SUFFIXES}:
        parts.pop()
    return " ".join(parts).strip()


def split_companies_param(text: str) -> List[str]:
    raw = (text or "").replace(",", "\n").split("\n")
    out = []
    seen = set()
    for x in raw:
        x = x.strip()
        if not x:
            continue
        k = normalize_company_name(x)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def parse_person_name(person: str) -> Tuple[Optional[str], str]:
    """
    Person Name: "11086 - Dom Digital"
    -> ("11086", "Dom Digital")
    Se não tiver '-', tenta achar um prefixo numérico.
    """
    s = str(person or "").strip()
    if not s:
        return None, ""
    if " - " in s:
        left, right = s.split(" - ", 1)
        acc = left.strip()
        comp = right.strip()
        return (acc if acc else None), comp
    if "-" in s:
        left, right = s.split("-", 1)
        acc = left.strip()
        comp = right.strip()
        return (acc if acc else None), comp
    # fallback: primeiro token numérico
    m = re.match(r"^\s*(\d+)\s+(.*)$", s)
    if m:
        return m.group(1), m.group(2).strip()
    return None, s


# =========================
# CSV readers
# =========================

def read_csv_flex(uploaded_file) -> pd.DataFrame:
    """
    Tenta ler CSV com delimitadores comuns.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    content = uploaded_file.getvalue()
    if isinstance(content, bytes):
        raw = content.decode("utf-8", errors="replace")
    else:
        raw = str(content)

    # tenta separadores: ; , \t
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(StringIO(raw), sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue

    # fallback pandas
    try:
        return pd.read_csv(StringIO(raw))
    except Exception:
        return pd.DataFrame()


def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Lida com formatos como "3 jan., 2026", "3 jan, 2026", "2026-01-03", etc.
    """
    # 1) tentativa direta
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.notna().any():
        return dt

    # 2) limpar "jan." etc
    cleaned = (
        s.astype(str)
        .str.replace(r"\.", "", regex=True)
        .str.replace("jan", "jan", regex=False)
        .str.replace("fev", "feb", regex=False)
        .str.replace("mar", "mar", regex=False)
        .str.replace("abr", "apr", regex=False)
        .str.replace("mai", "may", regex=False)
        .str.replace("jun", "jun", regex=False)
        .str.replace("jul", "jul", regex=False)
        .str.replace("ago", "aug", regex=False)
        .str.replace("set", "sep", regex=False)
        .str.replace("out", "oct", regex=False)
        .str.replace("nov", "nov", regex=False)
        .str.replace("dez", "dec", regex=False)
    )
    return pd.to_datetime(cleaned, errors="coerce", dayfirst=True)


# =========================
# Build logic (like backend service)
# =========================

def build_checklist_and_cards(
    facts: pd.DataFrame,
    companies_filter: List[str],
    thresholds: Thresholds,
) -> Tuple[str, List[Dict], List[Dict], List[Dict]]:
    """
    facts columns required:
      date (datetime64), account_id (str), company_name (str), company_key (str), total (float)
    returns: (day_ref_iso, checklist_rows, alerts_rows, cards_rows)
    """
    if facts.empty:
        return "", [], [], []

    if companies_filter:
        facts = facts[facts["company_key"].isin(set(companies_filter))].copy()

    if facts.empty:
        return "", [], [], []

    day_ref = facts["date"].max().date()
    d1 = day_ref - timedelta(days=1)

    base_acc = (
        facts.groupby(["account_id", "company_key", "company_name", "date"], as_index=False)["total"]
        .sum()
    )
    base_comp = (
        base_acc.groupby(["company_key", "company_name", "date"], as_index=False)["total"]
        .sum()
    )

    active_start = day_ref - timedelta(days=30)
    active_end = d1

    def accounts_lists(company_key: str, start: date, end: date) -> Tuple[List[str], List[str]]:
        mask = (
            (base_acc["company_key"] == company_key)
            & (base_acc["date"].dt.date >= start)
            & (base_acc["date"].dt.date <= end)
        )
        df = base_acc.loc[mask, ["account_id", "total"]].copy()
        if df.empty:
            return [], []
        sums = df.groupby("account_id", as_index=False)["total"].sum()
        sums["account_id"] = sums["account_id"].astype(str)
        active = sorted(set(sums.loc[sums["total"] > 0, "account_id"].tolist()))
        zero = sorted(set(sums.loc[sums["total"] <= 0, "account_id"].tolist()))
        active_set = set(active)
        zero = [a for a in zero if a not in active_set]
        return active, zero

    def sum_window_company(company_key: str, days: int) -> float:
        start = (day_ref - timedelta(days=days))
        mask = (
            (base_comp["company_key"] == company_key)
            & (base_comp["date"].dt.date >= start)
            & (base_comp["date"].dt.date <= d1)
        )
        return float(base_comp.loc[mask, "total"].sum())

    def today_total_company(company_key: str) -> float:
        mask = (base_comp["company_key"] == company_key) & (base_comp["date"].dt.date == day_ref)
        return float(base_comp.loc[mask, "total"].sum())

    checklist_rows: List[Dict] = []

    for company_key in sorted(base_comp["company_key"].unique()):
        names = base_comp.loc[base_comp["company_key"] == company_key, "company_name"].dropna()
        company_name = str(names.iloc[-1]) if not names.empty else company_key

        active_accounts, zero_accounts = accounts_lists(company_key, active_start, active_end)
        if not active_accounts:
            continue

        today_total = today_total_company(company_key)

        sum7 = sum_window_company(company_key, 7)
        sum15 = sum_window_company(company_key, 15)
        sum30 = sum_window_company(company_key, 30)

        avg7 = _safe_div(sum7, 7)
        avg15 = _safe_div(sum15, 15)
        avg30 = _safe_div(sum30, 30)

        var7 = calc_var(today_total, avg7)
        var15 = calc_var(today_total, avg15)
        var30 = calc_var(today_total, avg30)

        status = calc_status(var30, thresholds=thresholds)
        obs = calc_obs(status)

        checklist_rows.append({
            "company_key": company_key,
            "company_name": company_name,
            "accounts_list": ", ".join(active_accounts),
            "accounts_zero_list": ", ".join(zero_accounts),
            "accounts_zero_count": len(zero_accounts),

            "today_total": float(today_total),
            "avg_7d": float(avg7),
            "avg_15d": float(avg15),
            "avg_30d": float(avg30),

            "today_total_i": _round_int_nearest(today_total),
            "avg_7d_i": _round_int_nearest(avg7),
            "avg_15d_i": _round_int_nearest(avg15),
            "avg_30d_i": _round_int_nearest(avg30),

            "var_7d": float(var7),
            "var_15d": float(var15),
            "var_30d": float(var30),

            "var_7d_pct": _pct_int_ceil_ratio(var7),
            "var_15d_pct": _pct_int_ceil_ratio(var15),
            "var_30d_pct": _pct_int_ceil_ratio(var30),

            "status": status,
            "obs": obs,
            "day_ref": day_ref.isoformat(),
        })

    checklist_rows.sort(key=lambda r: (severity_rank(r["status"]), r["company_name"]))

    alerts_rows = [r for r in checklist_rows if r["status"] != "Normal"]

    fmt_id_day = day_ref.strftime("%Y%m%d")

    def pct_int_str(ratio: float) -> str:
        return f"{_pct_int_ceil_ratio(ratio)}%"

    cards_rows: List[Dict] = []
    for r in alerts_rows:
        card_id = f"{fmt_id_day}-{r['company_key']}"
        msg = (
            f"ALERTA: {r['status']}\n"
            f"Empresa: {r['company_name']}\n"
            f"Data: {r['day_ref']}\n"
            f"Motivo: {r['obs']}\n"
            f"Total(D): {r['today_total_i']}\n"
            f"Médias: 7d={r['avg_7d_i']} | 15d={r['avg_15d_i']} | 30d={r['avg_30d_i']}\n"
            f"Variação: vs30={pct_int_str(r['var_30d'])} | vs15={pct_int_str(r['var_15d'])} | vs7={pct_int_str(r['var_7d'])}\n"
            f"Contas (ativas): {r['accounts_list']}\n"
            f"Contas zeradas: {r['accounts_zero_count']} ({r['accounts_zero_list']})"
        )
        cards_rows.append({
            "card_id": card_id,
            "day_ref": r["day_ref"],
            "empresa": r["company_name"],
            "status": r["status"],
            "severidade": severity_rank(r["status"]),
            "motivo": r["obs"],
            "accounts_list": r["accounts_list"],
            "accounts_zero_count": r["accounts_zero_count"],
            "accounts_zero_list": r["accounts_zero_list"],
            "today_total_i": r["today_total_i"],
            "avg_7d_i": r["avg_7d_i"],
            "avg_15d_i": r["avg_15d_i"],
            "avg_30d_i": r["avg_30d_i"],
            "var_7d_pct": r["var_7d_pct"],
            "var_15d_pct": r["var_15d_pct"],
            "var_30d_pct": r["var_30d_pct"],
            "mensagem_discord": msg,
        })

    return day_ref.isoformat(), checklist_rows, alerts_rows, cards_rows


# =========================
# BaaS processing (optional)
# =========================

def build_baas_index(df_baas: pd.DataFrame) -> Dict[str, Dict]:
    """
    Returns:
      account_id(str) -> meta dict {posicao, conta, agencia, nome, plano, saldo_bloqueado, baas}
    Handles different separators/headers from Excel exports.
    """
    if df_baas.empty:
        return {}

    # normalize headers (strip)
    df_baas = df_baas.copy()
    df_baas.columns = [str(c).strip() for c in df_baas.columns]

    # expected columns
    col_map = {}
    # tolerate minor header differences
    for c in df_baas.columns:
        ck = c.strip().lower()
        if ck in {"posicao", "posição"}:
            col_map["posicao"] = c
        elif ck == "conta":
            col_map["conta"] = c
        elif ck in {"agencia", "agência"}:
            col_map["agencia"] = c
        elif ck == "nome":
            col_map["nome"] = c
        elif ck == "plano":
            col_map["plano"] = c
        elif ck == "saldo bloqueado" or ck == "saldo_bloqueado":
            col_map["saldo_bloqueado"] = c
        elif ck == "baas":
            col_map["baas"] = c

    if "conta" not in col_map or "saldo_bloqueado" not in col_map:
        return {}

    # parse
    df_baas["__conta"] = df_baas[col_map["conta"]].astype(str).str.strip()
    df_baas["__saldo_bloq"] = pd.to_numeric(df_baas[col_map["saldo_bloqueado"]], errors="coerce").fillna(0.0)

    idx: Dict[str, Dict] = {}
    for _, r in df_baas.iterrows():
        acc = str(r["__conta"]).strip()
        if not acc:
            continue
        idx[acc] = {
            "posicao": int(r[col_map["posicao"]]) if "posicao" in col_map and pd.notna(r[col_map["posicao"]]) else None,
            "conta": acc,
            "agencia": str(r[col_map["agencia"]]).strip() if "agencia" in col_map and pd.notna(r[col_map["agencia"]]) else None,
            "nome": str(r[col_map["nome"]]).strip() if "nome" in col_map and pd.notna(r[col_map["nome"]]) else None,
            "plano": str(r[col_map["plano"]]).strip() if "plano" in col_map and pd.notna(r[col_map["plano"]]) else None,
            "saldo_bloqueado": float(r["__saldo_bloq"]),
            "baas": str(r[col_map["baas"]]).strip() if "baas" in col_map and pd.notna(r[col_map["baas"]]) else None,
        }
    return idx


def annotate_with_baas(
    checklist_rows: List[Dict],
    cards_rows: List[Dict],
    baas_index: Dict[str, Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Adds:
      - to checklist row: caution_has_block (bool), caution_accounts (list[str]), caution_total (float)
      - to cards: caution_has_block (bool), caution_total (float), and injects a line in mensagem_discord
    Also returns a per-account list to render the BaaS checklist table (only accounts used by filtered companies).
    """
    if not checklist_rows:
        return checklist_rows, cards_rows, []

    # account -> has block?
    def has_block(acc: str) -> Tuple[bool, float]:
        m = baas_index.get(str(acc))
        if not m:
            return False, 0.0
        v = float(m.get("saldo_bloqueado") or 0.0)
        return (v > 0.0), v

    # prepare per-account rows (only those present in checklist accounts_list)
    baas_rows: List[Dict] = []
    seen_acc = set()

    # map company_name by key for BaaS table display
    by_key = {r["company_key"]: r["company_name"] for r in checklist_rows}

    # annotate checklist
    for r in checklist_rows:
        accounts = [a.strip() for a in str(r.get("accounts_list") or "").split(",") if a.strip()]
        blocked = []
        total_bloq = 0.0

        for acc in accounts:
            ok, v = has_block(acc)
            if ok:
                blocked.append(acc)
                total_bloq += v

            # build BaaS row if in baas_index
            if acc in baas_index and acc not in seen_acc:
                seen_acc.add(acc)
                meta = baas_index[acc]
                bloq = float(meta.get("saldo_bloqueado") or 0.0)
                baas_rows.append({
                    "posicao": meta.get("posicao"),
                    "conta": acc,
                    "agencia": meta.get("agencia"),
                    "empresa": meta.get("nome") or by_key.get(r["company_key"]) or "",
                    "saldo_bloqueado": bloq,
                    "status": "🔒 Bloqueio cautelar" if bloq > 0 else "✅ Sem bloqueio",
                })

        r["caution_has_block"] = bool(blocked)
        r["caution_accounts"] = blocked
        r["caution_total"] = float(total_bloq)

    # annotate cards by company_key join (cards contain company name only, so use name match)
    ck_by_name = {normalize_company_name(r["company_name"]): r for r in checklist_rows}

    for c in cards_rows:
        key = normalize_company_name(c.get("empresa") or "")
        row = ck_by_name.get(key)
        hasb = bool(row.get("caution_has_block")) if row else False
        total = float(row.get("caution_total") or 0.0) if row else 0.0
        c["caution_has_block"] = hasb
        c["caution_total"] = total

        # Inject an extra line in the message (keep simple)
        msg = c.get("mensagem_discord") or ""
        tag = "🔒 Bloqueio cautelar: SIM" if hasb else "🔓 Bloqueio cautelar: NÃO"
        if "\n🔒 Bloqueio cautelar:" not in msg and "\n🔓 Bloqueio cautelar:" not in msg:
            msg = msg + f"\n{tag}"
        c["mensagem_discord"] = msg

    # sort baas rows by (has block first, then saldo desc, then posicao)
    def _baas_sort_key(x: Dict):
        hasb = 1 if str(x.get("status", "")).startswith("🔒") else 0
        return (-hasb, -(x.get("saldo_bloqueado") or 0.0), (x.get("posicao") or 10**9))

    baas_rows.sort(key=_baas_sort_key)
    return checklist_rows, cards_rows, baas_rows


# =========================
# UI helpers
# =========================

def fmt_int_pt(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        n = 0
    return f"{n:,}".replace(",", ".")


def fmt_money_pt(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        v = 0.0
    # 2 casas, separador brasileiro
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def kpi_box(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:14px;min-height:92px;">
          <div style="color:#aab4c8;font-size:12px;line-height:1.2">{label}</div>
          <div style="font-size:22px;font-weight:700;margin-top:8px;line-height:1.1">{value}</div>
          <div style="color:#aab4c8;font-size:12px;margin-top:8px;line-height:1.2">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def paginate(df: pd.DataFrame, page_size: int, key_prefix: str) -> pd.DataFrame:
    total = len(df)
    if total <= page_size:
        return df
    pages = max(1, int(math.ceil(total / page_size)))
    page = st.number_input(
        "Página",
        min_value=1,
        max_value=pages,
        value=1,
        step=1,
        key=f"{key_prefix}_page",
    )
    start = (page - 1) * page_size
    end = start + page_size
    st.caption(f"Mostrando {start+1}–{min(end, total)} de {total}")
    return df.iloc[start:end].copy()


# =========================
# App UI
# =========================

st.title("Alertas (CSV → Checklist + Cards)")

with st.sidebar:
    st.header("Parâmetros")
    thresholds = Thresholds(
        queda_critica=float(st.number_input("Queda crítica", value=-0.60, step=0.01, format="%.2f")),
        aumento_relevante=float(st.number_input("Aumento relevante", value=0.80, step=0.01, format="%.2f")),
        investigar_abs=float(st.number_input("Investigar abs", value=0.30, step=0.01, format="%.2f")),
    )
    st.divider()
    st.caption("Uploads")
    up_trans = st.file_uploader("CSV Transações", type=["csv"], key="csv_trans")
    up_baas = st.file_uploader("CSV BaaS (opcional)", type=["csv"], key="csv_baas")

st.markdown("Cole as empresas e envie o CSV. O CSV BaaS é opcional (bloqueio cautelar).")

companies_text = st.text_area(
    "Empresas (uma por linha ou vírgula)",
    height=120,
    placeholder="Ex: Dom Digital, PIX NA HORA\nou uma por linha",
)

companies_filter = split_companies_param(companies_text)

# =========================
# Read & validate trans CSV
# =========================
if not up_trans:
    st.info("⬅️ Envie o CSV de transações na barra lateral para iniciar.")
    st.stop()

df_raw = read_csv_flex(up_trans)

if df_raw.empty:
    st.error("CSV de transações lido como vazio. Verifique o separador (vírgula/;).")
    st.stop()

df_raw.columns = [str(c).strip() for c in df_raw.columns]

# Required columns (as per your print)
# A: Data Transacao: Day
# B: Movement Account ID
# C: Person Name
# D: CREDIT UN
# E: DEBIT UN
required = ["Data Transacao: Day", "Movement Account ID", "Person Name", "CREDIT UN", "DEBIT UN"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"CSV de transações inválido. Colunas ausentes: {missing}")
    st.write("Colunas encontradas:", list(df_raw.columns))
    st.stop()

df = df_raw.copy()

df["date"] = parse_date_series(df["Data Transacao: Day"])
df["account_id"] = df["Movement Account ID"].astype(str).str.strip()

acc_from_person, comp_from_person = zip(*df["Person Name"].apply(parse_person_name))
df["account_id_person"] = pd.Series(acc_from_person).astype("string")
df["company_name"] = pd.Series(comp_from_person).astype("string")

# prefer Movement Account ID as account_id (you said it's the account number)
# but keep a fallback in case Movement Account ID is empty
df["account_id"] = df["account_id"].where(df["account_id"].str.len() > 0, df["account_id_person"].fillna(""))

# credit/debit
df["credit"] = pd.to_numeric(df["CREDIT UN"], errors="coerce").fillna(0.0)
df["debit"] = pd.to_numeric(df["DEBIT UN"], errors="coerce").fillna(0.0)
df["total"] = df["credit"] + df["debit"]

# company key normalization
df["company_key"] = df["company_name"].apply(normalize_company_name)

# drop invalids
df = df.dropna(subset=["date"])
df = df[df["company_key"].astype(str).str.len() > 0].copy()

if df.empty:
    st.error("CSV de transações ficou vazio após normalização (datas/empresa).")
    st.stop()

# Apply companies filter (IMPORTANT)
if companies_filter:
    df = df[df["company_key"].isin(set(companies_filter))].copy()

if df.empty:
    st.warning("Nenhuma empresa encontrada após filtro. Verifique nomes digitados.")
    st.stop()

# Build facts for checklist builder
facts = df[["date", "account_id", "company_name", "company_key", "total"]].copy()

day_ref_iso, checklist_rows, alerts_rows, cards_rows = build_checklist_and_cards(
    facts=facts,
    companies_filter=companies_filter,
    thresholds=thresholds,
)

# =========================
# Optional BaaS
# =========================
baas_rows = []
caution_accounts_count = 0

if up_baas:
    df_baas = read_csv_flex(up_baas)
    baas_index = build_baas_index(df_baas)
    checklist_rows, cards_rows, baas_rows = annotate_with_baas(checklist_rows, cards_rows, baas_index)

    # KPI: unique accounts with bloqueio > 0 among the matched accounts
    caution_accounts_count = sum(
        1 for r in (baas_rows or []) if str(r.get("status", "")).startswith("🔒")
    )

# =========================
# KPIs row
# =========================
k1, k2, k3, k4, k5 = st.columns(5)

critical = sum(1 for r in checklist_rows if r.get("status") == "Escalar (queda)")
alerts_count = sum(1 for r in checklist_rows if r.get("status") and r.get("status") != "Normal")
companies_count = len(checklist_rows)

with k1:
    kpi_box("Dia de referência", day_ref_iso or "—", "Aguardando processamento" if not day_ref_iso else "")
with k2:
    kpi_box("Empresas no checklist", fmt_int_pt(companies_count), "Filtradas pelo input")
with k3:
    kpi_box("Alertas", fmt_int_pt(alerts_count), "status != Normal")
with k4:
    kpi_box("Críticos", fmt_int_pt(critical), "Escalar (queda)")
with k5:
    kpi_box("Contas com bloqueio", fmt_int_pt(caution_accounts_count), "via CSV BaaS")

st.divider()

# =========================
# Charts
# =========================
st.subheader("Visão Geral")

df_check = pd.DataFrame(checklist_rows)
if not df_check.empty:
    cA, cB = st.columns(2)

    # Status distribution
    with cA:
        st.caption("Alertas por status")
        dist = df_check.groupby("status", as_index=False).size()
        fig1 = px.bar(dist, x="status", y="size", title=None)
        st.plotly_chart(fig1, use_container_width=True)

    # Top var30 (piores/melhores)
    with cB:
        st.caption("Top variação 30d (piores/melhores)")
        topN = st.selectbox("Top", [8, 10, 15], index=0)
        tmp = df_check[["company_name", "var_30d"]].copy()
        tmp["var_30d"] = pd.to_numeric(tmp["var_30d"], errors="coerce").fillna(0.0)
        worst = tmp.sort_values("var_30d", ascending=True).head(int(topN))
        best = tmp.sort_values("var_30d", ascending=False).head(int(topN))
        merged = pd.concat([worst, best], axis=0)
        fig2 = px.bar(merged, x="company_name", y="var_30d", title=None)
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =========================
# Checklist (paginated)
# =========================
st.subheader("Checklist")

# Filters
fcol1, fcol2 = st.columns([2, 1])
with fcol1:
    q = st.text_input("Buscar empresa", value="", placeholder="Buscar...")
with fcol2:
    status_filter = st.selectbox(
        "Filtrar status",
        ["Todos", "Escalar (queda)", "Investigar", "Gerenciar (aumento)", "Normal"],
        index=0
    )

df_show = df_check.copy()

if q.strip():
    qq = q.strip().upper()
    df_show = df_show[df_show["company_name"].astype(str).str.upper().str.contains(qq, na=False)]

if status_filter != "Todos":
    df_show = df_show[df_show["status"] == status_filter]

# Add lock icon column (no text)
if "caution_has_block" in df_show.columns:
    df_show["🔒"] = df_show["caution_has_block"].apply(lambda x: "🔒" if bool(x) else "")
else:
    df_show["🔒"] = ""

# Map integer columns to display
df_show_disp = pd.DataFrame({
    "Status": df_show["status"],
    "🔒": df_show["🔒"],
    "Empresa": df_show["company_name"],
    "Hoje": df_show.get("today_total_i", 0),
    "Média 7d": df_show.get("avg_7d_i", 0),
    "Média 15d": df_show.get("avg_15d_i", 0),
    "Média 30d": df_show.get("avg_30d_i", 0),
    "Var 30d": df_show.get("var_30d_pct", 0).astype(int).astype(str) + "%",
    "Contas ativas": df_show.get("accounts_list", ""),
    "Motivo": df_show.get("obs", ""),
})

# Sort default: severity, then name
order_map = {"Escalar (queda)": 0, "Investigar": 1, "Gerenciar (aumento)": 2, "Normal": 3}
df_show_disp["_sev"] = df_show_disp["Status"].map(order_map).fillna(9).astype(int)
df_show_disp = df_show_disp.sort_values(["_sev", "Empresa"]).drop(columns=["_sev"])

df_page = paginate(df_show_disp.reset_index(drop=True), PAGE_SIZE_CHECKLIST, "checklist")
st.dataframe(df_page, use_container_width=True, hide_index=True)

st.divider()

# =========================
# Checklist BaaS (per account) - paginated 10
# =========================
st.subheader("Checklist Bloqueio Cautelar (BaaS)")

if not baas_rows:
    st.caption("CSV BaaS não enviado, ou não foi possível mapear colunas (Conta + Saldo Bloqueado).")
else:
    df_baas_show = pd.DataFrame(baas_rows)

    # Keep only accounts that belong to filtered companies:
    # (Already ensured when building baas_rows; just in case)
    if not df_baas_show.empty:
        # Format
        df_baas_show["Saldo Bloqueado"] = df_baas_show["saldo_bloqueado"].apply(fmt_money_pt)
        df_baas_show_disp = pd.DataFrame({
            "Posição": df_baas_show["posicao"],
            "Conta": df_baas_show["conta"],
            "Agência": df_baas_show["agencia"],
            "Empresa": df_baas_show["empresa"],
            "Saldo Bloqueado": df_baas_show["Saldo Bloqueado"],
            "Status": df_baas_show["status"],
        })
        df_baas_page = paginate(df_baas_show_disp.reset_index(drop=True), PAGE_SIZE_BAAS, "baas")
        st.dataframe(df_baas_page, use_container_width=True, hide_index=True)

st.divider()

# =========================
# Cards
# =========================
st.subheader("Cards de Alertas")

df_cards = pd.DataFrame(cards_rows)

if df_cards.empty:
    st.info("Nenhum alerta (status != Normal) encontrado.")
else:
    card_filter = st.selectbox(
        "Filtrar cards",
        ["Todos", "Escalar (queda)", "Investigar", "Gerenciar (aumento)"],
        index=0
    )
    if card_filter != "Todos":
        df_cards = df_cards[df_cards["status"] == card_filter]

    for _, c in df_cards.iterrows():
        title = f"{c.get('status')} — {c.get('empresa')}"
        lock = " 🔒" if bool(c.get("caution_has_block", False)) else ""
        with st.expander(title + lock, expanded=False):
            st.code(str(c.get("mensagem_discord", "")), language="text")
            if bool(c.get("caution_has_block", False)):
                st.caption(f"Saldo bloqueado (soma contas): R$ {fmt_money_pt(c.get('caution_total', 0.0))}")