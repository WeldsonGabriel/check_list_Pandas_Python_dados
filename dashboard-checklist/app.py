# app.py — Streamlit dashboard (Checklist + BaaS) — v=20260203-1
# Regras:
# - Só processa quando clicar em "Processar"
# - Filtra empresas pelo textarea (vírgula ou uma por linha)
# - CSV Transações: col0=data, col2=person ("NNNN - Empresa"), col3=credit, col4=debit
# - day_ref = maior data do CSV
# - Médias 7/15/30 em [day_ref - N, day_ref-1]
# - Empresa só entra se tiver >=1 conta com soma_total > 0 no período [day_ref-30, day_ref-1]
# - Inteiros: arredonda para unidade mais próxima (0.5 pra cima)
# - var_30d em % no gráfico (e KPI/coluna)
# - BaaS (opcional): casa por CONTA com accounts_list; gera tabela por conta + KPI "Contas com bloqueio"
# - Cores: status (Escalar=vermelho, Investigar=amarelo, Gerenciar=azul, Normal=verde)
#          var_30d: <0 vermelho, >0 verde

import io
import math
import re
import unicodedata
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# Config
# =========================
st.set_page_config(page_title="Alertas - Checklist + BaaS", layout="wide")


# =========================
# Normalização / Helpers
# =========================
SUFFIXES = {
    " LTDA", " LTDA.", " S/A", " SA", " S.A", " S.A.", " ME", " EPP", " MEI",
    " EIRELI", " - ME", " - EPP", " - MEI"
}

def normalize_company_name(s: str) -> str:
    """Uppercase, remove acentos, pontuação, sufixos comuns."""
    if s is None:
        return ""
    s = str(s).strip().upper()

    # remove acentos
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # remove pontuação (mantém espaços e números/letras)
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # remove sufixos comuns (iterativo)
    changed = True
    while changed:
        changed = False
        for suf in list(SUFFIXES):
            if s.endswith(suf):
                s = s[: -len(suf)].strip()
                changed = True

    return s


def split_companies_input(text: str) -> list[str]:
    raw = (text or "")
    parts = [p.strip() for p in re.split(r"[\n,]+", raw) if p.strip()]
    seen = set()
    out = []
    for p in parts:
        k = normalize_company_name(p)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def round_int_nearest(x) -> int:
    """1.1 -> 1 ; 1.8 -> 2 ; 0.5 -> 1 ; -1.5 -> -1 (simétrico)"""
    try:
        v = float(x)
    except Exception:
        return 0
    if v >= 0:
        return int(math.floor(v + 0.5))
    return int(math.ceil(v - 0.5))


def fmt_int_pt(x) -> str:
    n = round_int_nearest(x)
    s = f"{n}"
    # separador de milhar com ponto
    return re.sub(r"\B(?=(\d{3})+(?!\d))", ".", s)


def fmt_pct1(x) -> str:
    """x é razão (ex.: -0.6012). Exibe % com 1 casa."""
    try:
        v = float(x) * 100.0
    except Exception:
        v = 0.0
    return f"{v:.1f}%"


def safe_div(n, d) -> float:
    try:
        n = float(n)
        d = float(d)
    except Exception:
        return 0.0
    if d == 0:
        return 0.0
    return n / d


def calc_var(today: float, avg: float) -> float:
    # var ratio: (today - avg) / avg, com proteção
    if avg == 0:
        return 0.0 if today == 0 else 1.0
    return (today - avg) / avg


def calc_status(var30: float, queda_critica: float, aumento_relevante: float, investigar_abs: float) -> str:
    if var30 <= queda_critica:
        return "Escalar (queda)"
    if abs(var30) >= investigar_abs and var30 < 0:
        return "Investigar"
    if var30 >= aumento_relevante:
        return "Gerenciar (aumento)"
    return "Normal"


def calc_obs(status: str) -> str:
    if status == "Escalar (queda)":
        return "Queda crítica vs média histórica"
    if status == "Investigar":
        return "Queda relevante — investigar"
    if status == "Gerenciar (aumento)":
        return "Aumento relevante — gerenciar capacidade"
    return "Dentro do padrão"


def severity_rank(status: str) -> int:
    # menor = mais crítico
    if status == "Escalar (queda)":
        return 0
    if status == "Investigar":
        return 1
    if status == "Gerenciar (aumento)":
        return 2
    if status == "Normal":
        return 3
    return 9


def status_color(status: str) -> str:
    s = (status or "").strip()
    if s == "Escalar (queda)":
        return "#e74c3c"
    if s == "Investigar":
        return "#f1c40f"
    if s == "Gerenciar (aumento)":
        return "#3498db"
    if s == "Normal":
        return "#2ecc71"
    return "#aab4c8"


def var_color(v: float) -> str:
    try:
        n = float(v)
    except Exception:
        return "#aab4c8"
    if n < 0:
        return "#e74c3c"
    if n > 0:
        return "#2ecc71"
    return "#aab4c8"


# =========================
# Leitura CSV Transações
# =========================
def parse_transactions_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Esperado:
      col0 = data
      col2 = person "NNNN - Empresa"
      col3 = credit
      col4 = debit
    Retorna DataFrame normalizado:
      date (date), account_id (str), company_name (str), company_key (str), total (float)
    """
    if not file_bytes:
        return pd.DataFrame()

    # tenta ; e , (muitos CSVs BR)
    df = None
    for sep in [",", ";", "\t"]:
        try:
            df_try = pd.read_csv(io.BytesIO(file_bytes), header=None, sep=sep, engine="python")
            if df_try.shape[1] >= 5:
                df = df_try
                break
        except Exception:
            continue

    if df is None or df.empty:
        return pd.DataFrame()

    # pega colunas
    col_date = df.iloc[:, 0]
    col_person = df.iloc[:, 2].astype(str)
    col_credit = pd.to_numeric(df.iloc[:, 3], errors="coerce").fillna(0.0)
    col_debit = pd.to_numeric(df.iloc[:, 4], errors="coerce").fillna(0.0)

    # data
    def to_date(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        # tenta vários formatos comuns
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y%m%d"]:
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                pass
        # fallback: pandas
        try:
            return pd.to_datetime(s, dayfirst=True).date()
        except Exception:
            return None

    dates = col_date.apply(to_date)
    mask_ok = dates.notna() & col_person.notna()
    df2 = pd.DataFrame({
        "date": dates[mask_ok].astype(object),
        "person": col_person[mask_ok].astype(str),
        "credit": col_credit[mask_ok].astype(float),
        "debit": col_debit[mask_ok].astype(float),
    })
    if df2.empty:
        return pd.DataFrame()

    # parse "NNNN - Empresa"
    # account_id = parte esquerda do traço; company_name = direita
    def split_person(s: str):
        s = (s or "").strip()
        # separa no primeiro " - " se existir; senão tenta "-"
        if " - " in s:
            left, right = s.split(" - ", 1)
        elif "-" in s:
            left, right = s.split("-", 1)
        else:
            # se não tiver traço, tenta pegar número inicial
            m = re.match(r"^\s*(\d+)\s+(.*)$", s)
            if m:
                left, right = m.group(1), m.group(2)
            else:
                left, right = "", s
        acc = re.sub(r"\D", "", left)  # só números
        comp = right.strip()
        return acc, comp

    parsed = df2["person"].apply(split_person)
    df2["account_id"] = parsed.apply(lambda x: x[0] if x else "")
    df2["company_name"] = parsed.apply(lambda x: x[1] if x else "")
    df2["company_key"] = df2["company_name"].apply(normalize_company_name)

    df2["total"] = (df2["credit"].abs() + df2["debit"].abs()).astype(float)

    df_out = df2[["date", "account_id", "company_name", "company_key", "total"]].copy()
    df_out = df_out[df_out["company_key"].astype(str).str.len() > 0]
    df_out["account_id"] = df_out["account_id"].astype(str)
    df_out = df_out[df_out["date"].notna()]
    return df_out


# =========================
# Checklist (core)
# =========================
def accounts_lists_for_company(base_acc: pd.DataFrame, company_key: str, start: date, end: date):
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
    zero = sums.loc[sums["total"] <= 0, "account_id"].tolist()

    active = sorted(set(active))
    zero = sorted(set(zero))
    # exclusão mútua
    active_set = set(active)
    zero = [a for a in zero if a not in active_set]
    return active, zero


def build_checklist(facts: pd.DataFrame, companies_keys: list[str], thresholds: dict):
    if facts.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    # filtra empresas se informado
    if companies_keys:
        facts = facts[facts["company_key"].isin(set(companies_keys))].copy()

    if facts.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    day_ref = max(facts["date"])
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

    def sum_window_company(company_key: str, days: int) -> float:
        start = day_ref - timedelta(days=days)
        mask = (
            (base_comp["company_key"] == company_key)
            & (base_comp["date"] >= start)
            & (base_comp["date"] <= d1)
        )
        return float(base_comp.loc[mask, "total"].sum())

    def today_total_company(company_key: str) -> float:
        mask = (base_comp["company_key"] == company_key) & (base_comp["date"] == day_ref)
        return float(base_comp.loc[mask, "total"].sum())

    rows = []
    for company_key in sorted(base_comp["company_key"].unique()):
        names = base_comp.loc[base_comp["company_key"] == company_key, "company_name"].dropna()
        company_name = str(names.iloc[-1]) if not names.empty else company_key

        active_accounts, zero_accounts = accounts_lists_for_company(
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
            queda_critica=float(thresholds["queda_critica"]),
            aumento_relevante=float(thresholds["aumento_relevante"]),
            investigar_abs=float(thresholds["investigar_abs"]),
        )
        obs = calc_obs(status)

        rows.append({
            "company_key": company_key,
            "company_name": company_name,
            "accounts_list": ", ".join(active_accounts),
            "accounts_zero_list": ", ".join(zero_accounts),
            "accounts_zero_count": len(zero_accounts),

            # floats
            "today_total": float(today_total),
            "avg_7d": float(avg7),
            "avg_15d": float(avg15),
            "avg_30d": float(avg30),
            "var_7d": float(var7),
            "var_15d": float(var15),
            "var_30d": float(var30),

            # ints (para UI)
            "today_total_i": round_int_nearest(today_total),
            "avg_7d_i": round_int_nearest(avg7),
            "avg_15d_i": round_int_nearest(avg15),
            "avg_30d_i": round_int_nearest(avg30),

            "status": status,
            "obs": obs,
            "day_ref": day_ref.isoformat(),
        })

    df_checklist = pd.DataFrame(rows)
    if df_checklist.empty:
        return day_ref, pd.DataFrame(), pd.DataFrame()

    df_checklist = df_checklist.sort_values(
        by=["status", "company_name"],
        key=lambda s: s.map(lambda x: severity_rank(str(x))) if s.name == "status" else s
    )

    df_alerts = df_checklist[df_checklist["status"] != "Normal"].copy()
    return day_ref, df_checklist, df_alerts


# =========================
# BaaS parsing + merge
# =========================
def parse_baas_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Colunas esperadas (nomes variam):
      Posição, Conta, Agência, Nome, Plano, Saldo, Saldo Bloqueado, BaaS

    Retorna normalizado:
      pos (int|None), account_id (str), agency (str), name (str), blocked (float), plan (str), baas (str)
    """
    if not file_bytes:
        return pd.DataFrame()

    df = None
    # tenta ; e , e \t com header
    for sep in [",", ";", "\t"]:
        try:
            df_try = pd.read_csv(io.BytesIO(file_bytes), sep=sep, engine="python")
            if df_try.shape[1] >= 3:
                df = df_try
                break
        except Exception:
            continue

    if df is None or df.empty:
        return pd.DataFrame()

    # normaliza nomes de coluna
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    def pick_col(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    c_pos = pick_col("posição", "posicao", "pos", "rank")
    c_acc = pick_col("conta", "account", "account_id")
    c_ag = pick_col("agência", "agencia", "agency")
    c_name = pick_col("nome", "empresa", "titular")
    c_plan = pick_col("plano", "plan")
    c_block = pick_col("saldo bloqueado", "saldo_bloqueado", "blocked", "bloqueado", "saldo bloqueio")
    c_baas = pick_col("baas", "b a a s")

    if not c_acc or not c_block:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["pos"] = pd.to_numeric(df[c_pos], errors="coerce") if c_pos else pd.Series([None] * len(df))
    out["account_id"] = df[c_acc].astype(str).apply(lambda x: re.sub(r"\D", "", x))
    out["agency"] = df[c_ag].astype(str).fillna("") if c_ag else ""
    out["name"] = df[c_name].astype(str).fillna("") if c_name else ""
    out["plan"] = df[c_plan].astype(str).fillna("") if c_plan else ""
    out["baas"] = df[c_baas].astype(str).fillna("") if c_baas else ""

    # bloqueado pode vir com vírgula decimal
    def to_float_br(x):
        if pd.isna(x):
            return 0.0
        s = str(x).strip()
        s = s.replace(".", "").replace(",", ".") if re.search(r"\d+,\d+", s) else s
        try:
            return float(s)
        except Exception:
            return 0.0

    out["blocked"] = df[c_block].apply(to_float_br).astype(float)
    out = out[out["account_id"].astype(str).str.len() > 0].copy()
    return out


def enrich_with_baas(df_checklist: pd.DataFrame, facts: pd.DataFrame, df_baas: pd.DataFrame):
    """
    - Faz join por account_id (BaaS) x accounts_list (checklist) via mapa account->company_key
    - Gera:
      - df_baas_filtered (somente contas das empresas do checklist)
      - adiciona no checklist:
          caution_accounts_count
          caution_accounts_list
          caution_blocked_sum
          caution_has_block (bool)
    - KPI: total contas com bloqueio (>0) no universo filtrado
    """
    if df_checklist.empty or facts.empty or df_baas.empty:
        df = df_checklist.copy()
        if not df.empty:
            df["caution_accounts_count"] = 0
            df["caution_accounts_list"] = ""
            df["caution_blocked_sum"] = 0.0
            df["caution_has_block"] = False
        return df, pd.DataFrame(), 0

    # mapa conta -> company_key (usa últimos nomes)
    # OBS: uma conta pertence a uma empresa no seu modelo
    acc_map = (
        facts[["account_id", "company_key", "company_name"]]
        .dropna()
        .drop_duplicates(subset=["account_id"], keep="last")
    )

    # filtra BaaS para contas presentes no checklist
    # primeiro pega set de contas ativas do checklist
    all_accounts = set()
    ck_accounts = df_checklist[["company_key", "accounts_list"]].copy()
    for _, r in ck_accounts.iterrows():
        parts = [p.strip() for p in str(r["accounts_list"] or "").split(",") if p.strip()]
        for a in parts:
            all_accounts.add(re.sub(r"\D", "", a))

    df_b = df_baas.copy()
    df_b = df_b[df_b["account_id"].isin(all_accounts)].copy()
    if df_b.empty:
        df = df_checklist.copy()
        df["caution_accounts_count"] = 0
        df["caution_accounts_list"] = ""
        df["caution_blocked_sum"] = 0.0
        df["caution_has_block"] = False
        return df, pd.DataFrame(), 0

    # adiciona empresa do lado via acc_map
    df_b = df_b.merge(acc_map, on="account_id", how="left")
    df_b = df_b[df_b["company_key"].notna()].copy()

    # status por conta: bloqueio cautelar se blocked > 0
    df_b["has_block"] = df_b["blocked"].astype(float) > 0

    # KPI total contas com bloqueio (no universo filtrado)
    kpi_blocked_accounts = int(df_b["has_block"].sum())

    # tabela por conta (para UI)
    df_baas_filtered = df_b[[
        "pos", "account_id", "agency", "company_name", "blocked", "has_block", "plan", "baas"
    ]].copy()

    # agregado por empresa
    agg = (
        df_baas_filtered.groupby("company_key", as_index=False)
        .agg(
            caution_accounts_count=("has_block", "sum"),
            caution_blocked_sum=("blocked", "sum"),
            caution_accounts_list=("account_id", lambda s: ", ".join(sorted(set([x for x in s.astype(str).tolist() if x]))))
        )
    )

    # mas caution_accounts_list acima lista TODAS as contas do BaaS filtrado, não só bloqueadas
    # vamos corrigir: list só das bloqueadas
    only_blocked = df_baas_filtered[df_baas_filtered["has_block"]].copy()
    agg_list = (
        only_blocked.groupby("company_key", as_index=False)
        .agg(caution_accounts_list=("account_id", lambda s: ", ".join(sorted(set(s.astype(str).tolist())))))
    )
    agg = agg.drop(columns=["caution_accounts_list"]).merge(agg_list, on="company_key", how="left")
    agg["caution_accounts_list"] = agg["caution_accounts_list"].fillna("")
    agg["caution_has_block"] = agg["caution_accounts_count"].fillna(0).astype(int) > 0

    df_out = df_checklist.merge(agg, on="company_key", how="left")
    df_out["caution_accounts_count"] = df_out["caution_accounts_count"].fillna(0).astype(int)
    df_out["caution_blocked_sum"] = df_out["caution_blocked_sum"].fillna(0.0).astype(float)
    df_out["caution_accounts_list"] = df_out["caution_accounts_list"].fillna("")
    df_out["caution_has_block"] = df_out["caution_has_block"].fillna(False).astype(bool)

    return df_out, df_baas_filtered, kpi_blocked_accounts


# =========================
# UI
# =========================
st.title("Alertas (CSV → Checklist + Cards) — Streamlit")

st.markdown(
    """
**Fluxo**
1) Cole as empresas (uma por linha ou separado por vírgula)  
2) Faça upload do CSV de transações (obrigatório)  
3) Faça upload do CSV BaaS (opcional)  
4) Clique em **Processar** (nada roda antes do clique)
"""
)

with st.sidebar:
    st.header("Parâmetros")
    queda_critica = st.number_input("Queda crítica", value=-0.60, step=0.01, format="%.2f")
    aumento_relevante = st.number_input("Aumento relevante", value=0.80, step=0.01, format="%.2f")
    investigar_abs = st.number_input("Investigar abs", value=0.30, step=0.01, format="%.2f")
    top_n = st.selectbox("Top (piores/melhores)", options=[8, 10, 15], index=0)
    st.session_state["top_n"] = int(top_n)

# Form (processa somente ao submit)
with st.form("process_form", clear_on_submit=False):
    companies_text = st.text_area("Empresas", height=140, placeholder="Ex: TREEAL\nEMPRESA X\nou Treeal, Empresa X")
    csv_tx = st.file_uploader("CSV Transações (obrigatório)", type=["csv"], accept_multiple_files=False)
    csv_baas = st.file_uploader("CSV BaaS (opcional)", type=["csv"], accept_multiple_files=False)
    submitted = st.form_submit_button("Processar")

# Estado
if "processed" not in st.session_state:
    st.session_state["processed"] = False
    st.session_state["day_ref"] = None
    st.session_state["df_checklist"] = pd.DataFrame()
    st.session_state["df_alerts"] = pd.DataFrame()
    st.session_state["df_baas_table"] = pd.DataFrame()
    st.session_state["kpi_blocked_accounts"] = 0


if submitted:
    companies_keys = split_companies_input(companies_text)
    if not companies_keys:
        st.error("Informe pelo menos 1 empresa para processar.")
    elif csv_tx is None:
        st.error("Envie o CSV de transações.")
    else:
        facts = parse_transactions_csv(csv_tx.getvalue())
        if facts.empty:
            st.error("CSV de transações lido, mas ficou vazio após normalização. Verifique colunas e separador.")
        else:
            thresholds = {
                "queda_critica": float(queda_critica),
                "aumento_relevante": float(aumento_relevante),
                "investigar_abs": float(investigar_abs),
            }
            day_ref, df_checklist, df_alerts = build_checklist(facts, companies_keys, thresholds)

            # BaaS
            df_baas = pd.DataFrame()
            if csv_baas is not None:
                df_baas = parse_baas_csv(csv_baas.getvalue())

            df_checklist2, df_baas_table, kpi_blocked_accounts = enrich_with_baas(
                df_checklist=df_checklist,
                facts=facts,
                df_baas=df_baas
            )

            st.session_state["processed"] = True
            st.session_state["day_ref"] = day_ref.isoformat() if isinstance(day_ref, date) else (day_ref or "")
            st.session_state["df_checklist"] = df_checklist2
            st.session_state["df_alerts"] = df_alerts
            st.session_state["df_baas_table"] = df_baas_table
            st.session_state["kpi_blocked_accounts"] = int(kpi_blocked_accounts)

# Só mostra resultados se processado
if not st.session_state["processed"]:
    st.info("Aguardando: informe empresas + CSV e clique em **Processar**.")
    st.stop()

df_checklist = st.session_state["df_checklist"].copy()
df_alerts = st.session_state["df_alerts"].copy()
df_baas_table = st.session_state["df_baas_table"].copy()
day_ref = st.session_state["day_ref"] or "—"

# =========================
# KPIs (5)
# =========================
alerts_count = int((df_checklist["status"] != "Normal").sum()) if not df_checklist.empty else 0
critical_count = int((df_checklist["status"] == "Escalar (queda)").sum()) if not df_checklist.empty else 0
companies_count = int(len(df_checklist)) if not df_checklist.empty else 0
blocked_accounts_count = int(st.session_state["kpi_blocked_accounts"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dia de referência", day_ref)
c2.metric("Empresas no checklist", fmt_int_pt(companies_count))
c3.metric("Alertas", fmt_int_pt(alerts_count))
c4.metric("Críticos", fmt_int_pt(critical_count))
c5.metric("Contas com bloqueio", fmt_int_pt(blocked_accounts_count))

st.divider()

# =========================
# GRÁFICOS (cores)
# =========================
colA, colB = st.columns(2)

with colA:
    st.subheader("Alertas por status")

    if df_checklist.empty:
        st.write("Sem dados.")
    else:
        order = ["Escalar (queda)", "Investigar", "Gerenciar (aumento)", "Normal"]
        counts = (
            df_checklist["status"]
            .fillna("Desconhecido")
            .value_counts()
            .rename_axis("status")
            .reset_index(name="count")
        )
        counts["status"] = counts["status"].astype(str)
        counts["order"] = counts["status"].apply(lambda x: order.index(x) if x in order else 999)
        counts = counts.sort_values(["order", "status"]).drop(columns=["order"])

        fig_status = px.bar(counts, x="status", y="count", text="count")
        fig_status.update_traces(
            marker_color=[status_color(s) for s in counts["status"]],
            textposition="outside",
            cliponaxis=False,
        )
        fig_status.update_layout(
            yaxis_title="Empresas",
            xaxis_title="",
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_status, use_container_width=True)

with colB:
    st.subheader("Top variação 30d (piores / melhores)")

    if df_checklist.empty:
        st.write("Sem dados.")
    else:
        top_n = int(st.session_state.get("top_n", 8))
        dfv = df_checklist.copy()
        dfv["var_30d"] = pd.to_numeric(dfv["var_30d"], errors="coerce")
        dfv = dfv.dropna(subset=["var_30d"])

        if dfv.empty:
            st.write("Sem variações numéricas.")
        else:
            worst = dfv.sort_values("var_30d", ascending=True).head(top_n)
            best = dfv.sort_values("var_30d", ascending=False).head(top_n)
            merged = pd.concat([worst, best], axis=0).drop_duplicates(subset=["company_name"], keep="first").copy()
            merged["var_pct"] = merged["var_30d"] * 100.0

            fig_var = px.bar(merged, x="company_name", y="var_pct", text="var_pct")
            fig_var.update_traces(
                marker_color=[var_color(v) for v in merged["var_30d"]],
                texttemplate="%{text:.1f}%",
                textposition="outside",
                cliponaxis=False,
            )
            fig_var.update_layout(
                yaxis_title="Variação (%)",
                xaxis_title="",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            fig_var.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#8892a6")
            st.plotly_chart(fig_var, use_container_width=True)

st.divider()

# =========================
# Checklist (filtrável)
# =========================
st.subheader("Checklist")

cF1, cF2 = st.columns([2, 1])
with cF1:
    q = st.text_input("Buscar empresa", value="", placeholder="Digite para filtrar...")
with cF2:
    status_filter = st.selectbox(
        "Status",
        options=["(Todos)", "Escalar (queda)", "Investigar", "Gerenciar (aumento)", "Normal"],
        index=0,
    )

df_show = df_checklist.copy()
if q.strip():
    Q = q.strip().upper()
    df_show = df_show[df_show["company_name"].astype(str).str.upper().str.contains(Q, na=False)]
if status_filter != "(Todos)":
    df_show = df_show[df_show["status"].astype(str) == status_filter]

# colunas para exibição (inteiros já formatados)
def build_checklist_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    out["Status"] = df["status"]
    out["Empresa"] = df["company_name"]
    out["Hoje"] = df["today_total"].apply(fmt_int_pt)
    out["Média 7d"] = df["avg_7d"].apply(fmt_int_pt)
    out["Média 15d"] = df["avg_15d"].apply(fmt_int_pt)
    out["Média 30d"] = df["avg_30d"].apply(fmt_int_pt)
    out["Var 30d"] = df["var_30d"].apply(fmt_pct1)
    out["Contas (ativas)"] = df["accounts_list"].fillna("")
    out["Motivo"] = df["obs"].fillna("")

    # coluna opcional de bloqueio cautelar (por empresa)
    if "caution_has_block" in df.columns:
        out["BaaS"] = df.apply(
            lambda r: ("🔒" if bool(r.get("caution_has_block", False)) else "—"),
            axis=1
        )
        # Se quiser também mostrar valor bloqueado total por empresa:
        out["Bloqueado (R$)"] = df.get("caution_blocked_sum", 0.0).apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    return out

st.dataframe(build_checklist_display(df_show), use_container_width=True, hide_index=True)

st.divider()

# =========================
# Checklist BaaS (por conta) + Paginação (10)
# =========================
st.subheader("Checklist Bloqueio Cautelar (BaaS) — por conta")

if df_baas_table is None or df_baas_table.empty:
    st.caption("Sem CSV BaaS ou nenhuma conta do BaaS casou com as contas ativas do checklist.")
else:
    # tabela base (somente contas filtradas do checklist)
    t = df_baas_table.copy()

    # formata
    t["Posição"] = pd.to_numeric(t["pos"], errors="coerce").fillna(0).astype(int)
    t["Conta"] = t["account_id"].astype(str)
    t["Agência"] = t["agency"].astype(str)
    t["Empresa"] = t["company_name"].astype(str)
    t["Saldo Bloqueado"] = t["blocked"].astype(float)
    t["Status"] = t["has_block"].apply(lambda x: "🔒 BLOQUEIO CAUTELAR" if bool(x) else "SEM BLOQUEIO")

    t = t.sort_values(["Saldo Bloqueado", "Posição"], ascending=[False, True]).reset_index(drop=True)

    page_size = 10
    total = len(t)
    total_pages = max(1, (total + page_size - 1) // page_size)

    page = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size

    view = t.iloc[start:end].copy()
    view["Saldo Bloqueado"] = view["Saldo Bloqueado"].apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.caption(f"Mostrando {start+1}-{min(end, total)} de {total} contas (página {page}/{total_pages})")

    st.dataframe(
        view[["Posição", "Conta", "Agência", "Empresa", "Saldo Bloqueado", "Status"]],
        use_container_width=True,
        hide_index=True
    )

st.divider()

# =========================
# Cards (alertas)
# =========================
st.subheader("Cards de Alertas")

if df_alerts is None or df_alerts.empty:
    st.caption("Nenhum alerta (status != Normal) para o filtro atual.")
else:
    # filtro card
    card_filter = st.selectbox(
        "Filtrar cards",
        options=["(Todos)", "Escalar (queda)", "Investigar", "Gerenciar (aumento)"],
        index=0
    )
    dfc = df_alerts.copy()
    if card_filter != "(Todos)":
        dfc = dfc[dfc["status"].astype(str) == card_filter]

    # monta cards simples
    for _, r in dfc.iterrows():
        status = str(r.get("status", ""))
        empresa = str(r.get("company_name", ""))
        obs = str(r.get("obs", ""))
        day = str(r.get("day_ref", ""))

        caution_line = ""
        if "caution_has_block" in df_checklist.columns:
            # pega a linha da empresa no checklist enriquecido
            ck = df_checklist[df_checklist["company_key"] == r["company_key"]]
            if not ck.empty:
                has = bool(ck.iloc[0].get("caution_has_block", False))
                blocked_sum = float(ck.iloc[0].get("caution_blocked_sum", 0.0))
                if has:
                    caution_line = f"\nBloqueio cautelar (BaaS): SIM 🔒 | Total bloqueado: R$ {blocked_sum:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                else:
                    caution_line = "\nBloqueio cautelar (BaaS): NÃO"

        msg = (
            f"ALERTA: {status}\n"
            f"Empresa: {empresa}\n"
            f"Data: {day}\n"
            f"Motivo: {obs}\n"
            f"Total(D): {round_int_nearest(r.get('today_total', 0))}\n"
            f"Médias: 7d={round_int_nearest(r.get('avg_7d', 0))} | 15d={round_int_nearest(r.get('avg_15d', 0))} | 30d={round_int_nearest(r.get('avg_30d', 0))}\n"
            f"Variação: vs30={fmt_pct1(r.get('var_30d', 0))} | vs15={fmt_pct1(r.get('var_15d', 0))} | vs7={fmt_pct1(r.get('var_7d', 0))}\n"
            f"Contas (ativas): {r.get('accounts_list', '')}"
            f"{caution_line}"
        )

        with st.expander(f"{status} — {empresa}", expanded=False):
            st.code(msg, language="text")
            st.caption("Copie e cole no Discord.")