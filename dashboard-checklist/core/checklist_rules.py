# core/checklist_rules.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Dict, List, Tuple
import io
import math
import re

import pandas as pd

from .normalizers import normalize_text


@dataclass(frozen=True)
class Thresholds:
    queda_critica: float = -0.60
    aumento_relevante: float = 0.80
    investigar_abs: float = 0.30


# ----------------------------
# Normalização de empresa
# ----------------------------
def normalize_company(name: str) -> str:
    return normalize_text(name)


def split_companies_param(companies: Optional[str]) -> List[str]:
    if not companies:
        return []
    parts = [p.strip() for p in str(companies).split(",")]
    return [p for p in parts if p]


# ----------------------------
# Rounding policy (seu padrão)
# 1.1 -> 1 ; 1.8 -> 2 ; 0.5 -> 1
# ----------------------------
def _round_int_nearest(x: float) -> int:
    try:
        v = float(x)
    except Exception:
        return 0
    if v >= 0:
        return int(math.floor(v + 0.5))
    return int(math.ceil(v - 0.5))


def _pct_int_ceil_ratio(var_ratio: float) -> int:
    """
    Converte razão em % inteiro.
    - Positivo: ceil (80.1 -> 81)
    - Negativo: ceil aproxima para zero (-60.1 -> -60)
    """
    try:
        pct = float(var_ratio) * 100.0
    except Exception:
        return 0
    return int(math.ceil(pct))


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return float(numer) / float(denom)


def calc_var(today_total: float, avg: float) -> float:
    if avg == 0:
        return 0.0
    return (float(today_total) - float(avg)) / float(avg)


def calc_status(var30: float, *, queda_critica: float, aumento_relevante: float, investigar_abs: float) -> str:
    # Mantém sua semântica
    if var30 <= queda_critica:
        return "Alerta (queda/ zerada)"
    if var30 >= aumento_relevante:
        return "Gerenciar (aumento)"
    if abs(var30) >= investigar_abs:
        return "Investigar"
    return "Normal"


def calc_obs(status: str) -> str:
    if status == "Alerta (quedaqueda/ zerada)":
        return "Queda crítica vs média histórica"
    if status == "Gerenciar (aumento)":
        return "Aumento relevante vs média histórica"
    if status == "Investigar":
        return "Variação relevante (abs) — checar contexto"
    return "Sem anomalia"


def severity_rank(status: str) -> int:
    # menor = mais crítico
    order = {
        "Alerta (quedaqueda/ zerada)": 0,
        "Investigar": 1,
        "Gerenciar (aumento)": 2,
        "Normal": 3,
    }
    return order.get(status, 9)


def _accounts_lists_for_company(
    base_acc: pd.DataFrame,
    company_key: str,
    start: date,
    end: date,
) -> Tuple[List[str], List[str]]:
    """
    active = soma_total > 0 no período
    zero   = soma_total <= 0 no período (inclui 0 e negativos)
    """
    mask = (
        (base_acc["company_key"] == company_key) &
        (base_acc["date"] >= start) &
        (base_acc["date"] <= end)
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

    # garante exclusão mútua
    aset = set(active)
    zero = [a for a in zero if a not in aset]

    return active, zero


# ----------------------------
# Leitura do CSV de transações
# ----------------------------
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
    # fallback: tenta split no primeiro "-"
    if "-" in s:
        a, b = s.split("-", 1)
        return a.strip(), b.strip()
    # sem separador -> retorna como empresa
    return "", s.strip()


def _to_float(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    # suporta "1.234,56" ou "1234,56" ou "1234.56"
    s = s.replace(".", "").replace(",", ".") if re.search(r"\d+,\d+", s) else s
    try:
        return float(s)
    except Exception:
        return 0.0


def _read_transactions_file(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes:
        return pd.DataFrame()

    text = file_bytes.decode("utf-8", errors="replace")
    sep = _guess_sep(text[:4000])

    # Lê sem depender de header perfeito
    df_raw = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",
        dtype=str,
        header=0
    )

    # Se vier sem header ou com header estranho, tentamos pelo índice
    # Esperado:
    # col0=date, col2=person, col3=credit, col4=debit
    cols = list(df_raw.columns)
    if len(cols) < 5:
        # tenta ler sem header
        df_raw = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            engine="python",
            dtype=str,
            header=None
        )
        if df_raw.shape[1] < 5:
            return pd.DataFrame()

        date_col, person_col, credit_col, debit_col = 0, 2, 3, 4
        df = pd.DataFrame()
        df["date_raw"] = df_raw.iloc[:, date_col].astype(str)
        df["person_raw"] = df_raw.iloc[:, person_col].astype(str)
        df["credit_raw"] = df_raw.iloc[:, credit_col]
        df["debit_raw"] = df_raw.iloc[:, debit_col]
    else:
        # usa por índice para ser compatível com o que você descreveu
        df = pd.DataFrame()
        df["date_raw"] = df_raw.iloc[:, 0].astype(str)
        df["person_raw"] = df_raw.iloc[:, 2].astype(str)
        df["credit_raw"] = df_raw.iloc[:, 3]
        df["debit_raw"] = df_raw.iloc[:, 4]

    # parse
    dates = pd.to_datetime(df["date_raw"], errors="coerce", dayfirst=True)
    df["date"] = dates.dt.date

    acc_emp = df["person_raw"].apply(_parse_person)
    df["account_id"] = acc_emp.apply(lambda t: t[0])
    df["company_name"] = acc_emp.apply(lambda t: t[1])

    df["credit"] = df["credit_raw"].apply(_to_float)
    df["debit"] = df["debit_raw"].apply(_to_float)
    df["total"] = df["credit"] + df["debit"]

    df["company_key"] = df["company_name"].apply(normalize_company)

    # remove linhas inválidas
    df = df.dropna(subset=["date"])
    df = df[df["company_key"].astype(str).str.len() > 0]

    # garante account_id como string (pode ser vazio em casos ruins)
    df["account_id"] = df["account_id"].astype(str).str.strip()

    return df[["date", "account_id", "company_name", "company_key", "total"]].copy()


# ----------------------------
# PIPE PRINCIPAL
# ----------------------------
def process_transactions_csv(
    file_bytes: bytes,
    companies_param: str,
    thresholds: Thresholds
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Retorna:
      day_ref_iso, checklist_rows, cards_rows
    """
    facts = _read_transactions_file(file_bytes)
    if facts.empty:
        return "", [], []

    # filtro por empresas (company_key normalizado)
    companies_list = split_companies_param(companies_param)
    if companies_list:
        wanted = {normalize_company(x) for x in companies_list}
        facts = facts[facts["company_key"].isin(wanted)].copy()

    if facts.empty:
        return "", [], []

    # day_ref = maior data do CSV (último dia real)
    day_ref: date = max(facts["date"])
    d1 = day_ref - timedelta(days=1)

    # base diária por conta
    base_acc = (
        facts.groupby(["account_id", "company_key", "company_name", "date"], as_index=False)["total"]
             .sum()
    )

    # base diária por empresa
    base_comp = (
        base_acc.groupby(["company_key", "company_name", "date"], as_index=False)["total"]
                .sum()
    )

    active_start = day_ref - timedelta(days=30)
    active_end = d1

    def sum_window_company(company_key: str, days: int) -> float:
        start = day_ref - timedelta(days=days)
        mask = (
            (base_comp["company_key"] == company_key) &
            (base_comp["date"] >= start) &
            (base_comp["date"] <= d1)
        )
        return float(base_comp.loc[mask, "total"].sum())

    def today_total_company(company_key: str) -> float:
        mask = (base_comp["company_key"] == company_key) & (base_comp["date"] == day_ref)
        return float(base_comp.loc[mask, "total"].sum())

    checklist_rows: List[Dict] = []

    for company_key in sorted(base_comp["company_key"].unique()):
        names = base_comp.loc[base_comp["company_key"] == company_key, "company_name"].dropna()
        company_name = str(names.iloc[-1]) if not names.empty else company_key

        active_accounts, zero_accounts = _accounts_lists_for_company(
            base_acc=base_acc,
            company_key=company_key,
            start=active_start,
            end=active_end,
        )
        if not active_accounts:
            continue

        accounts_list = ", ".join(active_accounts)
        accounts_zero_list = ", ".join(zero_accounts)
        accounts_zero_count = len(zero_accounts)

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

        status = calc_status(
            var30,
            queda_critica=thresholds.queda_critica,
            aumento_relevante=thresholds.aumento_relevante,
            investigar_abs=thresholds.investigar_abs,
        )
        obs = calc_obs(status)

        # inteiros (seu padrão)
        today_total_i = _round_int_nearest(today_total)
        avg7_i = _round_int_nearest(avg7)
        avg15_i = _round_int_nearest(avg15)
        avg30_i = _round_int_nearest(avg30)

        # % inteiro (ceil)
        var7_pct = _pct_int_ceil_ratio(var7)
        var15_pct = _pct_int_ceil_ratio(var15)
        var30_pct = _pct_int_ceil_ratio(var30)

        checklist_rows.append({
            "company_key": company_key,
            "company_name": company_name,

            "accounts_list": accounts_list,
            "accounts_zero_list": accounts_zero_list,
            "accounts_zero_count": accounts_zero_count,

            "today_total": float(today_total),
            "avg_7d": float(avg7),
            "avg_15d": float(avg15),
            "avg_30d": float(avg30),

            "today_total_i": today_total_i,
            "avg_7d_i": avg7_i,
            "avg_15d_i": avg15_i,
            "avg_30d_i": avg30_i,

            "var_7d": float(var7),
            "var_15d": float(var15),
            "var_30d": float(var30),

            "var_7d_pct": var7_pct,
            "var_15d_pct": var15_pct,
            "var_30d_pct": var30_pct,

            "status": status,
            "obs": obs,
            "day_ref": day_ref.isoformat(),
        })

    checklist_rows.sort(key=lambda r: (severity_rank(r["status"]), r["company_name"]))
    alerts_rows = [r for r in checklist_rows if r["status"] != "Normal"]

    # cards (mantém compatível com seu front)
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
            "mensagem_discord": msg,
        })

    return day_ref.isoformat(), checklist_rows, cards_rows
