# services.py
from __future__ import annotations

import io
import re
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from core import (
    Thresholds,
    AlertConfig,
    normalize_company,
    split_companies_input,
    parse_count,
    parse_money_pt,
    compute_week_ranges,
    calc_status_volume,
    severity_rank,
    compute_streak_tail,
    farol_by_streak,
    farol_by_money,
    farol_by_freq,
    fmt_money_pt,
)


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


def parse_transactions_csv(uploaded_file) -> pd.DataFrame:
    """
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
    Saída: account_id (str), saldo_total (float), saldo_bloqueado_total (float)
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

    if not conta_col and df.shape[1] >= 2:
        conta_col = df.columns[1]
    if not saldo_col and df.shape[1] >= 6:
        saldo_col = df.columns[min(4, df.shape[1] - 1)]
    if not bloq_col and df.shape[1] >= 7:
        bloq_col = df.columns[min(5, df.shape[1] - 1)]

    if not conta_col:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["account_id"] = df[conta_col].fillna("").astype(str).str.strip().str.replace(r"\D+", "", regex=True)
    out["saldo_total"] = df[saldo_col].apply(parse_money_pt) if saldo_col in df.columns else 0.0
    out["saldo_bloqueado_total"] = df[bloq_col].apply(parse_money_pt) if bloq_col in df.columns else 0.0

    out = out[out["account_id"].str.len() > 0].copy()
    out["saldo_total"] = pd.to_numeric(out["saldo_total"], errors="coerce").fillna(0.0).astype(float)
    out["saldo_bloqueado_total"] = pd.to_numeric(out["saldo_bloqueado_total"], errors="coerce").fillna(0.0).astype(float)

    out = out.groupby("account_id", as_index=False).agg(
        saldo_total=("saldo_total", "sum"),
        saldo_bloqueado_total=("saldo_bloqueado_total", "sum"),
    )
    return out


# =========================
# Builders (datasets)
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
            low_volume_threshold=int(low_volume_threshold),
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


def enrich_baas(df_checklist: pd.DataFrame, df_baas: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    from core import LOCK_ICON, UNLOCK_ICON

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


def build_daily_matrix(
    facts: pd.DataFrame,
    companies_keys: List[str],
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
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


def build_alerts(
    daily_table: pd.DataFrame,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    df_final: pd.DataFrame,
    cfg: AlertConfig,
) -> pd.DataFrame:
    if daily_table is None or daily_table.empty or df_final is None or df_final.empty:
        return pd.DataFrame()

    day_cols = [c for c in daily_table.columns if isinstance(c, pd.Timestamp)]
    if not day_cols:
        return pd.DataFrame()

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

        days_with_tx = int((series > 0).sum())
        freq_pct = (days_with_tx / max(1, len(series))) * 100.0

        streak_zero = compute_streak_tail(series, lambda v: int(v) == 0)

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

        srow = saldos[saldos["account_id"] == re.sub(r"\D+", "", conta)]
        saldo_total = float(srow["saldo_total"].iloc[0]) if not srow.empty else 0.0
        saldo_bloq = float(srow["saldo_bloqueado_total"].iloc[0]) if not srow.empty else 0.0
        has_block = bool(srow["has_block"].iloc[0]) if not srow.empty else False

        tone_zero = farol_by_streak(streak_zero, cfg.zero_yellow, cfg.zero_orange, cfg.zero_red)
        tone_down = farol_by_streak(streak_down, cfg.down_yellow, cfg.down_orange, cfg.down_red)
        tone_block = farol_by_money(saldo_bloq, cfg.block_yellow, cfg.block_orange, cfg.block_red)
        tone_freq = farol_by_freq(freq_pct, cfg.freq_yellow, cfg.freq_orange, cfg.freq_red)

        tone_score = {"red": 4, "orange": 3, "yellow": 2, "gray": 1, "green": 0}
        score = (
            tone_score.get(tone_zero, 0) * 1000
            + tone_score.get(tone_down, 0) * 500
            + tone_score.get(tone_block, 0) * 200
            + tone_score.get(tone_freq, 0) * 100
        )

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


# =========================
# Gráficos (helpers)
# =========================
def daily_totals_from_facts(
    facts: pd.DataFrame,
    companies_keys: List[str],
    *,
    company_name_filter: Optional[str] = None,
) -> pd.DataFrame:
    if facts is None or facts.empty:
        return pd.DataFrame(columns=["date_dt", "total"])

    base = facts.copy()
    if companies_keys:
        base = base[base["company_key"].isin(set(companies_keys))].copy()
    base["date_dt"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    base = base.dropna(subset=["date_dt"]).copy()

    if company_name_filter and company_name_filter != "TOTAL GERAL":
        base = base[base["company_name"].astype(str) == str(company_name_filter)].copy()

    if base.empty:
        return pd.DataFrame(columns=["date_dt", "total"])

    g = base.groupby("date_dt", as_index=False)["total_cnt"].sum().rename(columns={"total_cnt": "total"})
    return g.sort_values("date_dt").reset_index(drop=True)


def top_companies_from_facts(
    facts: pd.DataFrame,
    companies_keys: List[str],
    *,
    topn: int = 15,
) -> pd.DataFrame:
    if facts is None or facts.empty:
        return pd.DataFrame(columns=["company_name", "total"])

    base = facts.copy()
    if companies_keys:
        base = base[base["company_key"].isin(set(companies_keys))].copy()
    if base.empty:
        return pd.DataFrame(columns=["company_name", "total"])

    g = base.groupby("company_name", as_index=False)["total_cnt"].sum().rename(columns={"total_cnt": "total"})
    return g.sort_values("total", ascending=False).head(int(topn)).reset_index(drop=True)


def zero_and_down_counts_by_day(
    daily_table: pd.DataFrame,
    cfg: AlertConfig,
) -> pd.DataFrame:
    if daily_table is None or daily_table.empty:
        return pd.DataFrame(columns=["date_dt", "zero_accounts", "down_accounts"])

    day_cols = [c for c in daily_table.columns if isinstance(c, pd.Timestamp)]
    if not day_cols:
        return pd.DataFrame(columns=["date_dt", "zero_accounts", "down_accounts"])

    X = daily_table[day_cols].to_numpy(dtype=float)  # (n_accounts, n_days)
    _, n_days = X.shape

    zero_accounts = (X <= 0).sum(axis=0).astype(int)
    down_accounts = np.zeros(n_days, dtype=int)

    k = int(max(2, cfg.baseline_days))
    ratio = float(cfg.drop_ratio)

    for j in range(n_days):
        if j <= 0:
            down_accounts[j] = 0
            continue
        start = max(0, j - k)
        prev = X[:, start:j]
        baseline = np.mean(prev, axis=1)
        thresh = baseline * ratio
        down = (baseline > 0) & (X[:, j] < thresh)
        down_accounts[j] = int(down.sum())

    out = pd.DataFrame({
        "date_dt": pd.to_datetime(day_cols),
        "zero_accounts": zero_accounts,
        "down_accounts": down_accounts,
    }).sort_values("date_dt").reset_index(drop=True)

    return out
