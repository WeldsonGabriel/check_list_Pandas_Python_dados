# core.py
from __future__ import annotations

import re
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
import pandas as pd

try:
    from unidecode import unidecode
except Exception:
    unidecode = None


# =========================
# Constantes / domínio
# =========================
STATUS_ORDER = ["Alerta (queda/ zerada)", "Investigar", "Gerenciar (aumento)", "Normal"]

STATUS_COLOR = {
    "Alerta (queda/ zerada)": "#e74c3c",
    "Investigar": "#f1c40f",
    "Gerenciar (aumento)": "#3498db",
    "Normal": "#2ecc71",
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
# Dataclasses (config/regras)
# =========================
@dataclass(frozen=True)
class Thresholds:
    queda_critica: float = -0.60
    aumento_relevante: float = 0.80
    investigar_abs: float = 0.30


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

    # frequência (% dias com tx) -> menor = pior
    freq_yellow: float = 80.0
    freq_orange: float = 60.0
    freq_red: float = 40.0

    # baseline p/ detectar queda
    baseline_days: int = 7
    drop_ratio: float = 0.70


# =========================
# Normalização / parsing leve (domínio)
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


def parse_count(x) -> int:
    """BUGFIX: suporta '448.326' / '448,326' etc."""
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


def parse_money_pt(x) -> float:
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
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0


# =========================
# Formatação (domínio)
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
# Semanas (28d terminando no último dia do CSV)
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
# Status semanal
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
# Faróis (alertas diários)
# =========================
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
    # menor = pior
    if pct <= r:
        return "red"
    if pct <= o:
        return "orange"
    if pct <= y:
        return "yellow"
    return "green"


def compute_streak_tail(values: np.ndarray, predicate: Callable[[float], bool]) -> int:
    n = 0
    for v in values[::-1]:
        if predicate(v):
            n += 1
        else:
            break
    return n


def assistant_explain_row(row: pd.Series, cfg: AlertConfig) -> str:
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
        lines.append(
            f"- Baseline ({cfg.baseline_days}d): **{fmt_int_pt(int(round(baseline)))}** "
            f"| Limiar de queda ({int(cfg.drop_ratio*100)}%): **{fmt_int_pt(int(round(thresh)))}**"
        )
    if z > 0:
        lines.append(f"- 🔴 Zerado no tail: **{z} dia(s)** (consecutivos no final)")
    if baseline > 0 and d > 0:
        lines.append(f"- 🟠 Queda no tail: **{d} dia(s)** abaixo do limiar (vs baseline)")
    if bloq > 0:
        lines.append(f"- 🔒 Saldo bloqueado: **R$ {fmt_money_pt(bloq)}** | Saldo total: **R$ {fmt_money_pt(saldo)}**")
    else:
        lines.append(f"- 🔓 Sem bloqueio | Saldo total: **R$ {fmt_money_pt(saldo)}**")

    actions = []
    if z >= cfg.zero_yellow:
        actions.append("Validar interrupção (webhook/instabilidade/limite) e confirmar operação do cliente no período.")
    if baseline > 0 and d >= cfg.down_yellow and z == 0:
        actions.append("Checar causa de queda: sazonalidade, indisponibilidade, mudança de comportamento ou regra antifraude.")
    if bloq >= cfg.block_yellow:
        actions.append("Checar motivo do bloqueio e impacto operacional (valor elevado).")
    if freq <= cfg.freq_orange:
        actions.append("Checar se opera só em dias úteis/fds e ajustar leitura por calendário.")

    if actions:
        lines.append("")
        lines.append("**Sugestões objetivas (auto):**")
        for a in actions[:6]:
            lines.append(f"- {a}")

    return "\n".join(lines)
