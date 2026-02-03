from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    queda_critica: float = -0.60
    aumento_relevante: float = 0.80
    investigar_abs: float = 0.30


def process_transactions_csv(
    file_bytes: bytes,
    companies_param: str,
    thresholds: Thresholds
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    TODO: aqui você vai colar sua lógica do service (alerts_service.py)
    e adaptar para rodar local (sem Flask).

    Retorno:
      day_ref_iso, checklist_rows, cards_rows
    """
    # placeholder seguro
    return "", [], []
