from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional, Dict, List

import pandas as pd

from .normalizers import normalize_text


@dataclass
class BaasRow:
    posicao: Optional[int]
    conta: str
    agencia: str
    nome: str
    plano: str
    saldo: float
    saldo_bloqueado: float
    baas: str

    @property
    def blocked(self) -> bool:
        return float(self.saldo_bloqueado or 0) > 0


def _guess_sep(sample: str) -> str:
    # heurística simples: se tiver tab, é TSV
    if "\t" in sample:
        return "\t"
    # tenta ; se tiver bastante ;
    if sample.count(";") >= sample.count(","):
        return ";"
    return ","


def read_baas_file(file_bytes: bytes) -> pd.DataFrame:
    """
    Lê CSV BaaS e devolve DataFrame normalizado com colunas padrão:
      posicao, conta, agencia, nome, plano, saldo, saldo_bloqueado, baas, nome_norm
    """
    if not file_bytes:
        return pd.DataFrame()

    text = file_bytes.decode("utf-8", errors="replace")
    sep = _guess_sep(text[:4000])

    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        dtype=str,
        engine="python",
    )

    # Normaliza headers (tolerante a variações)
    cols = {c.strip().lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_pos = pick("posição", "posicao", "pos")
    c_conta = pick("conta", "account")
    c_ag = pick("agência", "agencia", "agency")
    c_nome = pick("nome", "empresa", "titular")
    c_plano = pick("plano")
    c_saldo = pick("saldo")
    c_bloq = pick("saldo bloqueado", "saldo_bloqueado", "bloqueado", "blocked")
    c_baas = pick("baas", "bás", "instituicao")

    # Se não achar o mínimo, devolve vazio
    if not c_conta or not c_bloq:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["posicao"] = pd.to_numeric(df[c_pos], errors="coerce") if c_pos else None
    out["conta"] = df[c_conta].fillna("").astype(str).str.strip()
    out["agencia"] = df[c_ag].fillna("").astype(str).str.strip() if c_ag else ""
    out["nome"] = df[c_nome].fillna("").astype(str) if c_nome else ""
    out["plano"] = df[c_plano].fillna("").astype(str) if c_plano else ""
    out["saldo"] = pd.to_numeric(df[c_saldo], errors="coerce").fillna(0.0) if c_saldo else 0.0
    out["saldo_bloqueado"] = pd.to_numeric(df[c_bloq], errors="coerce").fillna(0.0)
    out["baas"] = df[c_baas].fillna("").astype(str) if c_baas else ""

    out["nome_norm"] = out["nome"].map(normalize_text)

    return out


def build_baas_index(df_baas: pd.DataFrame) -> Dict[str, dict]:
    """
    Index por CONTA:
      conta(str) -> {saldo_bloqueado, blocked, meta...}
    """
    idx: Dict[str, dict] = {}
    if df_baas is None or df_baas.empty:
        return idx

    for _, r in df_baas.iterrows():
        conta = str(r.get("conta", "")).strip()
        if not conta:
            continue
        saldo_bloq = float(r.get("saldo_bloqueado", 0.0) or 0.0)
        idx[conta] = {
            "posicao": None if pd.isna(r.get("posicao")) else int(r.get("posicao")),
            "conta": conta,
            "agencia": str(r.get("agencia", "") or ""),
            "nome": str(r.get("nome", "") or ""),
            "nome_norm": str(r.get("nome_norm", "") or ""),
            "plano": str(r.get("plano", "") or ""),
            "saldo_bloqueado": saldo_bloq,
            "blocked": saldo_bloq > 0,
            "baas": str(r.get("baas", "") or ""),
        }
    return idx


def extract_accounts_list(accounts_list_str: str) -> List[str]:
    """
    Recebe string tipo: "11717, 11758, 222"
    Retorna ["11717","11758","222"]
    """
    if not accounts_list_str:
        return []
    parts = re.split(r"[,\s]+", str(accounts_list_str).strip())
    return [p for p in parts if p]
