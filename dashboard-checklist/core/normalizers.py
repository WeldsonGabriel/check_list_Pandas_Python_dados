import re
import unicodedata

_SUFFIXES = {
    " LTDA", " LTDA.", " SA", " S.A", " S.A.", " ME", " EPP", " EI",
    " EIRELI", " S/S", " S S", " S A"
}

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()

    # remove acentos
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # remove pontuação
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # remove sufixos comuns
    for suf in sorted(_SUFFIXES, key=len, reverse=True):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()

    return s
