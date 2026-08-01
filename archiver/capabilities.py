"""Capability dei modelli e parsing della taglia.

Tre gradini di affidabilità decrescente, in quest'ordine:
  declared  - dichiarate dal provider (Ollama >= 0.31 in /api/tags)
  heuristic - dedotte dal nome, per i provider OpenAI-compatible
  probed    - confermate da una richiesta reale, solo dentro il doctor

Il gradino "probed" esiste perché l'euristica ha falsi negativi reali: un
modello servito via OpenAI API può accettare immagini senza dirlo nel nome.
"""
from __future__ import annotations

import re
from typing import Optional

CAP_COMPLETION = "completion"
CAP_VISION = "vision"
CAP_EMBEDDING = "embedding"

SOURCE_DECLARED = "declared"
SOURCE_HEURISTIC = "heuristic"
SOURCE_PROBED = "probed"

# Un numero conta come taglia solo se seguito da "b" a fine token: in
# "qwen3.6-27b" il 3.6 è la versione e va ignorato.
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", re.IGNORECASE)

_VISION_TOKENS = (
    "llava",
    "moondream",
    "minicpm-v",
    "bakllava",
    "pixtral",
    "internvl",
    "vision",
    "-vl-",
    "-vl:",
    "vl-",
    "qwen2vl",
    "qwen2.5vl",
)

_IMAGE_ERROR_TOKENS = (
    "image",
    "multimodal",
    "vision",
    "image_url",
)


def parse_parameter_size(*texts: str) -> Optional[float]:
    """Prima taglia trovata scorrendo gli argomenti nell'ordine dato."""
    for text in texts:
        if not text:
            continue
        match = _SIZE_RE.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def guess_capabilities(*, model_id: str, root: str = "") -> frozenset[str]:
    """Capability dedotte dal nome, per i provider che non le dichiarano."""
    haystack = f"{model_id} {root}".lower()
    caps = {CAP_COMPLETION}
    if any(token in haystack for token in _VISION_TOKENS):
        caps.add(CAP_VISION)
    return frozenset(caps)


def interpret_probe(*, status: int, body: str) -> Optional[bool]:
    """True = vision confermata, False = text-only confermato, None = non conclusivo.

    Il probe può smentire l'euristica, non può fingere una certezza che non
    ha: un 500 o un timeout non dicono nulla sulle capability del modello.
    """
    if status == 200:
        return True
    if status in (400, 415, 422):
        lowered = body.lower()
        if any(token in lowered for token in _IMAGE_ERROR_TOKENS):
            return False
    return None
