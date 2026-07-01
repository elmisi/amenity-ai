from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class TaxonomyCategory:
    name: str
    description: str = ""
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class Taxonomy:
    categories: tuple[TaxonomyCategory, ...]

    @property
    def allowed_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.categories)


DEFAULT_TAXONOMY_EN: tuple[str, ...] = (
    "house | Home, property, rent, utilities, household paperwork | rent; lease; condominium; property tax; utility bill; electricity; gas; water; internet; home insurance; maintenance",
    "purchases | Purchases and subscriptions | receipt; order confirmation; subscription; e-commerce; warranty; invoice for goods/services",
    "travel | Travel and transportation | flight; hotel; booking; ticket; itinerary; car rental; travel insurance",
    "tax | Taxes and public administration | F24; tax return; agency letter; payment notice; PagoPA; municipality tax",
    "banking | Banking and payments (generic) | bank statement; transfer; card statement; account; payment receipt",
    "legal | Legal documents and compliance | contract; terms; privacy policy; legal letter; complaint; power of attorney",
    "work | Employment and professional documents | payslip; payroll; timesheet; employment agreement; HR",
    "personal | Personal documents, IDs, letters, handwritten notes | identity card; passport; driving licence; certificate; personal letter; handwritten note; notes; song lyrics; poem",
    "medical | Health and medical records | medical report; prescription; lab results; vaccination; invoice medical",
    "edu | Education and training | certificate; transcript; diploma; course material; thesis; enrollment",
    "media | Media and content | ebook; article; photo; screenshot; scan of photo; audio; video",
    "tech | Technical docs | manual; datasheet; spec; API documentation; architecture; configuration; logs",
    "unknown | Unclassified / skipped |",
)

DEFAULT_TAXONOMY_IT: tuple[str, ...] = (
    "casa | Abitazione e immobili: affitto, condominio, utenze, manutenzione, tasse casa | affitto; contratto locazione; condominio; amministratore; manutenzione; assicurazione casa; IMU; TARI; bolletta; luce; gas; acqua; internet",
    "acquisti | Acquisti e abbonamenti: ordini, ricevute, garanzie, servizi | ricevuta; scontrino; ordine; conferma ordine; abbonamento; rinnovo; garanzia; fattura acquisto; e-commerce",
    "viaggi | Viaggi e trasporti: prenotazioni, biglietti, itinerari | volo; biglietto; hotel; prenotazione; itinerario; noleggio auto; assicurazione viaggio; treno; trasporto",
    "tasse | Tasse e pubblica amministrazione: pagamenti e comunicazioni ufficiali | F24; dichiarazione redditi; Agenzia Entrate; tributo; avviso pagamento; PagoPA; Comune; cartella; imposta",
    "banca | Banca e pagamenti generici: estratti conto e movimenti | estratto conto; bonifico; carta; transazione; addebito; accredito; ricevuta pagamento; conto corrente",
    "legale | Documenti legali e compliance | contratto; termini; privacy; diffida; procura; atto; denuncia; ricorso; NDA; lettera legale",
    "lavoro | Documenti di lavoro e professionali | busta paga; cedolino; payroll; timesheet; contratto lavoro; offerta; HR; CU; CUD; lettera assunzione; dimissioni",
    "personale | Documenti personali, identità, lettere, appunti | carta identità; passaporto; patente; certificato; lettera personale; appunti; nota; testo canzone; poesia; scritto a mano",
    "salute | Documenti sanitari e medici | referto; ricetta; analisi; visita; certificato medico; vaccino; fattura medica; terapia; dieta",
    "studio | Scuola, università e formazione | attestato; certificato; diploma; transcript; materiale corso; iscrizione; esame; tesi",
    "media | Contenuti e media: libri, foto, screenshot, audio/video | ebook; libro; articolo; foto; screenshot; scansione foto; audio; video",
    "tecnica | Documenti tecnici: manuali, specifiche, documentazione | manuale; specifica; datasheet; documentazione API; architettura; configurazione; log; guida",
    "sconosciuto | Non classificato / saltato |",
)

DEFAULT_TAXONOMIES: dict[str, tuple[str, ...]] = {
    "en": DEFAULT_TAXONOMY_EN,
    "it": DEFAULT_TAXONOMY_IT,
}

# Backward compatibility alias
DEFAULT_TAXONOMY_LINES: tuple[str, ...] = DEFAULT_TAXONOMY_EN


def _get_user_config_dir() -> Path:
    """Get the user config directory for amenity-ai."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "amenity-stuff"


def _get_package_taxonomies_dir() -> Path:
    """Get the package taxonomies directory."""
    return Path(__file__).parent / "taxonomies"


def _load_taxonomy_from_file(path: Path) -> tuple[str, ...] | None:
    """Load taxonomy lines from a file. Returns None if file not found."""
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        lines = tuple(ln.rstrip("\n") for ln in text.splitlines() if ln.strip())
        return lines if lines else None
    except Exception:
        return None


def load_taxonomy_for_language(lang: str) -> tuple[str, ...]:
    """Load taxonomy for a language, checking user config first, then package files.

    Search order:
    1. ~/.config/amenity-stuff/taxonomies/{lang}.txt (user override)
    2. {package}/taxonomies/{lang}.txt (bundled)
    3. Hardcoded default (backward compatibility)
    """
    # 1. Try user config directory
    user_file = _get_user_config_dir() / "taxonomies" / f"{lang}.txt"
    lines = _load_taxonomy_from_file(user_file)
    if lines:
        return lines

    # 2. Try package taxonomies directory
    package_file = _get_package_taxonomies_dir() / f"{lang}.txt"
    lines = _load_taxonomy_from_file(package_file)
    if lines:
        return lines

    # 3. Fall back to hardcoded default
    return DEFAULT_TAXONOMIES.get(lang, DEFAULT_TAXONOMY_EN)


def get_default_taxonomy_for_language(lang: str) -> tuple[str, ...]:
    """Return the default taxonomy for a given language code.

    Uses load_taxonomy_for_language which checks:
    1. User config (~/.config/amenity-stuff/taxonomies/{lang}.txt)
    2. Package files (archiver/taxonomies/{lang}.txt)
    3. Hardcoded defaults
    """
    return load_taxonomy_for_language(lang)


def get_effective_language(output_language: str) -> str:
    """Resolve 'auto' to a concrete language code based on system locale."""
    if output_language in {"it", "en"}:
        return output_language
    # Detect from environment
    import locale
    try:
        loc = locale.getlocale()[0] or ""
        if loc.lower().startswith("it"):
            return "it"
    except Exception:
        pass
    return "en"


_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


def parse_taxonomy_lines(lines: Iterable[str]) -> tuple[Taxonomy, tuple[str, ...]]:
    """Parse user-editable taxonomy lines.

    Grammar (one per line):
      name | description | example1; example2; ...

    Returns: (taxonomy, errors)
    """

    errors: list[str] = []
    categories: list[TaxonomyCategory] = []

    for idx, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split("|")]
        name = parts[0].strip().lower() if parts else ""
        if not name:
            errors.append(f"Line {idx}: missing category name")
            continue
        if not _NAME_RE.fullmatch(name):
            errors.append(
                f"Line {idx}: invalid category name '{name}' (use: a-z, 0-9, '_' or '-', 2-64 chars)"
            )
            continue
        if name == "analysis" or name == "pending":
            errors.append(f"Line {idx}: reserved category name '{name}'")
            continue

        description = parts[1] if len(parts) >= 2 else ""
        examples_raw = parts[2] if len(parts) >= 3 else ""
        examples: tuple[str, ...] = ()
        if examples_raw.strip():
            ex = [e.strip() for e in examples_raw.split(";") if e.strip()]
            examples = tuple(ex[:12])

        categories.append(TaxonomyCategory(name=name, description=description, examples=examples))

    names = [c.name for c in categories]
    if "unknown" not in names:
        categories.append(TaxonomyCategory(name="unknown", description="Unclassified / skipped", examples=()))

    # De-duplicate keeping first occurrence.
    seen: set[str] = set()
    deduped: list[TaxonomyCategory] = []
    for c in categories:
        if c.name in seen:
            continue
        seen.add(c.name)
        deduped.append(c)

    if not deduped:
        errors.append("No valid categories found")
        deduped = [TaxonomyCategory(name="unknown", description="Unclassified / skipped", examples=())]

    return Taxonomy(categories=tuple(deduped)), tuple(errors)


def taxonomy_to_prompt_block(taxonomy: Taxonomy) -> str:
    lines: list[str] = []
    for c in taxonomy.categories:
        ex = ""
        if c.examples:
            ex = " Examples: " + "; ".join(c.examples)
        desc = (c.description or "").strip()
        if desc:
            lines.append(f"- {c.name}: {desc}{ex}".strip())
        else:
            lines.append(f"- {c.name}:{ex}".strip())
    return "\n".join(lines)
