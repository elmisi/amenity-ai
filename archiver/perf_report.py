from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


def _is_num(x: object) -> bool:
    if not isinstance(x, (int, float)):
        return False
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return False
    return True


def _percent(n: int, d: int) -> str:
    if d <= 0:
        return "0%"
    return f"{(100.0 * n / d):.0f}%"


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    idx = max(0, min(len(values) - 1, idx))
    return values[idx]


def _summarize_seconds(values: Iterable[object]) -> tuple[int, float | None, float | None, float | None]:
    vals = [float(v) for v in values if _is_num(v)]
    if not vals:
        return 0, None, None, None
    mean = sum(vals) / len(vals)
    p50 = _quantile(vals, 0.50)
    p95 = _quantile(vals, 0.95)
    return len(vals), mean, p50, p95


def print_performance_report(*, source_root: Path) -> None:
    cache_path = source_root.expanduser().resolve() / ".amenity-stuff" / "cache.json"
    if not cache_path.exists():
        print(f"No cache found: {cache_path}")
        return

    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to read cache: {cache_path} ({type(exc).__name__})")
        return
    if not isinstance(raw, dict):
        print(f"Unexpected cache format: {cache_path}")
        return

    entries = [v for v in raw.values() if isinstance(v, dict)]
    total = len(entries)

    status_counts: dict[str, int] = {}
    pdf = 0
    image = 0
    ocr = 0
    text = 0
    missing_year = 0

    ocr_time: list[object] = []
    facts_llm: list[object] = []
    facts_total: list[object] = []
    classify_llm: list[object] = []

    for e in entries:
        st = str(e.get("status") or "")
        status_counts[st] = status_counts.get(st, 0) + 1

        rel = str(e.get("rel_path") or "")
        ext = rel.lower().rsplit(".", 1)[-1] if "." in rel else ""
        if ext == "pdf":
            pdf += 1
        elif ext in {"jpg", "jpeg", "png"}:
            image += 1

        method = e.get("extract_method")
        if method == "ocr":
            ocr += 1
        elif method == "text":
            text += 1

        if st == "classified":
            year = e.get("reference_year")
            if not isinstance(year, str) or not year.strip():
                missing_year += 1

        ocr_time.append(e.get("ocr_time_s"))
        facts_llm.append(e.get("facts_llm_time_s"))
        facts_total.append(e.get("facts_time_s"))
        classify_llm.append(e.get("classify_llm_time_s"))

    n_ocr, ocr_mean, ocr_p50, ocr_p95 = _summarize_seconds(ocr_time)
    n_fllm, fllm_mean, fllm_p50, fllm_p95 = _summarize_seconds(facts_llm)
    n_ftot, ftot_mean, ftot_p50, ftot_p95 = _summarize_seconds(facts_total)
    n_cllm, cllm_mean, cllm_p50, cllm_p95 = _summarize_seconds(classify_llm)

    def fmt(x: float | None) -> str:
        return "-" if x is None else f"{x:.1f}s"

    print("amenity-ai performance report")
    print(f"Source: {source_root.expanduser().resolve()}")
    print(f"Cache:  {cache_path}")
    print("")

    print(f"Files: {total} (pdf={pdf}, images={image})")
    print(f"Extract: ocr={ocr} ({_percent(ocr, total)}), text={text} ({_percent(text, total)})")
    if status_counts:
        parts = [f"{k}={v}" for k, v in sorted(status_counts.items(), key=lambda kv: (-kv[1], kv[0])) if k]
        print("Status:", ", ".join(parts) if parts else "-")
    print("")

    print("Timings (mean / p50 / p95):")
    print(f"- OCR:          {fmt(ocr_mean)} / {fmt(ocr_p50)} / {fmt(ocr_p95)}  (n={n_ocr})")
    print(f"- Facts LLM:     {fmt(fllm_mean)} / {fmt(fllm_p50)} / {fmt(fllm_p95)}  (n={n_fllm})")
    print(f"- Facts total:   {fmt(ftot_mean)} / {fmt(ftot_p50)} / {fmt(ftot_p95)}  (n={n_ftot})")
    print(f"- Classify LLM:  {fmt(cllm_mean)} / {fmt(cllm_p50)} / {fmt(cllm_p95)}  (n={n_cllm})")
    print("")

    notes: list[str] = []
    if fllm_mean is not None and cllm_mean is not None and fllm_mean > cllm_mean:
        notes.append("Bottleneck: Facts LLM dominates runtime (consider a smaller model for Scan/Facts).")
    if missing_year:
        notes.append(f"{missing_year} file(s) are classified but have no reference year.")
    if not notes:
        notes.append("No obvious issues detected in the cache summary.")

    print("Notes:")
    for n in notes:
        print(f"- {n}")

