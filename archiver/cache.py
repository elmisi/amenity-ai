from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from .scanner import ScanItem


class SaveThrottle:
    """Decide when a cache write is worth its cost.

    One write re-serialises the whole cache. At one file every twenty seconds
    that is invisible; with several in flight it starts stealing frames from
    the event loop, which is where the write happens.
    """

    def __init__(
        self,
        *,
        min_interval_s: float = 5.0,
        min_dirty: int = 25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._min_dirty = min_dirty
        self._clock = clock
        self._dirty = 0
        self._last = clock()

    def record(self) -> bool:
        """Register one changed entry. True when it is time to write."""
        self._dirty += 1
        due = (
            self._dirty >= self._min_dirty
            or (self._clock() - self._last) >= self._min_interval_s
        )
        if due:
            self._reset()
        return due

    def flush(self) -> bool:
        """True when unwritten changes remain. Call at the end of a run."""
        if self._dirty == 0:
            return False
        self._reset()
        return True

    def _reset(self) -> None:
        self._dirty = 0
        self._last = self._clock()


@dataclass(frozen=True)
class CacheEntry:
    rel_path: str
    size_bytes: int
    mtime_iso: str
    status: str
    moved_to: Optional[str] = None
    reason: Optional[str] = None
    category: Optional[str] = None
    reference_year: Optional[str] = None
    proposed_name: Optional[str] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    analysis_time_s: Optional[float] = None
    model_used: Optional[str] = None
    summary_long: Optional[str] = None
    facts_json: Optional[str] = None
    llm_raw_output: Optional[str] = None
    extract_method: Optional[str] = None
    extract_time_s: Optional[float] = None
    llm_time_s: Optional[float] = None
    ocr_time_s: Optional[float] = None
    ocr_mode: Optional[str] = None
    facts_time_s: Optional[float] = None
    facts_llm_time_s: Optional[float] = None
    facts_model_used: Optional[str] = None
    classify_time_s: Optional[float] = None
    classify_llm_time_s: Optional[float] = None
    classify_model_used: Optional[str] = None


class CacheStore:
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root
        self._path = self.source_root / ".amenity-stuff" / "cache.json"
        self._data: dict[str, CacheEntry] = {}

    def load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}
            return
        if not isinstance(raw, dict):
            self._data = {}
            return
        data: dict[str, CacheEntry] = {}
        for rel_path, entry in raw.items():
            if not isinstance(rel_path, str) or not isinstance(entry, dict):
                continue
            try:
                data[rel_path] = CacheEntry(
                    rel_path=rel_path,
                    size_bytes=int(entry.get("size_bytes", 0)),
                    mtime_iso=str(entry.get("mtime_iso", "")),
                    status=str(entry.get("status", "")),
                    moved_to=entry.get("moved_to"),
                    reason=entry.get("reason"),
                    category=entry.get("category"),
                    reference_year=entry.get("reference_year"),
                    proposed_name=entry.get("proposed_name"),
                    summary=entry.get("summary"),
                    confidence=entry.get("confidence"),
                    analysis_time_s=entry.get("analysis_time_s"),
                    model_used=entry.get("model_used"),
                    summary_long=entry.get("summary_long"),
                    facts_json=entry.get("facts_json"),
                    llm_raw_output=entry.get("llm_raw_output"),
                    extract_method=entry.get("extract_method"),
                    extract_time_s=entry.get("extract_time_s"),
                    llm_time_s=entry.get("llm_time_s"),
                    ocr_time_s=entry.get("ocr_time_s"),
                    ocr_mode=entry.get("ocr_mode"),
                    facts_time_s=entry.get("facts_time_s"),
                    facts_llm_time_s=entry.get("facts_llm_time_s"),
                    facts_model_used=entry.get("facts_model_used"),
                    classify_time_s=entry.get("classify_time_s"),
                    classify_llm_time_s=entry.get("classify_llm_time_s"),
                    classify_model_used=entry.get("classify_model_used"),
                )
            except Exception:
                continue
        self._data = data

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        payload = {k: asdict(v) for k, v in self._data.items()}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)

    def get_matching(self, item: ScanItem) -> Optional[CacheEntry]:
        rel = self._rel_path(item.path)
        entry = self._data.get(rel)
        if not entry:
            return None
        if entry.size_bytes != item.size_bytes:
            return None
        if entry.mtime_iso != item.mtime_iso:
            return None
        if entry.status in {"pending", "analysis", "extracting", "scanning", "classifying", "normalizing"}:
            return None
        return entry

    def upsert(self, item: ScanItem) -> None:
        rel = self._rel_path(item.path)
        self._data[rel] = CacheEntry(
            rel_path=rel,
            size_bytes=item.size_bytes,
            mtime_iso=item.mtime_iso,
            status=item.status,
            moved_to=item.moved_to,
            reason=item.reason,
            category=item.category,
            reference_year=item.reference_year,
            proposed_name=item.proposed_name,
            summary=item.summary,
            confidence=item.confidence,
            analysis_time_s=item.analysis_time_s,
            model_used=item.model_used,
            summary_long=item.summary_long,
            facts_json=item.facts_json,
            llm_raw_output=item.llm_raw_output,
            extract_method=item.extract_method,
            extract_time_s=item.extract_time_s,
            llm_time_s=item.llm_time_s,
            ocr_time_s=item.ocr_time_s,
            ocr_mode=item.ocr_mode,
            facts_time_s=item.facts_time_s,
            facts_llm_time_s=item.facts_llm_time_s,
            facts_model_used=item.facts_model_used,
            classify_time_s=item.classify_time_s,
            classify_llm_time_s=item.classify_llm_time_s,
            classify_model_used=item.classify_model_used,
        )

    def invalidate(self, item: ScanItem) -> None:
        rel = self._rel_path(item.path)
        self._data.pop(rel, None)

    def clear(self) -> None:
        self._data.clear()

    def _rel_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.source_root))
        except Exception:
            return str(path)
