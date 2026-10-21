#!/usr/bin/env python3
"""OpportunityMatcher_EventCentric_UPDATED.py

Purpose
-------
Event-centric opportunity classifier: Analyze events independently and classify
strong, *client-gate compliant* business opportunities without user context.

Client Gate Criteria (enforced)
-------------------------------
Hard exclusions (auto-fail):
- price moves
- market recaps
- opinion/editorial
- macro narrative
- funding-only without a clear external action

Minimum evidence to pass (must have at least one):
- explicit partnership/integration
- grants/RFP/program open
- vendor/procurement need
- support added with a clear BD action

Required outputs (Finder/Refiner):
- is_opportunity
- reason (why opportunity)
- evidence_snippet (verbatim)
- recommended_action (ONLY if is_opportunity==true; grounded in evidence)

Pipeline
--------
1. Fetch events from DB
2. BATCHER AGENT: group related events
3. GATEKEEPER AGENT: apply exclusions + extract evidence snippets/types
4. FINDER AGENT: produce compliant opportunity objects for gated events
5. CRITIC AGENT: compliance-first evaluation (keep/reframe/discard)
6. REFINER AGENT: regenerate compliant opportunities based on critic feedback
7. Deterministic confidence = f(sub-scores) (not "vibes")
8. Deduplicate and save to DB

Notes on storage
----------------
The DB schema in this script (opportunities_filter) stores the main opportunity fields
including `opportunity_details`, but does not include separate columns for
is_opportunity/evidence_snippet/recommended_action.
This implementation maps:
- reason: stored as a structured multi-line string including evidence + reason
- suggested_outreach_angle: stored as recommended_action (only for TRUE)
"""

from __future__ import annotations
from urllib.parse import urlparse

import argparse
import csv
import json
import math
import os
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openai import OpenAI



import hashlib
from datetime import timedelta
from pathlib import Path


_NULL_LIKE = {"", "null", "none", "n/a", "na", "unknown", "-"}
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!\?])\s+")
_MOJIBAKE_HINT_RE = re.compile(r"(?:â[\x80-\xbf]|Ã.|Â|ΓÇ)")
_MOJIBAKE_REPLACEMENTS = {
    "â€“": "-",
    "â€”": "-",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€¦": "...",
    "Â·": "-",
    "Â": "",
    "ΓÇô": "-",
    "ΓÇö": "-",
    "ΓÇÖ": "'",
    "ΓÇÖ": "'",
    "ΓÇ£": '"',
    "ΓÇ¥": '"',
    "ΓÇª": "...",
}


def normalize_target_company(raw: Optional[str]) -> Optional[str]:
    """
    Enforce single canonical target company.
    If multiple companies are provided, keep the first non-empty token.
    """
    if not raw:
        return None

    raw = raw.strip()

    # Split on common separators
    parts = re.split(r"[|,/;]", raw)

    for p in parts:
        candidate = p.strip()
        if candidate:
            return candidate

    return None


def clean_text(v: str | None) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in _NULL_LIKE:
        return None
    return s


def _repair_mojibake_text(text: str) -> str:
    s = text or ""
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        s = s.replace(bad, good)
    if _MOJIBAKE_HINT_RE.search(s):
        try:
            repaired = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired and _MOJIBAKE_HINT_RE.search(repaired) is None:
                s = repaired
        except Exception:
            pass
    return s


def sanitize_opportunity_field_text(v: Optional[str]) -> Optional[str]:
    text = clean_text(v)
    if text is None:
        return None
    text = _repair_mojibake_text(text)
    text = unicodedata.normalize("NFKD", text)
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
        .replace("\u00a0", " ")
    )
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def smart_truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[: max_chars - 3].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return (cut or text[: max_chars - 3].rstrip()) + "..."


def normalize_title_text(title: Optional[str], max_chars: int = 140) -> str:
    title = (sanitize_opportunity_field_text(title) or "").strip()
    title = re.sub(r"^\s*Opportunity\s*:\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" -:;,.")
    if len(title) > max_chars:
        title = smart_truncate(title, max_chars).strip(" -:;,.")
    return title or "Opportunity"


def _normalize_sentence(text: Optional[str]) -> str:
    s = re.sub(r"\s+", " ", (sanitize_opportunity_field_text(text) or "").strip()).strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def normalize_summary_text(summary: Optional[str], fallback_parts: List[Optional[str]], min_words: int = 35, max_chars: int = 600) -> str:
    sentences = [_normalize_sentence(s) for s in _SENT_SPLIT_RE.split((sanitize_opportunity_field_text(summary) or "").strip()) if _normalize_sentence(s)]
    extras: List[str] = []
    for part in fallback_parts:
        for sentence in _SENT_SPLIT_RE.split((sanitize_opportunity_field_text(part) or "").strip()):
            normalized = _normalize_sentence(sentence)
            if normalized and normalized not in extras:
                extras.append(normalized)

    for extra in extras:
        if len(sentences) >= 4:
            break
        if extra not in sentences:
            sentences.append(extra)

    while len(sentences) < 3 and extras:
        extra = extras.pop(0)
        if extra not in sentences:
            sentences.append(extra)

    sentences = sentences[:4]
    out = re.sub(r"\s+", " ", " ".join(sentences)).strip()
    if len(out) > max_chars:
        out = smart_truncate(out, max_chars)
    trimmed = [s.strip() for s in _SENT_SPLIT_RE.split(out) if s.strip()]
    if len(trimmed) > 4:
        out = " ".join(trimmed[:4]).strip()
    if _word_count(out) < min_words:
        for extra in extras:
            candidate = re.sub(r"\s+", " ", f"{out} {extra}").strip()
            if len(candidate) > max_chars:
                continue
            out = candidate
            if _word_count(out) >= min_words:
                break
    return out or "A grounded BD opportunity was identified with a concrete action surface and a plausible outreach motion."


def normalize_reason_text(reason: Optional[str], max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", (sanitize_opportunity_field_text(reason) or "").strip()).strip()
    return smart_truncate(text, max_chars) if text else ""


def normalize_outreach_text(outreach: Optional[str], fallback: Optional[str] = None, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", (sanitize_opportunity_field_text(outreach) or sanitize_opportunity_field_text(fallback) or "").strip()).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return smart_truncate(text, max_chars) if text else ""


def normalize_details_text(details: Optional[str], min_words: int = 70, max_chars: int = 1600) -> Optional[str]:
    text = re.sub(r"\s+", " ", (sanitize_opportunity_field_text(details) or "").strip()).strip()
    if not text:
        return ""
    sentences = [_normalize_sentence(s) for s in _SENT_SPLIT_RE.split(text) if _normalize_sentence(s)]
    if len(sentences) < 6:
        return ""
    if len(sentences) > 15:
        sentences = sentences[:15]
    text = " ".join(sentences).strip()
    if len(text) > max_chars:
        text = smart_truncate(text, max_chars)
    trimmed_sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(trimmed_sentences) < 6 or len(trimmed_sentences) > 15:
        return ""
    if _word_count(text) < min_words:
        return ""
    return text




def extract_reason_text_from_reason(reason_block: Optional[str]) -> str:
    text = sanitize_opportunity_field_text(reason_block) or ""
    m = re.search(r"Reason:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return re.sub(r"\s+", " ", text).strip()


def build_fallback_opportunity_details(opportunity: "Opportunity", source_events: List["Event"]) -> str:
    summary = normalize_summary_text(
        opportunity.summary,
        [extract_reason_text_from_reason(opportunity.reason)],
        min_words=20,
        max_chars=520,
    )
    reason_text = normalize_reason_text(extract_reason_text_from_reason(opportunity.reason), max_chars=420)
    evidence_snippet = sanitize_opportunity_field_text(extract_evidence_snippet_from_reason(opportunity.reason) or "") or ""
    outreach = normalize_outreach_text(opportunity.suggested_outreach_angle, opportunity.suggested_outreach_angle, max_chars=260)
    company = sanitize_opportunity_field_text(opportunity.target_company) or sanitize_opportunity_field_text(opportunity.title) or "The team"

    source_sentences: List[str] = []
    for ev in source_events[:2]:
        for part in [ev.summary, ev.description]:
            for sentence in _SENT_SPLIT_RE.split((sanitize_opportunity_field_text(part) or "").strip()):
                normalized = _normalize_sentence(sentence)
                if normalized and normalized not in source_sentences:
                    source_sentences.append(normalized)
                if len(source_sentences) >= 3:
                    break
            if len(source_sentences) >= 3:
                break
        if len(source_sentences) >= 3:
            break

    details_parts = [
        summary,
        reason_text,
        source_sentences[0] if len(source_sentences) > 0 else "",
        source_sentences[1] if len(source_sentences) > 1 else "",
        f"Key source evidence: {evidence_snippet}." if evidence_snippet else "",
        f"Recommended outreach should lead with {outreach[:-1] if outreach.endswith('.') else outreach}." if outreach else "",
        f"This opportunity should be framed around the actual opening with {company}, while preserving the source's current stage and constraints.",
    ]
    raw = " ".join(p.strip() for p in details_parts if p and p.strip())
    normalized = normalize_details_text(raw, min_words=40, max_chars=1600)
    if normalized:
        return normalized
    return smart_truncate(raw, 1600)

def derive_target_company(event_urls: list[str], sources: list[str]) -> str | None:
    # 1) Prefer URL domain (deterministic, no hardcoding)
    for u in (event_urls or []):
        u = (u or "").strip()
        if not u:
            continue
        try:
            host = urlparse(u).netloc.strip().lower()
            if host:
                return host
        except Exception:
            continue

    # 2) Fallback: source label
    for s in (sources or []):
        s = (s or "").strip()
        if not s:
            continue
        if ":" in s:
            left = s.split(":", 1)[0].strip()
            if left:
                return left
        return s

    return None


# ==========================================================
# Hardcoded file paths (from Model2_v12)
# - category_master.json sits in the same folder as this script
# - Final_events.jsonl sits inside ./data (or ./Data) under this script folder
# ==========================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Fixed output folder for redundant artifacts; fall back to script dir if missing.
FIXED_OUTPUT_DIR = Path(r"C:\Sukhendu\UPWORK-WORK\Fiverr-CAM-Python\0_TEMP")
OUTPUT_DIR = FIXED_OUTPUT_DIR if FIXED_OUTPUT_DIR.exists() else SCRIPT_DIR

# Prefer ./data, but support ./Data for Windows environments where it already exists.
DATA_DIR = SCRIPT_DIR / "data"
if not DATA_DIR.exists():
    alt = SCRIPT_DIR / "Data"
    if alt.exists():
        DATA_DIR = alt
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Final events JSONL (hardcoded)
_EVENTS_CANDIDATES = [
    SCRIPT_DIR / "Data" / "Final_events.jsonl",
    SCRIPT_DIR / "Data" / "final_events.jsonl",
    SCRIPT_DIR / "data" / "Final_events.jsonl",
    SCRIPT_DIR / "data" / "final_events.jsonl",
]
FINAL_EVENTS_JSONL_PATH = next((pp for pp in _EVENTS_CANDIDATES if pp.exists()), _EVENTS_CANDIDATES[0])

# Category master JSON (hardcoded)
CATEGORY_MASTER_JSON_PATH = SCRIPT_DIR / "category_master.json"


# ---------------------------
# JSONL / Category File Loading
# ---------------------------

def _read_database_config(filepath: str) -> Dict[str, str]:
    """
    Read database configuration from CSV file.

    
    CSV file should have columns: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SCHEMA
    Returns a dictionary with connection parameters.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Database config file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Database config file is empty or has no headers")

        config_row = next(reader, None)
        if config_row is None:
            raise ValueError("Database config file has no data rows")

    required_fields = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    for field in required_fields:
        if field not in config_row or not config_row[field]:
            raise ValueError(f"Missing or empty required field in database config: {field}")

    return {
        "DB_HOST": config_row["DB_HOST"].strip(),
        "DB_PORT": config_row["DB_PORT"].strip(),
        "DB_USER": config_row["DB_USER"].strip(),
        "DB_PASSWORD": config_row["DB_PASSWORD"].strip(),
        "DB_NAME": config_row["DB_NAME"].strip(),
        "DB_SCHEMA": config_row.get("DB_SCHEMA", "public").strip() or "public",
    }

def _try_paths(path_str: str, candidates: List[str]) -> str:
    """Return first existing path among provided candidates."""
    if path_str and os.path.exists(path_str):
        return path_str
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return path_str


def load_events_from_jsonl(path_str: str, days_back: int = 30, limit: Optional[int] = None) -> List["Event"]:
    """Load events from a JSONL file (one JSON object per line).

    Expected keys (best-effort): event_id, title, summary_text or summary,
    body_text or description, published_at, source, url.
    """
    # Try a few common relative locations without requiring CLI usage.
    path_str = _try_paths(path_str, [
        "Final_events.jsonl",
        "final_events.jsonl",
        os.path.join("data", "Final_events.jsonl"),
        os.path.join("data", "final_events.jsonl"),
        os.path.join("Data", "Final_events.jsonl"),
        os.path.join("Data", "final_events.jsonl"),
        os.path.join(os.getcwd(), "data", "Final_events.jsonl"),
        os.path.join(os.getcwd(), "Data", "Final_events.jsonl"),
    ])
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"events JSONL not found: {path_str}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=int(days_back))
    events: List[Event] = []

    def _parse_dt(v: Any) -> Optional[datetime]:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if isinstance(v, (int, float)):
            try:
                return datetime.fromtimestamp(float(v), tz=timezone.utc)
            except Exception:
                return None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                dt = datetime.fromisoformat(s)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                return None
        return None

    with open(path_str, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            published_raw = obj.get("published_at") or obj.get("published") or obj.get("date") or obj.get("time")
            dt = _parse_dt(published_raw)
            if dt and dt < cutoff:
                continue

            event_id = obj.get("event_id") or obj.get("id") or obj.get("guid") or ""
            if not event_id:
                # stable fallback if missing
                event_id = hashlib.sha1((obj.get("url") or line).encode("utf-8", errors="ignore")).hexdigest()[:24]

            title = obj.get("title") or ""
            summary = obj.get("AI_summary_text") or obj.get("summary_text") or obj.get("summary") or obj.get("rss_summary_text") or ""
            body = obj.get("body_text") or obj.get("description") or obj.get("content") or ""
            source = obj.get("source") or obj.get("source_name") or ""
            url = obj.get("url") or obj.get("link") or ""
            ai_enrichment = obj.get("ai_enrichment") if isinstance(obj.get("ai_enrichment"), dict) else {}
            bd_signal = obj.get("bd_signal") if isinstance(obj.get("bd_signal"), dict) else ai_enrichment.get("bd_signal")
            event_type = ai_enrichment.get("event_type")
            if event_type is not None:
                event_type = str(event_type).strip()
            else:
                event_type = ""
            key_points = ai_enrichment.get("key_points") if isinstance(ai_enrichment.get("key_points"), list) else []
            why_it_matters = ai_enrichment.get("why_it_matters") if isinstance(ai_enrichment.get("why_it_matters"), list) else []
            recommended_action = ai_enrichment.get("recommended_action")
            if recommended_action is not None:
                recommended_action = str(recommended_action).strip()
            else:
                recommended_action = ""
            contact_leads = ai_enrichment.get("contact_leads") if isinstance(ai_enrichment.get("contact_leads"), list) else []

            events.append(Event(
                event_id=str(event_id),
                title=title,
                summary=summary,
                description=body,
                tags=[],
                published_at=dt.isoformat() if dt else (published_raw or ""),
                source=source,
                url=url,
                event_type=event_type,
                key_points=[str(x).strip() for x in key_points if str(x).strip()][:5],
                bd_signal=bd_signal if isinstance(bd_signal, dict) else None,
                why_it_matters=[str(x).strip() for x in why_it_matters if str(x).strip()][:5],
                recommended_action=recommended_action,
                contact_leads=contact_leads[:5],
            ))
            if limit and len(events) >= limit:
                break

    # Sort newest first when dates are present
    def _sort_key(e: Event):
        try:
            s = (e.published_at or "").replace("Z", "+00:00")
            return datetime.fromisoformat(s) if s else datetime.min.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    events.sort(key=_sort_key, reverse=True)
    return events


FILTER_MASTER_JSON_PATH = SCRIPT_DIR / "filters_master.json"

def openai_filter_schema():
    return {
        "name": "filter_classification",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "opportunity_id": {"type": "string"},
                            "filter_chain": {"type": "array", "items": {"type": "string"}},
                            "filter_sector": {"type": "array", "items": {"type": "string"}},
                            "filter_seeking": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "opportunity_id",
                            "filter_chain",
                            "filter_sector",
                            "filter_seeking"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        }
    }


def load_filter_master(path: str):
    """
    Loads filters_master.json and returns:
    - chains_map:  {filter_text_id: full_object}
    - sectors_map: {filter_text_id: full_object}
    - seeking_map: {filter_text_id: full_object}
    """

    if not os.path.exists(path):
        print(f"[OpportunityMatcher] Filter master file not found: {path}")
        return {}, {}, {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[OpportunityMatcher] Failed to read filter master JSON: {e}")
        return {}, {}, {}

    chains_map = {}
    sectors_map = {}
    seeking_map = {}

    filters = data.get("filters", [])

    for item in filters:
        if not isinstance(item, dict):
            continue

        if not item.get("enabled", False):
            continue

        ftype = item.get("filter_type")
        fid = item.get("filter_text_id")

        if not ftype or not fid:
            continue

        if ftype == "chain":
            chains_map[fid] = item
        elif ftype == "sector":
            sectors_map[fid] = item
        elif ftype == "seeking":
            seeking_map[fid] = item

    print(
        f"[OpportunityMatcher] Loaded Filters: "
        f"chains={len(chains_map)}, "
        f"sectors={len(sectors_map)}, "
        f"seeking={len(seeking_map)}"
    )

    return chains_map, sectors_map, seeking_map




def load_categories_map(path_str: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Load categories from category_master.json.

    Supports CalyxOnAI category_master.json shape:
    {
      "version": "...",
      "categories": [
        {
          "category_id": 4,
          "category_name": "...",
          "category_text_id": "tech",
          "definition": "...",
          "active": true
        }
      ]
    }

    Returns:
        categories_map: {alias_or_id_or_name -> category_text_id}
        category_definitions: {category_text_id -> definition}
        category_names: {category_text_id -> category_name}
    """

    path_str = _try_paths(path_str, [
        "category_master.json",
        os.path.join("data", "category_master.json"),
        os.path.join("Data", "category_master.json"),
        os.path.join(os.getcwd(), "data", "category_master.json"),
        os.path.join(os.getcwd(), "Data", "category_master.json"),
        os.path.join(os.getcwd(), "category_master.json"),
    ])
    if not os.path.exists(path_str):
        return {}, {}, {}

    with open(path_str, "r", encoding="utf-8") as f:
        data = json.load(f)

    def _norm(x: Any) -> str:
        return re.sub(r"\s+", " ", str(x or "").strip())

    categories_map: Dict[str, str] = {}
    category_definitions: Dict[str, str] = {}
    category_names: Dict[str, str] = {}

    # Preferred: dict with "categories": [...]
    if isinstance(data, dict) and isinstance(data.get("categories"), list):
        for r in data.get("categories", []) or []:
            if not isinstance(r, dict):
                continue

            cid = r.get("category_id")
            text_id = _norm(r.get("category_text_id"))
            name = _norm(r.get("category_name") or r.get("name") or r.get("label"))
            definition = _norm(r.get("definition"))
            active = r.get("active", True)

            if not active:
                continue
            if not text_id:
                continue

            # alias/id/name -> canonical category_text_id
            if cid is not None and _norm(cid):
                categories_map[_norm(cid)] = text_id
            if name:
                categories_map[name] = text_id
            categories_map[text_id] = text_id

            if name:
                category_names[text_id] = name
            if definition:
                category_definitions[text_id] = definition

        return categories_map, category_definitions, category_names

    # Legacy fallbacks (best effort, but should not be needed now)
    if isinstance(data, dict):
        for k, v in data.items():
            kk = _norm(k)
            if kk:
                categories_map[kk] = _norm(v)
        return categories_map, category_definitions, category_names

    if isinstance(data, list):
        for r in data:
            if not isinstance(r, dict):
                continue
            text_id = _norm(r.get("category_text_id") or r.get("text_id") or r.get("id") or r.get("key"))
            name = _norm(r.get("category_name") or r.get("name") or r.get("label") or text_id)
            definition = _norm(r.get("definition"))
            if not text_id:
                continue
            categories_map[text_id] = text_id
            if name:
                categories_map[name] = text_id
                category_names[text_id] = name
            if definition:
                category_definitions[text_id] = definition

    return categories_map, category_definitions, category_names

# ---------------------------
# Reachability / Realism Gate (Client Feedback)
# ---------------------------

MEGA_COUNTERPARTIES = {
    "standard chartered", "coinbase", "cftc", "sec", "fca", "bny", "bny mellon", "jpmorgan",
    "visa", "mastercard", "blackrock", "goldman", "morgan stanley", "citigroup", "citi",
    "hsbc", "barclays", "bank of america", "bofa", "fidelity", "binance", "kraken",
    "european central bank", "ecb", "federal reserve", "treasury", "imf", "world bank",
}

OPEN_MOTION_RE = re.compile(
    r"\b(apply|applications? open|call for proposals|rfp|rfq|rfi|tender|procurement|vendor|partner program|partners? portal|join(?:ing)? program|accepting (?:partners|vendors|bids))\b",
    re.IGNORECASE,
)

ELLIPSIS_RE = re.compile(r"\.{3,}")

def is_mega_counterparty(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    for k in MEGA_COUNTERPARTIES:
        if k in t:
            return True
    return False

def has_open_motion(evidence_type: str, evidence_snippet: str, title: str = "", target_company: str = "") -> bool:
    combined = " ".join([title or "", target_company or "", evidence_snippet or ""]).strip()
    if OPEN_MOTION_RE.search(combined):
        return True
    # Evidence-type based allowances
    if evidence_type in {"grant_or_rfp_or_program_open", "vendor_or_procurement_need", "token_launch_or_tge"}:
        return True
    if evidence_type == "support_added_with_bd_action" and OPEN_MOTION_RE.search(evidence_snippet or ""):
        return True
    # Partnerships are only acceptable for mega counterparties if it is an explicit named partnership/integration.
    if evidence_type == "partnership_or_integration":
        return True
    return False

def enforce_verbatim_snippet(snippet: Optional[str], max_words: int = 38) -> Optional[str]:
    if snippet is None:
        return None
    s = (snippet or "").strip()
    if not s:
        return None
    # Remove ellipses markers; keep text verbatim otherwise (no paraphrase).
    s = ELLIPSIS_RE.sub(" ", s).strip()
    # Cap to ~max_words by taking the first max_words words (still a verbatim prefix).
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s.strip() or None



def slice_body_for_gatekeeper(desc: Optional[str], head: int = 4000, tail: int = 2000) -> str:
    """Return first `head` + last `tail` characters of desc for Gatekeeper grounding.
    Total default size = 6,000 chars (4,000 head + 2,000 tail), preserving top context and bottom intake/contact lines.
    If desc is shorter than head+tail, return as-is.
    """
    s = clean_text(desc) or ""
    if len(s) <= head + tail:
        return s
    return s[:head] + "\n...\n" + s[-tail:]


def truncate_bd_signal_for_gatekeeper(signal: Any, max_str: int = 300, max_items: int = 20, max_json: int = 2000) -> Any:
    """Prevent bd_signal from inflating the Gatekeeper prompt."""
    if not isinstance(signal, dict):
        return signal

    def _truncate(val: Any) -> Any:
        if isinstance(val, str):
            return smart_truncate(val, max_str)
        if isinstance(val, list):
            return [_truncate(v) for v in val[:max_items]]
        if isinstance(val, dict):
            out = {}
            for k in list(val.keys())[:max_items]:
                out[k] = _truncate(val[k])
            return out
        return val

    out = _truncate(signal)
    try:
        if len(json.dumps(out, default=_json_safe)) > max_json:
            keep_keys = [k for k in [
                "has_action_surface", "action_surface_type", "action_detail",
                "evidence_type", "evidence_snippet", "confidence", "source"
            ] if k in out]
            if keep_keys:
                return {k: out[k] for k in keep_keys}
            return {"truncated": True}
    except Exception:
        return {"truncated": True}
    return out


def gatekeeper_payload_limits() -> Dict[str, int]:
    """Use a consistent Gatekeeper payload budget across all horizons."""
    return {
        "summary_max": 800,
        "body_head": 6000,
        "body_tail": 3000,
        "bd_max_str": 300,
        "bd_max_items": 20,
        "bd_max_json": 2000,
    }


def normalize_critic_rating(value: Any) -> float:
    """Normalize critic ratings to a 0-10 scale."""
    try:
        rating = float(value)
    except Exception:
        return 0.0
    if 0.0 <= rating <= 1.0:
        return rating * 10.0
    return rating


_GATEKEEPER_UNUSABLE_BODY_RE = re.compile(
    r"("
    r"the forum only sets cookies|"
    r"by clicking on accept you consent|"
    r"accept (all )?cookies|"
    r"cookie settings|"
    r"privacy policy|"
    r"terms of service|"
    r"sign in to continue|"
    r"log in to continue|"
    r"login required|"
    r"enable javascript|"
    r"access denied|"
    r"subscribe to continue"
    r")",
    re.IGNORECASE,
)


def gatekeeper_body_is_unusable(text: Optional[str]) -> bool:
    cleaned = re.sub(r"\s+", " ", (clean_text(text) or "").strip()).strip()
    if not cleaned:
        return True
    if _GATEKEEPER_UNUSABLE_BODY_RE.search(cleaned):
        return True
    if len(cleaned) < 80 and re.search(r"\b(cookie|consent|login|sign in|privacy|terms)\b", cleaned, flags=re.IGNORECASE):
        return True
    return False


def slice_body_for_finder(desc: Optional[str], head: int = 4000, tail: int = 2000) -> str:
    """Return first `head` + last `tail` characters of desc (captures contact/boilerplate at end).
    If desc is shorter than head+tail, return as-is.
    """
    s = (desc or "")
    if len(s) <= head + tail:
        return s
    return s[:head] + "\n...\n" + s[-tail:]


# ---------------------------
# Optional Imports
# ---------------------------

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None

try:
    from openai import OpenAI, APIConnectionError, APITimeoutError
except Exception:
    OpenAI = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


class OpenAIRequestTooLarge(Exception):
    pass


class GatekeeperTruncatedError(Exception):
    pass


class GatekeeperParseError(Exception):
    pass


class TokenTracker:
    """Track OpenAI token usage and estimated costs."""

    def __init__(self):
        self.model_usage = {}
        self.agent_usage = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_estimated_cost = 0.0
        self.models_missing_pricing = set()
        self.models_missing_pricing_warned = set()

    def get_pricing(self, model: str):
        """Return (input_cost_per_1m, output_cost_per_1m) in USD per 1M tokens."""
        pricing = {
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
            "gpt-4.1": (2.0, 8.0),
            "gpt-4.1-mini": (0.40, 1.60),
            "o3": (2.0, 8.0),
            "gpt-5.1": (1.25, 10.0),
            "o3-mini": (1.1, 4.4),
            "gpt-5-mini": (0.25, 2.0),
        }
        return pricing.get(model)

    def resolve_pricing(self, model: str, max_attempts: int = 2):
        """Try a couple of pricing lookups before failing open with zero-cost accounting."""
        attempts = [model]

        # Second attempt: lightweight alias normalization for common model names.
        normalized = re.sub(r"-(latest|preview)$", "", (model or "").strip(), flags=re.IGNORECASE)
        if normalized and normalized not in attempts:
            attempts.append(normalized)

        for idx, candidate in enumerate(attempts[:max_attempts], 1):
            pricing = self.get_pricing(candidate)
            if pricing:
                return pricing, idx

        return None, min(max_attempts, len(attempts))

    def _ensure_model(self, model: str):
        if model not in self.model_usage:
            self.model_usage[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0,
                "pricing_known": True,
            }

    def _ensure_agent(self, agent: str):
        if agent not in self.agent_usage:
            self.agent_usage[agent] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0,
                "pricing_known": True,
                "models": set(),
            }

    def add_usage(self, usage, model: str = "gpt-4o", agent: Optional[str] = None):
        if not usage:
            return

        self._ensure_model(model)
        if agent:
            self._ensure_agent(agent)
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)

        self.model_usage[model]["input_tokens"] += input_tokens
        self.model_usage[model]["output_tokens"] += output_tokens
        if agent:
            self.agent_usage[agent]["input_tokens"] += input_tokens
            self.agent_usage[agent]["output_tokens"] += output_tokens
            self.agent_usage[agent]["models"].add(model)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        pricing, attempts_used = self.resolve_pricing(model, max_attempts=2)
        if not pricing:
            self.model_usage[model]["pricing_known"] = False
            self.models_missing_pricing.add(model)
            if agent:
                self.agent_usage[agent]["pricing_known"] = False
            if model not in self.models_missing_pricing_warned:
                print(
                    f"[OpportunityMatcher] WARNING: No pricing found for model '{model}' after "
                    f"{attempts_used} attempts. Using $0.00 estimated cost for this model and continuing."
                )
                self.models_missing_pricing_warned.add(model)
            return

        input_cost_per_1m, output_cost_per_1m = pricing
        cost = (input_tokens / 1_000_000) * input_cost_per_1m + (output_tokens / 1_000_000) * output_cost_per_1m
        self.model_usage[model]["estimated_cost"] += cost
        if agent:
            self.agent_usage[agent]["estimated_cost"] += cost
        self.total_estimated_cost += cost

    def print_summary(self):
        print("\n" + "=" * 72)
        print("[OpportunityMatcher] AGENT-WISE USAGE SUMMARY")
        print("=" * 72)

        for agent, data in self.agent_usage.items():
            total_tokens = data["input_tokens"] + data["output_tokens"]
            models = ", ".join(sorted(data.get("models", set())))
            print(f"\nAgent: {agent}")
            print("-" * 72)
            print(f"  Models used:    {models or 'unknown'}")
            print(f"  Input tokens:   {data['input_tokens']:,}")
            print(f"  Output tokens:  {data['output_tokens']:,}")
            print(f"  Total tokens:   {total_tokens:,}")
            if data.get("pricing_known", True):
                print(f"  Estimated cost: ${data['estimated_cost']:.4f}")
            else:
                print(f"  Estimated cost: >=${data['estimated_cost']:.4f} (some model pricing unknown)")

        print("\n" + "=" * 72)
        print("[OpportunityMatcher] MODEL-WISE USAGE SUMMARY")
        print("=" * 72)

        for model, data in self.model_usage.items():
            total_tokens = data["input_tokens"] + data["output_tokens"]
            print(f"\nModel: {model}")
            print("-" * 72)
            print(f"  Input tokens:   {data['input_tokens']:,}")
            print(f"  Output tokens:  {data['output_tokens']:,}")
            print(f"  Total tokens:   {total_tokens:,}")
            if data.get("pricing_known", True):
                print(f"  Estimated cost: ${data['estimated_cost']:.4f}")
            else:
                print("  Estimated cost: unknown (pricing not configured)")

        print("\n" + "=" * 72)
        print("[OpportunityMatcher] TOTAL USAGE SUMMARY")
        print("=" * 72)

        total_tokens = self.total_input_tokens + self.total_output_tokens
        print(f"  Total input tokens:   {self.total_input_tokens:,}")
        print(f"  Total output tokens:  {self.total_output_tokens:,}")
        print(f"  Grand total tokens:   {total_tokens:,}")
        if self.models_missing_pricing:
            missing = ", ".join(sorted(self.models_missing_pricing))
            print(f"  Total estimated cost: >=${self.total_estimated_cost:.4f} (excluding models with unknown pricing: {missing})")
        else:
            print(f"  Total estimated cost: ${self.total_estimated_cost:.4f}")
        print("=" * 72)


def load_openai_api_keys() -> List[str]:
    keys: List[str] = []
    for env_name in ("OPENAI_API_KEY_1", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3", "OPENAI_API_KEY"):
        value = os.getenv(env_name)
        if value and value not in keys:
            keys.append(value)
    return keys


class ThreadLocalOpenAIClientPool:
    """Assign one OpenAI API key per worker thread, with fallback cycling."""

    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("At least one OpenAI API key is required")
        self._api_keys = list(api_keys)
        self._lock = threading.Lock()
        self._next_key_idx = 0
        self._local = threading.local()

    def get_client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is not None:
            return client
        with self._lock:
            key = self._api_keys[self._next_key_idx % len(self._api_keys)]
            self._next_key_idx += 1
        client = OpenAI(api_key=key)
        self._local.client = client
        self._local.api_key_tail = key[-5:]
        return client


def pretty_thread_name() -> str:
    raw = threading.current_thread().name
    m = re.search(r"_(\d+)$", raw)
    if m:
        return f"Thread_{int(m.group(1)) + 1}"
    return raw


# ---------------------------
# Configuration
# ---------------------------

DEFAULT_FILTER_MODEL = "gpt-4.1-mini"
FILTER_BATCH_SIZE = 10
MAX_BATCH_WORKERS = 3

DEFAULT_MODEL = "gpt-5.1"      # Finder + Batcher default
DEFAULT_GATEKEEPER_MODEL = "gpt-4o-mini"
DEFAULT_CRITIC_MODEL = "o3"
DEFAULT_ENRICHMENT_MODEL = "gpt-4.1"
DEFAULT_REFINER_MODEL = "gpt-5.1"
DEFAULT_RECOVERY_MODEL = "o3"
DEFAULT_WATCHLIST_PROMOTION_MODEL = "gpt-5.1"

DEFAULT_MAX_EVENTS_PER_BATCH = 10
DEFAULT_MAX_OUTPUT_TOKENS = 2000
DEFAULT_MIN_CONFIDENCE = 0.60
RECOVERY_MAX_CANDIDATES = 20
RECOVERY_MAX_RESTORED = 3
RECOVERY_FALLBACK_CANDIDATES = 8
RECOVERY_CHUNK_SIZE = 10
RECOVERY_MAX_EMPTY_CHUNKS = 2
RECOVERY_MIN_CONFIDENCE = 0.45
RECOVERY_MIN_CANDIDATES_TO_RUN = 6
DROP_RECOVERY_MIN_CANDIDATES_TO_RUN = 1
DROP_RECOVERY_MIN_CONFIDENCE_FLOOR = 0.45
DROP_RECOVERY_RATIO_FORCE_ONE = 0.40
DROP_RECOVERY_RATIO_FORCE_TWO = 0.60
DROP_RECOVERY_RATIO_FORCE_THREE = 0.80
RECOVERY_SKIP_IF_PRE_DEDUPE_COUNT_AT_LEAST = 25
WATCHLIST_PROMOTION_MAX_CANDIDATES = 15
WATCHLIST_PROMOTION_MAX_PROMOTED = 6


# Time horizon (weeks) used across ALL agent prompts; overridden by CLI flag --time-horizon-weeks
TIME_HORIZON_WEEKS = 12
TIME_HORIZON_STR = "1â€“12 weeks"
MAX_OPENAI_RETRIES = 3
OPENAI_RETRY_BACKOFF = 3.0

BATCH_SHRINK_STEP = 5
MIN_EVENTS_PER_BATCH = 3

# ---------------------------
# Category Guardrails
# ---------------------------

# Evidence-type -> allowed category_text_id set
ALLOWED_BY_EVIDENCE: Dict[str, set[str]] = {
    # Narrow: only true open programs / RFPs / calls.
    "grant_or_rfp_or_program_open": {"grant"},

    # Explicit procurement / vendor selection signals can also imply
    # technical integration or partnership motion in public intake cases.
    "vendor_or_procurement_need": {"proc", "tech", "part"},

    # Partnerships/integrations: can be technical integration or partnership motion.
    "partnership_or_integration": {"tech", "part"},

    # Token launch / TGE is its own category (primary).
    "token_launch_or_tge": {"tge"},

    # Institutional allocation/treasury decisions can create BD motion (custody, infra, partnerships).
    # Keep this permissive but still definition-enforced downstream.
    "institutional_allocation_or_treasury": {"part", "tech", "comm"},

    # Regulatory/compliance updates: only keep if truly actionable; category is usually "comm" (compliance/commercial)
    # or "tech" when it implies an integration path. Definition enforcement remains the hard check.
    "regulatory_or_compliance_update": {"comm", "tech", "part"},

    # "support_added_with_bd_action" can overlap multiple BD motions.
    "support_added_with_bd_action": {"tech", "part", "comm", "tge"},
}

# Strict "gener" rule: only allow "gener" for these evidence types.
STRICT_GENER_ALLOWED_EVIDENCE: set[str] = {"support_added_with_bd_action"}

# Cardinality guardrail: cap categories per opportunity (post-validation)
MAX_CATEGORIES_PER_OPPORTUNITY = 2

# Lightweight normalization for common non-canonical LLM category tokens.
# These aliases are intentionally narrow and still flow through evidence-type
# guardrails plus definition-consistency checks downstream.
CATEGORY_TOKEN_ALIASES: Dict[str, str] = {
    "amm": "tech",
    "api": "tech",
    "audit": "proc",
    "bridge": "tech",
    "bridging": "tech",
    "compliance": "comm",
    "custody": "tech",
    "decentralized-storage": "tech",
    "dex": "tech",
    "infra-integration": "tech",
    "infrastructure": "tech",
    "integration": "tech",
    "integrations": "tech",
    "liquidity": "tech",
    "payments": "comm",
    "sdk": "tech",
    "security": "tech",
    "staking": "tech",
    "validator": "tech",
    "validators": "tech",
    "wallet": "tech",
    "wallets": "tech",
    "zero-knowledge": "tech",
    "zero knowledge": "tech",
}


def _now_iso() -> str:
    return datetime.now().date().isoformat()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def confidence_from_subscores(sub: Dict[str, int]) -> float:
    """Deterministic confidence from sub-scores.

    Expected keys: fit_score, evidence_score, actionability_score, feasibility_score (each 0..3).
    """
    keys = ["fit_score", "evidence_score", "actionability_score", "feasibility_score"]
    total = 0
    for k in keys:
        v = int(sub.get(k, 0))
        v = max(0, min(3, v))
        total += v
    return _clamp(total / 12.0)


# ---------------------------
# Data Models
# ---------------------------

@dataclass
class Event:
    event_id: str
    title: Optional[str]
    summary: Optional[str]
    description: Optional[str]
    tags: List[str]
    published_at: Optional[str]
    source: Optional[str]
    url: Optional[str]
    event_type: Optional[str] = None
    key_points: Optional[List[str]] = None
    bd_signal: Optional[Dict[str, Any]] = None
    why_it_matters: Optional[List[str]] = None
    recommended_action: Optional[str] = None
    contact_leads: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title or "",
            "summary": self.summary or "",
            "description": slice_body_for_finder(self.description),
            "tags": self.tags or [],
            "published_at": self.published_at or "",
            "source": self.source or "",
            "url": self.url or "",
            "event_type": self.event_type or "",
            "key_points": self.key_points or [],
            "bd_signal": self.bd_signal or {},
            "why_it_matters": self.why_it_matters or [],
            "recommended_action": self.recommended_action or "",
            "contact_leads": self.contact_leads or [],
        }


@dataclass
class GateResult:
    event_id: str
    hard_exclusion: bool
    hard_exclusion_reason: Optional[str]
    evidence_type: str
    evidence_snippet: Optional[str]
    supporting_snippet: Optional[str] = None

    def passes(self) -> bool:
        return (not self.hard_exclusion) and self.evidence_type != "none" and bool(self.evidence_snippet)


def normalize_hard_exclusion_reason(reason: Optional[str], evidence_type: Optional[str] = None) -> str:
    """Normalize malformed/empty hard exclusion reasons from gatekeeper output.

    We occasionally receive boolean-like placeholders (e.g., "true") instead of
    textual reasons; convert those into stable reason codes for audit quality.
    """
    txt = clean_text(reason) or ""
    lowered = txt.lower()
    bad_literals = {
        "",
        "true",
        "false",
        "hard_exclusion=true",
        "hard_exclusion=false",
        "hard exclusion true",
        "hard exclusion false",
        "1",
        "0",
        "yes",
        "no",
        "none",
        "null",
        "n/a",
        "na",
    }
    if lowered in bad_literals or lowered.startswith("hard_exclusion="):
        return "no_qualifying_evidence" if (evidence_type or "").strip().lower() == "none" else "hard_exclusion_without_reason"
    return txt


@dataclass
class Opportunity:
    opportunity_id: str
    title: str
    summary: str
    reason: str
    who_to_contact: Optional[str]
    suggested_outreach_angle: str  # mapped to recommended_action
    categories: List[str]
    filter_chain: str
    filter_sector: str
    filter_seeking: str
    time_found: str
    confidence: float
    tags: List[str]
    target_company: Optional[str]
    sources: List[str]
    event_ids: str  # pipe-separated
    event_titles: str  # pipe-separated
    event_url: str  # pipe-separated
    bd_weeks: int
    evidence_type: str = ""
    evidence_snippet: str = ""
    supporting_snippet: str = ""
    opportunity_details: Optional[str] = None
    audit_label: Optional[str] = None
    finalizer_reason: Optional[str] = None


# ---------------------------
# Database Operations
# ---------------------------

class DatabaseManager:
    def __init__(self, host: str, port: int, user: str, password: str, database: str, schema: str = "public"):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 not installed")

        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        # Explicit schema control helps when the DB user has a non-default search_path.
        self.schema = (schema or "public").strip()

        self.conn = psycopg2.connect(host=host, port=port, user=user, password=password, database=database)
        self.conn.autocommit = False

        # Ensure queries hit the expected schema (matches your older script behavior in practice).
        self._set_search_path()

    def _set_search_path(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SET search_path TO {self.schema}")
        except Exception:
            # If permissions disallow SET, continue; queries may still work if fully qualified.
            pass

    def debug_db_snapshot(self) -> None:
        """Print a small DB snapshot to quickly diagnose 'Loaded 0 events' scenarios."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT current_database() AS db, current_schema() AS schema")
                meta = cur.fetchone() or {}

                # These counts are intentionally lightweight and safe.
                cur.execute(f"SELECT COUNT(*) AS n FROM {self.schema}.final_events")
                n_events = (cur.fetchone() or {}).get("n", 0)
                cur.execute(f"SELECT COUNT(*) AS n FROM {self.schema}.final_categories")
                n_cats = (cur.fetchone() or {}).get("n", 0)

            print(f"[OpportunityMatcher] DB Snapshot: database={meta.get('db')} schema={meta.get('schema')} (configured schema={self.schema})")
            print(f"[OpportunityMatcher] DB Snapshot: final_events={n_events} rows | final_categories={n_cats} rows")
        except Exception as e:
            print(f"[OpportunityMatcher] DB Snapshot: unavailable ({e})")

    def fetch_events(self, limit: Optional[int] = None, days_back: int = 30) -> List[Event]:
        query = """
            SELECT
                event_id,
                title,
                summary_text,
                body_text,
                published_at,
                source,
                url
            FROM {schema}.final_events
            WHERE published_at >= NOW() - INTERVAL %s
            ORDER BY published_at DESC
        """
        query = query.format(schema=self.schema)
        if limit:
            query += f" LIMIT {limit}"

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (f"{days_back} days",))
            rows = cur.fetchall()

        events: List[Event] = []
        for r in rows:
            events.append(
                Event(
                    event_id=r["event_id"],
                    title=r.get("title"),
                    summary=r.get("summary_text"),
                    description=r.get("body_text"),
                    tags=[],
                    published_at=r["published_at"].isoformat() if r.get("published_at") else None,
                    source=r.get("source"),
                    url=r.get("url"),
                )
            )
        return events

    def get_categories(self) -> dict:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT category_name, category_text_id FROM {self.schema}.final_categories")
            rows = cur.fetchall()
        return {r["category_text_id"]: r["category_name"] for r in rows}

    def check_connection(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            self.conn.autocommit = False
            self._set_search_path()

    def save_opportunities(self, opportunities: List[Opportunity]) -> int:
        if not opportunities:
            return 0

        insert_columns = [
            "opportunity_id",
            "title",
            "summary",
            "reason",
            "who_to_contact",
            "suggested_outreach_angle",
            "categories",
            "filter_chain",
            "filter_sector",
            "filter_seeking",
            "time_found",
            "confidence",
            "tags",
            "target_company",
            "sources",
            "event_ids",
            "event_titles",
            "event_url",
            "bd_weeks",
            "opportunity_details",
        ]
        update_columns = [
            "title",
            "summary",
            "reason",
            "who_to_contact",
            "suggested_outreach_angle",
            "categories",
            "filter_chain",
            "filter_sector",
            "filter_seeking",
            "time_found",
            "confidence",
            "tags",
            "target_company",
            "sources",
            "event_ids",
            "event_titles",
            "event_url",
            "bd_weeks",
            "opportunity_details",
        ]
        placeholders = ",".join(["%s"] * len(insert_columns))
        update_set = ",\n                ".join(f"{col} = EXCLUDED.{col}" for col in update_columns)

        query = f"""
            INSERT INTO {self.schema}.opportunities_filter (
                {", ".join(insert_columns)}
            ) VALUES (
                {placeholders}
            )
            ON CONFLICT (opportunity_id) DO UPDATE SET
                {update_set}
        """

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self.check_connection()
                with self.conn.cursor() as cur:
                    for o in opportunities:
                        tag_values: List[str] = []
                        for tag in (o.tags or []):
                            tag_text = clean_text(tag)
                            # Keep user-facing tags clean: never persist internal audit marker tags.
                            if tag_text and not tag_text.lower().startswith("audit:") and tag_text not in tag_values:
                                tag_values.append(tag_text)
                        cur.execute(
                            query,
                            (
                                o.opportunity_id,
                                o.title,
                                o.summary,
                                o.reason,
                                o.who_to_contact,
                                o.suggested_outreach_angle,
                                ",".join([c for c in (o.categories or []) if str(c).strip()]),
                                ",".join([x for x in str(o.filter_chain or "").split(",") if x.strip()]),
                                ",".join([x for x in str(o.filter_sector or "").split(",") if x.strip()]),
                                ",".join([x for x in str(o.filter_seeking or "").split(",") if x.strip()]),    
                                o.time_found,
                                o.confidence,
                                tag_values,
                                o.target_company,
                                o.sources,
                                o.event_ids,
                                o.event_titles,
                                o.event_url,
                                o.bd_weeks,
                                o.opportunity_details,
                            ),
                        )
                self.conn.commit()
                return len(opportunities)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                print(f"[OpportunityMatcher] WARNING: DB error attempt {attempt}/{max_retries}: {e}")
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                    continue
                raise

        return 0

    def close(self):
        if self.conn:
            self.conn.close()


# ---------------------------
# OpenAI Schemas
# ---------------------------

def openai_batching_schema() -> Dict[str, Any]:
    return {
        "name": "event_batching",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "batches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "batch_id": {"type": "number"},
                            "event_ids": {"type": "array", "items": {"type": "string"}},
                            "relationship_reason": {"type": "string"},
                        },
                        "required": ["batch_id", "event_ids", "relationship_reason"],
                    },
                }
            },
            "required": ["batches"],
        },
        "strict": True,
    }


def openai_gatekeeper_schema() -> Dict[str, Any]:
    return {
        "name": "gatekeeper_results",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "event_id": {"type": "string"},
                            "hard_exclusion": {"type": "boolean"},
                            "hard_exclusion_reason": {"type": ["string", "null"]},
                            "evidence_type": {
                                "type": "string",
                                "enum": [
                                    "partnership_or_integration",
                                    "grant_or_rfp_or_program_open",
                                    "vendor_or_procurement_need",
                                    "token_launch_or_tge",
                                    "institutional_allocation_or_treasury",
                                    "regulatory_or_compliance_update",
                                    "support_added_with_bd_action",
                                    "none"
                                ],
                            },
                            "evidence_snippet": {"type": ["string", "null"]},
                            "supporting_snippet": {"type": ["string", "null"]},
                        },
                        "required": [
                            "event_id",
                            "hard_exclusion",
                            "hard_exclusion_reason",
                            "evidence_type",
                            "evidence_snippet",
                            "supporting_snippet",
                        ],
                    },
                }
            },
            "required": ["results"],
        },
        "strict": True,
    }


def openai_opportunity_schema_v2() -> Dict[str, Any]:
    """Client-required output schema + sub-scores for deterministic confidence."""
    return {
        "name": "opportunity_extraction_v2",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "opportunities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "is_opportunity": {"type": "boolean"},
                            "title": {"type": "string"},
                            "reason": {"type": "string"},
                            "evidence_type": {
                                "type": "string",
                                "enum": [
                                    "partnership_or_integration",
                                    "grant_or_rfp_or_program_open",
                                    "vendor_or_procurement_need",
                                    "token_launch_or_tge",
                                    "institutional_allocation_or_treasury",
                                    "regulatory_or_compliance_update",
                                    "support_added_with_bd_action"
                                ],
                            },
                            "evidence_snippet": {"type": "string"},
                            "supporting_snippet": {"type": ["string", "null"]},
                            "recommended_action": {"type": ["string", "null"]},
                            "who_to_contact": {"type": ["string", "null"]},
                            "target_company": {"type": ["string", "null"]},
                            "categories": {"type": "array", "items": {"type": "string"}},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "source_event_ids": {"type": "array", "items": {"type": "string"}},
                            "sub_scores": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "fit_score": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "evidence_score": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "actionability_score": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "feasibility_score": {"type": "integer", "minimum": 0, "maximum": 3},
                                },
                                "required": ["fit_score", "evidence_score", "actionability_score", "feasibility_score"],
                            },
                        },
                        "required": [
                            "is_opportunity",
                            "title",
                            "reason",
                            "evidence_type",
                            "evidence_snippet",
                            "supporting_snippet",
                            "recommended_action",
                            "who_to_contact",
                            "target_company",
                            "categories",
                            "tags",
                            "source_event_ids",
                            "sub_scores",
                        ],
                    },
                }
            },
            "required": ["opportunities"],
        },
        "strict": True,
    }


def openai_feedback_schema_v2() -> Dict[str, Any]:
    """Schema for Critic responses (response_format=json_schema).

    Compatibility notes:
    - process_events_batch expects: feedback["overall_rating"] and feedback["opportunity_feedback"][i]["index"].
    - The nested reframed_opportunity MUST stay schema-identical to openai_opportunity_schema_v2().items.
    """
    opp_items_schema = openai_opportunity_schema_v2().get("schema", {}).get("properties", {}).get("opportunities", {}).get("items", {})

    return {
        "name": "opportunity_feedback_v2",
        "schema": {
            "type": "object",
            "properties": {
                "overall_rating": {"type": "number"},
                "opportunity_feedback": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "index": {"type": "integer"},
                            "status": {"type": "string", "enum": ["keep", "reframe", "discard"]},
                            "audit_bucket": {
                                "type": ["string", "null"],
                                "enum": ["keep", "keep_but_reframe", "manual_review", "cut", None],
                            },
                            "reason_code": {"type": ["string", "null"]},
                            "feedback": {"type": "string"},
                            "corrected_sub_scores": {
                                "type": ["object", "null"],
                                "properties": {
                                    "fit_score": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "evidence_score": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "actionability_score": {"type": "integer", "minimum": 0, "maximum": 3},
                                    "feasibility_score": {"type": "integer", "minimum": 0, "maximum": 3},
                                },
                                "additionalProperties": False,
                            },
                            "reframed_opportunity": {
                                "anyOf": [
                                    opp_items_schema,
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["index", "status", "feedback", "reframed_opportunity"],
                    },
                }
            },
            "required": ["overall_rating", "opportunity_feedback"],
            "additionalProperties": False
        }
    }


def openai_keep_enrichment_schema_v1() -> Dict[str, Any]:
    return {
        "name": "keep_opportunity_enrichment_v1",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "reason": {"type": "string"},
                "suggested_outreach_angle": {"type": "string"},
                "opportunity_details": {"type": ["string", "null"]},
            },
            "required": ["title", "summary", "reason", "suggested_outreach_angle", "opportunity_details"],
        },
        "strict": True,
    }


def openai_recovery_schema_v1() -> Dict[str, Any]:
    opp_items_schema = openai_opportunity_schema_v2().get("schema", {}).get("properties", {}).get("opportunities", {}).get("items", {})
    return {
        "name": "dropped_item_recovery_v1",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "index": {"type": "integer"},
                            "recover": {"type": "boolean"},
                            "rationale": {"type": "string"},
                            "opportunity": {
                                "anyOf": [
                                    opp_items_schema,
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["index", "recover", "rationale", "opportunity"],
                    },
                }
            },
            "required": ["decisions"],
        },
        "strict": True,
    }

# ---------------------------
# Agent: Batcher
# ---------------------------

def batch_related_events(
    client: OpenAI,
    model: str,
    events: List[Event],
    max_batch_size: int,
    tracker: Optional[TokenTracker] = None,
) -> List[List[Event]]:
    if len(events) <= max_batch_size:
        return [events]

    event_summaries = [
        {
            "event_id": ev.event_id,
            "title": ev.title or "",
            "summary": ev.summary or "",
            "source": ev.source or "",
        }
        for ev in events
    ]

    system_prompt = """You group related events for downstream business-opportunity screening.

Group events together if they refer to the same company/project, the same initiative, the same product launch,
or different coverage of the same underlying development.

Rules:
- Maximum 40 events per batch.
- Every event must appear in exactly one batch.
- Prefer keeping related coverage together.
"""

    user_prompt = f"""EVENTS:
{json.dumps(event_summaries, separators=(",", ":"))}

Task: create intelligent batches (<=40 events each), returning event_ids and a brief relationship_reason."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": openai_batching_schema()},
            temperature=0.2,
            max_completion_tokens=2000,
        )
        if tracker:
            tracker.add_usage(resp.usage, model, agent="Batcher")

        raw = resp.choices[0].message.content
        if not raw or not raw.strip():
            raise ValueError("Empty batching response")

        parsed = json.loads(raw)
        event_map = {ev.event_id: ev for ev in events}
        # IMPORTANT: Never drop events due to an incomplete or malformed model response.
        # The LLM may omit some event_ids or repeat them across batches; we must enforce:
        # - every event appears in exactly one batch
        batches: List[List[Event]] = []
        assigned: set[str] = set()

        for b in parsed.get("batches", []):
            batch_ids = b.get("event_ids", []) or []
            batch_events: List[Event] = []
            for eid in batch_ids:
                if eid in event_map and eid not in assigned:
                    batch_events.append(event_map[eid])
                    assigned.add(eid)
            if batch_events:
                batches.append(batch_events)

        # Add any missing/unassigned events as simple chunks (preserves recall).
        missing_ids = [eid for eid in event_map.keys() if eid not in assigned]
        if missing_ids:
            for i in range(0, len(missing_ids), max_batch_size):
                chunk_ids = missing_ids[i : i + max_batch_size]
                batches.append([event_map[eid] for eid in chunk_ids])

        return batches or [events]

    except Exception as e:
        print(f"  WARNING: Batching failed ({e}), using simple chunking")
        return [events[i : i + max_batch_size] for i in range(0, len(events), max_batch_size)]


# ---------------------------
# Agent: Gatekeeper
# ---------------------------

def call_gatekeeper(
    client: OpenAI,
    model: str,
    events: List[Event],
    tracker: Optional[TokenTracker] = None,
) -> Dict[str, GateResult]:
    """Apply client gate criteria and extract evidence snippets/types per event."""
    audit_today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    payload_cfg = gatekeeper_payload_limits()
    short_horizon_mode = TIME_HORIZON_WEEKS <= 2

    # Adjust token limit based on batch size (minimum 1800, scale up for larger batches)
# Conservative scaling to prevent truncation
    tokens_per_event = 220      # was 100
    base_tokens = 1200          # was 800
    safety_buffer = 2200        # extra headroom for JSON structure

    max_tokens = base_tokens + (len(events) * tokens_per_event) + safety_buffer

    # Hard cap to prevent runaway cost/latency
    max_tokens = min(max_tokens, 4000)

    prefiltered_results: Dict[str, GateResult] = {}
    llm_events: List[Event] = []
    events_payload: List[Dict[str, Any]] = []
    for ev in events:
        gated_description = slice_body_for_gatekeeper(
            ev.description,
            head=payload_cfg["body_head"],
            tail=payload_cfg["body_tail"],
        )
        if gatekeeper_body_is_unusable(gated_description):
            prefiltered_results[ev.event_id] = GateResult(
                event_id=ev.event_id,
                hard_exclusion=True,
                hard_exclusion_reason="unusable_source_body",
                evidence_type="none",
                evidence_snippet=None,
                supporting_snippet=None,
            )
            continue

        llm_events.append(ev)
        events_payload.append(
            {
                "event_id": ev.event_id,
                "title": smart_truncate(clean_text(ev.title) or "", 200),
                "summary": smart_truncate(clean_text(ev.summary) or "", payload_cfg["summary_max"]),
                "description": gated_description,
                "published_at": ev.published_at or "",
                "bd_signal": truncate_bd_signal_for_gatekeeper(
                    ev.bd_signal or {},
                    max_str=payload_cfg["bd_max_str"],
                    max_items=payload_cfg["bd_max_items"],
                    max_json=payload_cfg["bd_max_json"],
                ),
                "source": ev.source or "",
                "url": ev.url or "",
            }
        )

    if not events_payload:
        return prefiltered_results

    system_prompt = f"""You are a compliance gatekeeper for business-opportunity detection.

Your job is binary: PASS or FAIL each event. You do not score quality. You only determine whether
a qualifying BD action surface exists and extract a verbatim snippet proving it.
Today is {audit_today}. Use this date when deciding whether a submission window, deadline, or action surface is already closed.

If a structured bd_signal field is provided in the event payload, use it as a weak prior only: it may help you find the likely action surface faster, but it NEVER overrides the event text.
bd_signal is advisory metadata, not evidence. Do not pass an event solely because bd_signal.has_action_surface=true. If bd_signal conflicts with the event text, trust the event text and fail closed.

HARD EXCLUSIONS (auto-fail):
- price moves, market recaps, opinion/editorial, macro narrative
- pure funding/investment/valuation news WITHOUT an external action surface
- institutional portfolio/ETF rebalancing commentary WITHOUT an external action surface
- applicant-side asks where the posting entity is seeking funding, approval, listing, or support for itself from the target ecosystem
  and the reader is neither the grantor nor the intended commercial counterparty
- applicant-side grant proposals, devgrant proposals, draft proposals, or sponsor-seeking posts for the proposer's own project
  are FAIL unless the source is clearly issued by the grantor, buyer, or named commercial counterparty running intake
- administrative or non-commercial participation flows such as emergency contact registration, generic beta tester signups,
  delegate races, committee applications, recognized delegate programs, contributor elections, hackathons, contests,
  or community programs aimed at individuals rather than BD buyers
- generic signup / self-serve product trial language such as "visit the site", "enter your email", "sign up", "try it", or "join the waitlist"
  unless the same text explicitly opens an enterprise, integration, partner, distributor, operator, auditor, or vendor path
- expired or closed surfaces: if the text says applications closed, submission period ended, deadline passed, or the action date is
  already in the past relative to {audit_today}, fail the event

SOFT EXCLUSIONS (usually fail unless explicit external intake is stated):
- governance-only forum threads, roadmap/status updates, research reports, monitoring dashboards
- announcements that are informational only (no partner/vendor/applicant action surface)
- standards, RFC, and build-phase feedback threads usually fail unless they also expose a concrete implementation,
  integration, funded build, partner, vendor, or operator path for external teams
- generic feedback, input-sought, topic-submission, proposal-discussion, or draft-ERC/forum-review posts usually fail unless
  the same source clearly asks external teams to implement, integrate, onboard, supply, audit, or contract now
- launch announcements, open-source releases, and roadmap signals such as "more chains coming" unless they explicitly invite
  external partners, integrators, vendors, or applicants through a named path

MINIMUM EVIDENCE TO PASS (must have >=1 and include a verbatim snippet):
- partnership_or_integration: explicit named partnership OR explicit integration (named parties, integration announced, partnership program).
- grant_or_rfp_or_program_open: explicit applications open / RFP/RFQ/RFI / call for proposals / grant program with external participants
- vendor_or_procurement_need: explicit tender/procurement/vendor search/RFP for vendors, or clear buying intent (not investment)
- token_launch_or_tge: explicit token launch / TGE / token generation event ONLY if the same text also exposes an external
  application, onboarding, partner, vendor, exchange, market maker, or integration path beyond mere launch awareness
- regulatory_or_compliance_update: explicit license/approval/registration/compliance change that creates an actionable BD motion
  (ONLY if there is an external action surface: application, partner onboarding, vendor intake, integration path, or contact channel)
- institutional_allocation_or_treasury: explicit corporate/treasury/institutional allocation decision that creates an actionable BD motion
  (ONLY if there is an external action surface: vendor selection, platform onboarding, integration, partner channel, RFP)
- support_added_with_bd_action: new support/feature added AND an explicit BD action surface (partners/integrators/vendors encouraged)

IMPORTANT DISAMBIGUATION (reduce 'grant' inflation):
- Venture funding rounds, fund closes, M&A rumors, "raises $X", "Series A/B", "launches fund" are NOT grants/RFP.
  These should be hard_exclusion=true unless the text includes an explicit external program/RFP/intake.

POSITIVE ALLOWLIST (these can pass even when the motion is implicit, but ONLY if the text itself clearly supports them):
- explicit request for audit, security review, implementation support, migration support, integration help, or deployment support
- explicit partner, distribution, operator, incubator, or co-build ask directed at external teams or companies
- named API, SDK, webhook, plugin, docs, or platform capability tied to a concrete business use case and clearly meant for
  external adoption; an explicit "contact us" is helpful but not required if the post is obviously aimed at integrators
- operator/provider admission or onboarding paths with a named next step
- standards or build-phase threads ONLY when they ask external teams to implement, integrate, or co-develop a concrete system now
- explicit request for external teams to evaluate, test, or integrate a public repo / SDK / tool / testnet when the source
  includes at least one of: a public repo link, a testnet/tooling trial, or an explicit contact path
- ongoing grants/programs may pass even if the source is a recap, but ONLY when the text clearly says the program remains live,
  ongoing, or still accepting contributors/applicants now
- named-target partnership or technical-alignment asks may pass when the source explicitly seeks a specific company, foundation,
  protocol, or operator for integration, audit, migration, deployment, pilot inclusion, or execution support; these may later
  classify as Inverted Outreach, but they are still valid opportunities for the consolidated review list

MATCHER-STAGE FLEXIBILITY:
- At matcher stage, do NOT require a formal buyer-side intake in every passing case.
- PASS when the source shows a concrete external BD path that is current and specific, even if it is early-stage or role-inverted, such as:
  - a named request for external teams to evaluate, test, pilot, or integrate now
  - a named design-partner or technical-feedback ask directed at relevant teams or companies
  - a live vendor/service intake, product evaluation path, or public onboarding path with docs, repo, testnet, form, or contact channel
  - an ongoing grant/program surface clearly shown as live now, even if evidenced through a current submission example
- These should PASS only when the source text supports a real next step now; they should NOT pass if the text is merely informational,
  speculative, agenda-only, community-participation-only, or generic discussion.

SUPPORTING SIGNALS ONLY (not sufficient on their own):
- "feedback welcome", "questions for the community", "join the chat / discuss", "testing requested", or an open-source release with docs
  are NOT by themselves enough to pass.
- Do NOT pass solely because the source exposes a public repo, registry listing, beta launch, submission form, newsletter/project-update form,
  referral code, or governance proposal.
- These only PASS when the same source explicitly asks relevant external teams or companies to integrate, onboard, evaluate, test, pilot,
  or integrate now, supply services, contract, apply to a live program now, or join a design-partner / technical-feedback process directed
  at relevant teams or companies.
- These can support a pass ONLY when the same text also shows a stronger current BD opening such as:
  explicit application/intake/RFP/grant window, vendor/audit/procurement need, partner/integrator/distributor/operator ask,
  live implementation path for outside teams, or a direct request to adopt, integrate, deploy, supply, or co-build now.
- Exception: a feedback/testing request MAY pass when the source explicitly invites external teams to evaluate or integrate now
  and includes a public repo link, a testnet/tooling trial, or an explicit contact path.
- Do NOT fail merely because a post includes docs, a repo, or a chat link; fail only when those are the main signal and no stronger
  action surface is present.

FAIL-CLOSED DEFAULTS:
- Self-introduction, monthly update, progress update, or complaint posts should fail unless they also contain a concrete current external intake.
- Newsletter exposure, announcement amplification, or "submit your updates" style visibility requests are not BD opportunities.
- Generic contact-only language ("reach out", "email us", "contact us after delegating", "DM for questions") is not enough without a
  specific partner/integration/vendor/application path in the same text.
- News/media/inference-led items about acquisitions, partnerships, hiring, expansion, or product direction should fail unless the source
  itself exposes a direct external action surface now.
- Applicant-side proposals and soft-signal posts should usually fail closed. Treat words like "proposal", "grant request",
  "application", "apply", "feedback", "discussion", "agenda", "claim profile", "open source", "contributions welcome",
  "bug bounty", or "meetup" as weak signals by default when they are the main evidence.
- Do not fail solely because an item is early-stage, on a forum, or framed as a proposal if the same text clearly shows an
  external implementation, evaluation, validator/operator, or technical-alignment path.
- Treat governance committee applications, recognized delegate programs, topic submissions, RFCs, draft ERCs,
  and generic proposal review threads as FAIL by default unless a buyer-side vendor, partner, or implementation intake is explicit.
- Do NOT pass posts merely because they contain those words, or because a repo/docs/forum/chat/profile/testnet is public.
  They may still PASS when the same text ALSO shows a concrete buyer-side path now, such as:
  a named intake/application window, vendor/audit/procurement need, direct request for partners/integrators/operators,
  urgent migration or upgrade ask, explicit onboarding flow, or a clear implementation/contact path tied to action now.
- Governance/RFC/open-source discussion posts should also fail closed when they mainly ask for comments, agenda topics, feedback,
  issues, forks, PRs, standards input, or general community contributions. Do NOT pass these merely because a repo, docs, testnet,
  plugin, SDK, or forum thread is public. They only pass when the source explicitly asks external teams to implement, integrate,
  onboard, supply, audit, migrate, or contract now through a concrete buyer-side path.

{"STRICT SHORT-HORIZON MODE (" + str(TIME_HORIZON_WEEKS) + " weeks):\n- Pass only if the text supports a realistic BD next step within about " + str(TIME_HORIZON_WEEKS) + " weeks.\n- Fail if the motion depends on future roadmap delivery, future funding, future governance approval, future hiring, or speculative expansion beyond this window.\n- For launches, SDKs, integrations, or partner announcements, require a clearly current external action surface now (applications, onboarding, implementation, procurement, partner intake, or a named contact path).\n- Fail recap-style, awareness-only, or announcement-only items unless the same text clearly invites an external team to apply, integrate, onboard, supply, or contact now.\n- Generic 'explore', 'engage', 'could partner', or 'positioning' language is NOT enough on its own; require a present-tense intake or implementation path in the source text.\n" if short_horizon_mode else ""}

ACTION-SURFACE SCAN (lightweight):
- If you do not find an action surface on first read, scan the body for verbs like:
  apply, open, program, grant, RFP/RFQ/RFI, partner, integrate, onboarding, procurement, vendor.
- Do NOT treat proposal/application/feedback/discussion words as sufficient by themselves; require the buyer-side path
  to be explicit in the same source text.

For each event:
- Set hard_exclusion=true if the event is primarily exclusion content.
- Choose evidence_type from the allowed enums; choose 'none' if no qualifying evidence.
- Provide evidence_snippet: one verbatim core quote (target about 20-38 words) from summary/description proving the action surface.
- Provide supporting_snippet: one second short verbatim line for context when available (about 12-30 words); otherwise null.
  If you cannot find a verbatim excerpt, set evidence_type='none' and both snippet fields to null.
- If the opportunity is ambiguous, role-inverted, or only weakly implied, fail closed with evidence_type='none'.

Output ONLY JSON per schema."""

    user_prompt = f"""EVENTS:
{json.dumps(events_payload, default=_json_safe, separators=(",", ":"))}

Return gatekeeper results for every event_id."""
    prompt_len = len(system_prompt) + len(user_prompt)
    payload_len = len(user_prompt)

    def _call_gatekeeper_with_prompts(sys_prompt: str, user_prompt: str, max_tokens: int) -> Any:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": openai_gatekeeper_schema()},
            temperature=0,
            max_completion_tokens=max_tokens,
        )

    def _gatekeeper_compact_prompt(base_prompt: str) -> str:
        return (
            base_prompt
            + "\n\nCOMPACTNESS RULES:\n"
            + "- evidence_snippet should be one strong verbatim quote, ideally 20-38 words.\n"
            + "- supporting_snippet should be one short verbatim context line, about 12-30 words when available.\n"
            + "- hard_exclusion_reason should be only a few words.\n"
            + "- Return the smallest valid JSON that satisfies the schema."
        )

    def _gatekeeper_output_is_unusually_long(resp: Any, raw: str, requested_max_tokens: int) -> bool:
        usage = getattr(resp, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        raw_len = len(raw) if raw else 0
        long_token_threshold = min(max(int(requested_max_tokens * 0.78), 2200), 3200)
        long_raw_threshold = max(9000, len(events) * 700)
        return completion_tokens >= long_token_threshold or raw_len >= long_raw_threshold

    def _run_gatekeeper_request(
        sys_prompt: str,
        user_prompt: str,
        requested_max_tokens: int,
    ) -> tuple[Any, str, int, int, int]:
        active_prompt = sys_prompt
        active_max_tokens = requested_max_tokens
        active_prompt_len = len(active_prompt) + len(user_prompt)

        resp = _call_gatekeeper_with_prompts(active_prompt, user_prompt, active_max_tokens)
        if tracker:
            tracker.add_usage(resp.usage, model, agent="Gatekeeper")
        raw = resp.choices[0].message.content or ""

        usage = getattr(resp, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        raw_len = len(raw)
        likely_truncated = bool(usage) and completion_tokens >= active_max_tokens * 0.95
        unusually_long = bool(usage) and _gatekeeper_output_is_unusually_long(resp, raw, active_max_tokens)

        if likely_truncated or unusually_long:
            reason = "likely truncated" if likely_truncated else "unusually long"
            compact_prompt = _gatekeeper_compact_prompt(sys_prompt)
            compact_max_tokens = min(requested_max_tokens, 2600)
            compact_prompt_len = len(compact_prompt) + len(user_prompt)
            print(
                f"  WARNING: Gatekeeper response {reason}; retrying compact "
                f"(completion_tokens={completion_tokens}, max_tokens={active_max_tokens}, raw_len={raw_len}, "
                f"batch_size={len(llm_events)}, prompt_len={active_prompt_len}, payload_len={payload_len})"
            )
            resp = _call_gatekeeper_with_prompts(compact_prompt, user_prompt, compact_max_tokens)
            if tracker:
                tracker.add_usage(resp.usage, model, agent="Gatekeeper")
            raw = resp.choices[0].message.content or ""
            active_prompt = compact_prompt
            active_max_tokens = compact_max_tokens
            active_prompt_len = compact_prompt_len

        return resp, raw, active_prompt_len, payload_len, active_max_tokens

    resp, raw, prompt_len, payload_len, max_tokens = _run_gatekeeper_request(system_prompt, user_prompt, max_tokens)

    # Check if the final response was truncated
    if resp.usage and hasattr(resp.usage, 'completion_tokens'):
        if resp.usage.completion_tokens >= max_tokens * 0.95:
            raw_len = len(raw) if raw else 0
            print(
                f"  WARNING: Gatekeeper response likely truncated "
                f"(completion_tokens={resp.usage.completion_tokens}, max_tokens={max_tokens}, raw_len={raw_len}, "
                f"batch_size={len(llm_events)}, prompt_len={prompt_len}, payload_len={payload_len})"
            )
            err = GatekeeperTruncatedError("Gatekeeper response likely truncated")
            err.completion_tokens = resp.usage.completion_tokens
            err.max_tokens = max_tokens
            err.raw_len = raw_len
            err.batch_size = len(events)
            err.prompt_len = prompt_len
            err.payload_len = payload_len
            raise err

    try:
        parsed = json.loads(raw)
    except Exception:
        err = GatekeeperParseError("Gatekeeper JSON parse failed")
        err.raw_len = len(raw) if raw else 0
        err.batch_size = len(llm_events)
        err.completion_tokens = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
        err.max_tokens = max_tokens
        err.prompt_len = prompt_len
        err.payload_len = payload_len
        raise err

    # TEMP: Log gatekeeper prompt/response sizes for the next few runs
    if resp.usage and hasattr(resp.usage, 'completion_tokens'):
        raw_len = len(raw) if raw else 0
        print(
            f"  [Gatekeeper] Sizes: "
            f"prompt_len={prompt_len}, payload_len={payload_len}, "
            f"completion_tokens={resp.usage.completion_tokens}, max_tokens={max_tokens}, "
            f"raw_len={raw_len}, batch_size={len(llm_events)}"
        )

    out: Dict[str, GateResult] = dict(prefiltered_results)
    for r in parsed.get("results", []):
        evidence_type = r.get("evidence_type") or "none"
        gr = GateResult(
            event_id=r["event_id"],
            hard_exclusion=bool(r["hard_exclusion"]),
            hard_exclusion_reason=normalize_hard_exclusion_reason(r.get("hard_exclusion_reason"), evidence_type),
            evidence_type=evidence_type,
            evidence_snippet=enforce_verbatim_snippet(r.get("evidence_snippet")),
            supporting_snippet=enforce_verbatim_snippet(r.get("supporting_snippet"), max_words=30),
        )
        out[gr.event_id] = gr

    # Lightweight second pass: only for evidence_type='none'
    none_ids = [eid for eid, gr in out.items() if gr.evidence_type == "none" and not gr.hard_exclusion]
    if none_ids:
        # Build minimal payload of only "none" events to re-scan for explicit intake language
        retry_events = [
            {
                "event_id": ev.event_id,
                "title": ev.title or "",
                "summary": ev.summary or "",
                "description": slice_body_for_gatekeeper(
                    ev.description,
                    head=payload_cfg["body_head"],
                    tail=payload_cfg["body_tail"],
                ),
                "published_at": ev.published_at or "",
                "source": ev.source or "",
                "url": ev.url or "",
            }
            for ev in llm_events
            if ev.event_id in none_ids
        ]
        retry_system = f"""You are a strict evidence finder.

Goal: for each event, find an EXPLICIT external intake/action surface if it exists.
Today is {audit_today}. Do not treat expired or closed deadlines as active opportunities.
Look for exact phrases indicating applications/open programs/RFPs/grants/partner intake/integration/vendor or procurement.
Do NOT treat generic sign-up / self-serve trial language as BD intake unless the text explicitly opens an enterprise,
integration, partner, distributor, operator, auditor, or vendor path.
Positive allowlist:
- audit / security review / implementation support / migration support / integration help requests
- explicit partner / distribution / operator / incubator / co-build asks
- named API / SDK / webhook / plugin / docs / platform capabilities tied to a business use case and clearly meant for
  external adopters, even without a formal intake email
- explicit request for external teams to evaluate, test, or integrate a public repo / SDK / tool / testnet when the source
  includes at least one of: a public repo link, a testnet/tooling trial, or an explicit contact path
- ongoing grants/programs or recap posts that explicitly say the program remains live/ongoing now
If no explicit intake language exists, return evidence_type='none'.
If you find qualifying evidence, evidence_snippet should be a strong verbatim quote (about 20-38 words),
and supporting_snippet should add one short context line when available.

Output ONLY JSON per schema."""
        retry_user = f"""EVENTS:
{json.dumps(retry_events, default=_json_safe, separators=(",", ":"))}

Return gatekeeper results for every event_id."""
        retry_max_tokens = min(max_tokens, 3600)
        retry_resp, retry_raw, _, _, _ = _run_gatekeeper_request(retry_system, retry_user, retry_max_tokens)
        try:
            retry_parsed = json.loads(retry_raw)
            for r in retry_parsed.get("results", []):
                if not r.get("event_id"):
                    continue
                evidence_type = r.get("evidence_type") or "none"
                if evidence_type == "none":
                    continue
                gr = GateResult(
                    event_id=r["event_id"],
                    hard_exclusion=bool(r.get("hard_exclusion", False)),
                    hard_exclusion_reason=normalize_hard_exclusion_reason(r.get("hard_exclusion_reason"), evidence_type),
                    evidence_type=evidence_type,
                    evidence_snippet=enforce_verbatim_snippet(r.get("evidence_snippet")),
                    supporting_snippet=enforce_verbatim_snippet(r.get("supporting_snippet"), max_words=30),
                )
                out[gr.event_id] = gr
        except Exception:
            # If retry fails, keep original results
            pass
    return out


# ---------------------------
# Agent: Finder (Opportunity Extraction)
# ---------------------------

def call_finder(
    client: OpenAI,
    model: str,
    gated_events: List[Tuple[Event, GateResult]],
    categories_map: dict,
    category_catalog: List[Dict[str, str]],
    max_output_tokens: int,
    tracker: Optional[TokenTracker] = None,
) -> List[Dict[str, Any]]:
    """Generate compliant opportunities ONLY from events that already pass Gatekeeper."""
    audit_today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    events_payload = []
    for ev, gr in gated_events:
        events_payload.append(
            {
                **ev.to_dict(),
                "gate": {
                    "evidence_type": gr.evidence_type,
                    "evidence_snippet": gr.evidence_snippet,
                    "supporting_snippet": gr.supporting_snippet,
                },
            }
        )

    system_prompt = f"""You are a business development analyst generating *client-gate compliant* opportunities.
Today is {audit_today}. Use that exact date when deciding if a deadline or submission window is already expired.

NON-NEGOTIABLE RULES:
1) You are ONLY given events that already passed Gatekeeper.
2) Your output MUST follow the required schema fields.
3) evidence_snippet MUST be verbatim (use the provided gate.evidence_snippet as the core proof quote; do not invent).
   - Keep evidence_snippet at about 20-38 words so it preserves the actual BD meaning.
   - supporting_snippet should be a second short verbatim context line (about 12-30 words) when available.
   - Prefer evidence_snippet + supporting_snippet together over a tiny fragment that loses the action surface.
4) evidence_type MUST match gate.evidence_type for the supporting event(s); do not invent new evidence types.
5) recommended_action is ONLY allowed when is_opportunity=true and must be grounded in evidence_snippet and supporting_snippet when present.
6) NO hard-exclusion content. If you suspect an event is actually exclusion content, set is_opportunity=false.
7) Categories MUST align with the provided definitions; do not guess.
8) If the action surface is weak/implicit (e.g., roadmap/status update, governance-only thread, monitoring report),
   set is_opportunity=false unless the event text explicitly invites external partners/vendors/applicants.
9) If the posting entity is seeking funding, approval, listing, or support for itself from the target ecosystem,
   and our reader is not the grantor or intended commercial counterparty, set is_opportunity=false.
9a) Grant/proposal posts may only PASS when the text explicitly invites external partners/integrators to adopt,
    integrate, or operationalize a live product AND you can quote that ask in evidence_snippet.
    Require at least one of: (a) explicit partner/integration/adoption invitation,
    (b) public repo or live product + explicit "integrate/adopt/build with" language,
    (c) concrete contact path for partners/integrators.
    Do NOT treat grant amount/category as evidence of opportunity.
10) Do NOT treat hackathons, contests, delegate races, contributor programs, emergency/admin registration,
    or community participation flows as BD opportunities unless the text exposes a clear commercial buyer or vendor intake.
11) Do NOT treat token launches, open-source releases, or product launches as opportunities unless the text explicitly
    invites external partners, integrators, vendors, applicants, or named counterparties to act now.
12) Do NOT treat generic signup, self-serve product trial, waitlist, or "visit the site and try it" language as procurement
    or BD intake unless the same text explicitly opens an enterprise, integration, partner, distributor, operator, auditor,
    or vendor path.
13) If the source says applications closed, submission period ended, deadline passed, or the action date is already in the past
    relative to {audit_today}, set is_opportunity=false.
14) If the case is plausible but too ambiguous to publish safely, fail closed here so it can fall into manual review /
    near-miss handling rather than being overstated as a final opportunity.
15) POSITIVE ALLOWLIST: do keep real-but-implicit BD surfaces when the text explicitly shows one of these and the realistic actor
    is a company:
    - audit / security review / implementation support / migration support / integration help requests
    - partner / distribution / operator / incubator / co-build asks
    - named API / SDK / webhook / plugin / docs / platform adoption paths tied to a concrete business use case and clearly
      intended for external integrators or business adopters, even if there is no formal intake email
    - operator/provider admission or onboarding paths with a named next step
    - standards/build threads ONLY when they ask external teams to implement, integrate, or co-develop a concrete system now
    - ongoing grants/programs when the text clearly says the program is still live, ongoing, or accepting contributors/applicants now
16) STRONG MISSED PATTERN RESCUE: when one of the following appears, bias toward is_opportunity=true unless the item is clearly
    closed, purely volunteer/community, or governance-only:
    - a live SDK / API / webhook / plugin / infra release whose whole point is external product integration or enterprise adoption
    - an explicit production gap such as "unaudited", "not production-ready", security review, remediation, or implementation need
    - a company positioning itself for distribution / operator / channel / institutional partner expansion in a concrete market
    - a public proposal under review where the realistic BD motion is to approach the proposing team as an audit / infra /
      implementation / integration partner, rather than to lobby the grantor

bd_signal guidance:
- If bd_signal is present in the event payload, treat it as weak metadata only.
- It can help you locate the action surface faster, but it must never override evidence_snippet or supporting_snippet.

Additional context guidance:
- why_it_matters, recommended_action, and contact_leads may be present from preprocess AI enrichment.
- Treat these as weak metadata only. They can help phrasing, but they must never override evidence_snippet, supporting_snippet, or gate evidence.

De-duplication:
- Avoid producing multiple opportunities for the same target_company + same action.

Sub-scores (0-3 each):
- fit_score: commercial fit (generic BD fit is ok, but be conservative)
- evidence_score: how explicit the evidence is (3=explicit named partnership/integration/RFP)
- actionability_score: can you do a concrete next step within {TIME_HORIZON_STR}?
  (If action is likely >{TIME_HORIZON_WEEKS} weeks, score this low.)
- feasibility_score: realistic access and timeline

Do NOT output confidence directly; output sub_scores and the caller will compute confidence deterministically.
"""

    user_prompt = f"""GATED EVENTS (each includes gate evidence_type + evidence_snippet + supporting_snippet):
{json.dumps(events_payload, default=_json_safe, separators=(",", ":"))}

AVAILABLE CATEGORIES (use only category_text_id values):
{json.dumps(category_catalog, default=_json_safe, separators=(",", ":"))}

TASK:
For each distinct business opportunity you can justify, output an object with:
- is_opportunity
- reason (1-3 sentences)
- evidence_snippet (verbatim)
- supporting_snippet (verbatim context line when useful, else null)
- recommended_action (only if TRUE)
- target_company, who_to_contact (if you can infer), categories/tags
- source_event_ids (must be exact event_id values above)
- sub_scores (0-3 each)

Quality bar:
- It is acceptable to return zero opportunities.
- Prefer 2-5 high-quality items per batch (still grounded in evidence).
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": openai_opportunity_schema_v2()},
        temperature=0.0,
        top_p=1.0,
        max_completion_tokens=max_output_tokens,
    )

    if tracker:
        tracker.add_usage(resp.usage, model, agent="Finder")

    raw = resp.choices[0].message.content
    parsed = json.loads(raw)
    return parsed.get("opportunities", [])


# ---------------------------
# Agent: Critic
# ---------------------------

def call_critic(
    client: OpenAI,
    model: str,
    opportunities: List[Dict[str, Any]],
    events_by_id: Optional[Dict[str, Event]] = None,
    tracker: Optional[TokenTracker] = None,
) -> Dict[str, Any]:
    """Compliance-first critic.

    Hardening:
    - Reject empty / whitespace responses before JSON parsing.
    - Retry once at temperature=0 (same model) to mitigate transient/truncation issues.
    - Fallback to o3-mini if primary critic fails.
    - Final fallback returns a safe neutral rating with no item-level feedback.
    """

    # Provide critic only the fields relevant to compliance and grounding
    keys = [
        "is_opportunity",
        "title",
        "summary",
        "reason",
        "evidence_type",
        "evidence_snippet",
        "supporting_snippet",
        "recommended_action",
        "suggested_outreach_angle",
        "target_company",
        "who_to_contact",
        "categories",
        "confidence",
        "source_event_ids",
        "sub_scores",
    ]
    short = []
    for o in opportunities:
        row = {k: o.get(k) for k in keys}
        source_bd_signals = []
        for eid in o.get("source_event_ids", []) or []:
            ev = events_by_id.get(str(eid)) if events_by_id else None
            if ev and isinstance(ev.bd_signal, dict) and ev.bd_signal:
                source_bd_signals.append({"event_id": str(eid), "bd_signal": ev.bd_signal})
        row["source_bd_signals"] = source_bd_signals
        short.append(row)

    audit_today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    system_prompt = f"""You are a compliance critic for business opportunities.
Today is {audit_today}. Use that exact date when deciding whether deadlines or submission windows are already closed.

Moderate audit alignment:
- Think like a commercial BD auditor using the publication guideline, but do NOT be maximally harsh.
- `keep` should correspond to guideline `Keep`.
- `reframe` should correspond to guideline `Keep but Reframe`, and may also be used for borderline `Manual Review` cases that still have a real company-relevant BD surface.
- `discard` should correspond to guideline `Cut`.
- When an item is commercially plausible and company-relevant but the wording is soft, broad, or slightly speculative, prefer REFRAME over DISCARD.
- Only use DISCARD when the item is genuinely non-publishable under the guideline even after a reasonable reframe.

Your job is to enforce the client rules while preserving plausible opportunities:
- If the item looks like hard exclusion content (price moves / macro / opinion / market recap) -> DISCARD.
- If is_opportunity=true but the minimal evidence type is not genuinely present -> DISCARD.
- If the event is governance-only, roadmap/status, monitoring/reporting, or informational with no explicit external intake -> prefer DISCARD.
- If the posting entity is seeking funding, approval, listing, or support for itself from the target ecosystem,
  and our reader is neither the grantor nor the intended commercial counterparty -> DISCARD.
- If the item is an administrative or non-commercial participation flow (emergency contact registration, delegate race,
  generic contributor intake, hackathon, contest, or community sign-up) with no clear BD buyer motion -> DISCARD.
- If the realistic actor is an individual creator, governance hobbyist, volunteer participant, or generic researcher
  rather than a commercially relevant Web3 company -> DISCARD.
- If the item is a token launch, product launch, or open-source release with no explicit external intake or counterparty motion -> DISCARD.
- If the item is generic signup, self-serve product trial, waitlist, or "visit the site and try it" language with no explicit
  enterprise, integration, partner, distributor, operator, auditor, or vendor path -> DISCARD.
- If the source says applications closed, submission period ended, deadline passed, or the action date is already in the past
  relative to {audit_today} -> DISCARD.
- If the opportunity is ambiguous enough that a human would need to tighten the framing before publishing, prefer REFRAME, not DISCARD.
- Use DISCARD only for hard exclusions, obviously closed opportunities, wrong-role opportunities, or clear non-commercial junk.
- Guideline-style cuts that should usually DISCARD:
  media recap / PR recap with no external intake;
  governance-only, admin-only, or community participation items;
  generic signup / self-serve / waitlist items;
  applicant-side asks where our reader is not the grantor or intended counterparty;
  inference-only BD angles that are not explicit in the source.
- POSITIVE ALLOWLIST: do not auto-discard if the text clearly shows one of these company-relevant motions:
  audit / security review / implementation support / migration support / integration help requests;
  partner / distribution / operator / incubator / co-build asks;
  named API / SDK / webhook / plugin / docs / platform adoption paths tied to a concrete business use case and clearly intended
  for external integrators or business adopters, even without a formal intake email;
  operator/provider admission or onboarding paths;
  ongoing grants/programs when the text clearly says the program remains live/ongoing now;
  standards/build threads that explicitly ask external teams to implement, integrate, or co-develop a concrete system now.
- STRONG MISSED PATTERN RESCUE: do not discard solely for lack of formal intake wording when the source clearly shows:
  a live SDK / API / webhook / plugin / infra release meant for external integration or enterprise adoption;
  an explicit production gap such as "unaudited", "not production-ready", security review, remediation, or implementation need;
  a concrete distribution / operator / channel / institutional partner expansion motion;
  or a public proposal under review where the real BD motion is to approach the proposing team as an audit / infra /
  implementation / integration partner rather than to lobby the grantor.
 - When an item matches those strong missed patterns and the action is commercially legible, prefer KEEP or REFRAME over DISCARD.
 - In particular, do not discard a strong candidate merely because the counterparty path is implicit if the source is plainly inviting
   external adopters, integrators, auditors, operators, distributors, or infrastructure partners.

bd_signal is advisory; first priority is evidence_snippet, then supporting_snippet, then bd_signal.

For each opportunity:
- status: keep | reframe | discard
- audit_bucket: keep | keep_but_reframe | manual_review | cut
- reason_code: short reason such as explicit_intake, integration_surface, audit_surface, generic_signup, governance_only, admin_only,
  wrong_role, inference_only, media_recap_no_intake, closed_expired, launch_no_counterparty, or ambiguous_but_salvageable
- feedback: short, specific
- corrected_sub_scores: if sub-scores are inflated/deflated
- reframed_opportunity: if status==reframe, provide a fully corrected version that follows schema.

Default posture:
- KEEP or REFRAME plausible opportunities.
- DISCARD only when a hard exclusion clearly applies.
- If the item is commercially legible but somewhat soft, implicit, or early-stage, prefer REFRAME over DISCARD.

REALISM / REACHABILITY HARD RULE (PILOT-CRITICAL):
- If you cannot plausibly imagine this opportunity receiving a reply from a
  partnerships / business development inbox within about {TIME_HORIZON_WEEKS} weeks -> DISCARD.

TIME HORIZON (NON-NEGOTIABLE):
- The opportunity must be actionable within {TIME_HORIZON_STR}.
- If the best realistic next step is likely >{TIME_HORIZON_WEEKS} weeks away, or timing is speculative/unclear -> DISCARD.

This includes (must discard):
- Pitching mega-institutions or regulators without an explicit partner intake,
  program application, RFP/procurement, or named integration already announced.
- "Strategic" or "thought leadership" angles with no external action surface.
- PR announcements that do not expose a concrete integration, vendor, or partner motion.
- SECONDARY INFERENCE opportunities: items inferred from a deal, expansion, funding,
  or partnership between other parties that do NOT themselves expose a new external
  action surface for the target company (e.g., "Company A expands after AMD deal").
- APPLICANT-SIDE opportunities where the source is mainly a team asking a foundation / DAO / ecosystem to fund or approve
  its own roadmap, and the proposed outreach casts our reader in the wrong role,
  unless the text explicitly opens a sponsor / implementation / audit / infrastructure partner slot.

FINAL GROUNDING REQUIREMENT (NON-NEGOTIABLE):
- The recommended_action MUST be explicitly justified by a specific phrase or sentence
  in the evidence_snippet and supporting_snippet when present. If you cannot point to the exact
  words in the provided evidence that justify the outreach -> DISCARD.

SOFT QUALITY CHECK (do not overuse):
- If the evidence_snippet lacks classic action words, do NOT discard solely for that reason.
- supporting_snippet may carry the contextual business meaning that makes the opportunity legible.
- For softer but commercially legible items, prefer REFRAME when the overall source still points to a real external
  integration, audit, operator, distribution, SDK/API adoption, or implementation surface.

This rule OVERRIDES phrasing quality, novelty, confidence, and sub-scores.

IMPORTANT: For each item in opportunity_feedback, include the integer field 'index' corresponding to the position of the input opportunity in the provided list.
"""

    user_prompt = f"""OPPORTUNITIES:
{json.dumps(short, separators=(",", ":"))}

Return your evaluation per schema."""

    def _attempt(m: str, attempt_label: str) -> Dict[str, Any]:
        # NOTE: Some reasoning models (e.g., o3 / o3-mini) do not support temperature control.
        # For maximum compatibility, only pass temperature when supported.
        create_kwargs = {
            "model": m,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_schema", "json_schema": openai_feedback_schema_v2()},
            "max_completion_tokens": 2400,
        }
        if m not in {"o3", "o3-mini"}:
            create_kwargs["temperature"] = 0.0

        resp = client.chat.completions.create(**create_kwargs)
        if tracker:
            tracker.add_usage(resp.usage, m, agent="Critic")

        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            raise ValueError(f"Empty critic response ({attempt_label})")
        return json.loads(raw)

    # Primary model: try twice (often resolves transient empty/truncated responses)
    try:
        return _attempt(model, f"{model}/try1")
    except Exception as e1:
        print(f"  [Critic] WARNING: Critic failed: {e1}")
        try:
            return _attempt(model, f"{model}/try2")
        except Exception as e2:
            print(f"  [Critic] WARNING: Critic failed (retry): {e2}")

    # Fallback: o3-mini (only if not already using it)
    if model != "o3-mini":
        try:
            return _attempt("o3-mini", "o3-mini/try1")
        except Exception as e3:
            print(f"  [Critic] WARNING: Critic failed (o3-mini): {e3}")

    # Fallback: gpt-4o-mini (cheap + stable) if o3/o3-mini failed
    if model != "gpt-4o-mini":
        try:
            return _attempt("gpt-4o-mini", "gpt-4o-mini/try1")
        except Exception as e4:
            print(f"  [Critic] WARNING: Critic failed (gpt-4o-mini): {e4}")

    return {"overall_rating": 5, "overall_feedback": "Critic failed", "opportunity_feedback": []}


# ---------------------------
# Agent: Refiner

# ---------------------------

def call_refiner(
    client: OpenAI,
    model: str,
    gated_events: List[Tuple[Event, GateResult]],
    original_opps: List[Dict[str, Any]],
    critic_feedback: Dict[str, Any],
    categories_map: dict,
    category_catalog: List[Dict[str, str]],
    max_output_tokens: int,
    tracker: Optional[TokenTracker] = None,
) -> List[Dict[str, Any]]:
    """Regenerate a compliant set using critic feedback; may drop weak items."""

    events_payload = []
    for ev, gr in gated_events:
        events_payload.append(
            {
                **ev.to_dict(),
                "gate": {
                    "evidence_type": gr.evidence_type,
                    "evidence_snippet": gr.evidence_snippet,
                    "supporting_snippet": gr.supporting_snippet,
                },
            }
        )

    audit_today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    system_prompt = f"""You are the REFINER agent. Produce a final, client-gate compliant opportunity set.
Today is {audit_today}. Use that exact date when deciding whether deadlines or submission windows are already closed.

NON-NEGOTIABLE CONSTRAINTS (prevents inference regression):
1) You may ONLY operate on the ORIGINAL OPPORTUNITIES provided.
   - You can DISCARD or REFRAME them, but DO NOT introduce new opportunities.
2) You must NOT add new companies, new partnerships, new products, or new claims.
   If a claim is not supported by the provided evidence_snippet/supporting_snippet, remove it.
3) evidence_snippet MUST be verbatim and MUST be taken exactly from gate evidence.
   supporting_snippet should preserve one short supporting line from gate evidence when useful.
4) recommended_action is ONLY allowed when is_opportunity=true AND must be explicitly justified by the exact words in evidence_snippet/supporting_snippet.
5) TIME HORIZON: actionable within {TIME_HORIZON_STR} only.
6) Prefer DISCARD over weak REFRAME.
7) If the posting entity is seeking funding, approval, listing, or support for itself from the target ecosystem,
   and our reader is not the grantor or intended commercial counterparty, DISCARD.
8) Do not preserve hackathons, contests, delegate races, contributor programs, emergency/admin registration,
   or other non-commercial participation flows unless a clear BD buyer motion is explicitly present.
9) Do not preserve token launches, product launches, or open-source releases unless the text explicitly exposes
   a real external intake path, partner motion, vendor motion, or named commercial counterparty.
10) Do not preserve generic signup, self-serve product trial, waitlist, or "visit the site and try it" language unless the
    same text explicitly opens an enterprise, integration, partner, distributor, operator, auditor, or vendor path.
11) If the source says applications closed, submission period ended, deadline passed, or the action date is already in the
    past relative to {audit_today}, DISCARD.
12) If the item is plausible but still ambiguous after critic feedback, DISCARD rather than publishing;
    these belong in manual review / near-miss handling, not the final opportunity set.
13) POSITIVE ALLOWLIST: preserve real-but-implicit BD surfaces when the text explicitly shows one of these and the realistic
    actor is a company:
    - audit / security review / implementation support / migration support / integration help requests
    - partner / distribution / operator / incubator / co-build asks
    - named API / SDK / webhook / plugin / docs / platform adoption paths tied to a concrete business use case and clearly
      intended for external integrators or business adopters, even if there is no formal intake email
    - operator/provider admission or onboarding paths with a named next step
    - standards/build threads ONLY when they ask external teams to implement, integrate, or co-develop a concrete system now
    - ongoing grants/programs when the text clearly says the program is still live, ongoing, or accepting contributors/applicants now

bd_signal guidance:
- If bd_signal is present in the gated event payload, use it only as supporting metadata when choosing between DISCARD and REFRAME.
- It must not add new facts, and it must never override evidence_snippet/supporting_snippet.

Additional context guidance:
- why_it_matters, recommended_action, and contact_leads may be present from preprocess AI enrichment.
- Treat these as weak metadata only. They can help phrasing, but they must never override evidence_snippet, supporting_snippet, or gate evidence.

Output only JSON per schema."""

    user_prompt = f"""GATED EVENTS:
{json.dumps(events_payload, default=_json_safe, separators=(",", ":"))}

ORIGINAL OPPORTUNITIES:
{json.dumps(original_opps, separators=(",", ":"))}

CRITIC FEEDBACK:
{json.dumps(critic_feedback, separators=(",", ":"))}

AVAILABLE CATEGORIES (use only category_text_id values):
{json.dumps(category_catalog, separators=(",", ":"))}

TASK:
Return a refined list of compliant opportunities (2-5 typical; 0 is acceptable)."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": openai_opportunity_schema_v2()},
        temperature=0,
        max_completion_tokens=max_output_tokens,
    )

    if tracker:
        tracker.add_usage(resp.usage, model, agent="Refiner")

    raw = resp.choices[0].message.content
    parsed = json.loads(raw)
    return parsed.get("opportunities", [])

def build_event_evidence_package(ev: Event, evidence_snippet: str, evidence_type: str = "") -> Dict[str, Any]:
    clean_title = clean_text(ev.title) or ""
    clean_summary = smart_truncate(clean_text(ev.summary) or "", 700)
    body_excerpt = smart_truncate(clean_text(ev.description) or "", 900)
    action_surface = smart_truncate(clean_text(evidence_snippet) or "", 420)
    source_domain = ""
    try:
        if ev.url:
            source_domain = (urlparse(ev.url).netloc or "").lower()
    except Exception:
        source_domain = ""
    return {
        "clean_title": clean_title,
        "clean_summary": clean_summary,
        "strong_body_excerpt": body_excerpt,
        "explicit_action_surface": action_surface,
        "source_context": {
            "source": clean_text(ev.source) or "",
            "source_domain": source_domain,
            "published_at": ev.published_at or "",
            "evidence_type": evidence_type or "",
        },
    }


def build_enrichment_event_context(ev: Event, evidence_snippet: str) -> Dict[str, Any]:
    body = (ev.description or "").strip()
    if body:
        if len(body) <= 5500:
            body_context = body
        else:
            head = body[:4000].rstrip()
            tail = body[-1500:].lstrip()
            body_context = f"{head}\n...\n{tail}"
    else:
        fallback_parts = [
            (ev.summary or "").strip(),
            (evidence_snippet or "").strip(),
            (ev.title or "").strip(),
        ]
        body_context = "\n".join([p for p in fallback_parts if p])

    return {
        "event_id": ev.event_id,
        "title": ev.title or "",
        "summary": ev.summary or "",
        "body_context": body_context,
        "evidence_package": build_event_evidence_package(ev, evidence_snippet),
        "source": ev.source or "",
        "url": ev.url or "",
    }


def build_recovery_reference_examples(opportunities: List[Opportunity], max_items: int = 6) -> List[Dict[str, Any]]:
    ranked = sorted(opportunities, key=lambda o: (-float(o.confidence or 0.0), o.title or ""))
    examples: List[Dict[str, Any]] = []
    for opp in ranked[:max_items]:
        examples.append(
            {
                "title": opp.title,
                "target_company": opp.target_company,
                "categories": opp.categories,
                "confidence": float(opp.confidence or 0.0),
                "summary": opp.summary,
                "evidence_type": extract_evidence_type_from_reason(opp.reason) or "",
                "evidence_snippet": extract_evidence_snippet_from_reason(opp.reason) or "",
            }
        )
    return examples


def watchlist_promotion_priority(opp: Opportunity) -> float:
    ev_type = extract_evidence_type_from_reason(opp.reason) or ""
    haystack = " ".join(
        [
            opp.title or "",
            opp.summary or "",
            opp.event_titles or "",
            extract_reason_text_from_reason_block(opp.reason) or "",
            extract_evidence_snippet_from_reason(opp.reason) or "",
            opp.suggested_outreach_angle or "",
            opp.target_company or "",
        ]
    )

    score = float(opp.confidence or 0.0)
    if ev_type == "vendor_or_procurement_need":
        score += 3.5
    elif ev_type in {"partnership_or_integration", "support_added_with_bd_action"}:
        score += 2.25
    elif ev_type == "grant_or_rfp_or_program_open":
        score -= 1.0

    if _WATCHLIST_PROMOTION_COMMERCIAL_RE.search(haystack):
        score += 3.0
    if _WATCHLIST_PROMOTION_DEPRIORITIZE_RE.search(haystack):
        score -= 3.5
    if _WATCHLIST_PROMOTION_PUBLIC_RE.search(haystack):
        score += 1.75
    if _WATCHLIST_PROMOTION_INVERTED_RE.search(haystack):
        score -= 2.25

    if "grant" in [normalize_key(c) for c in (opp.categories or [])]:
        score -= 0.75

    return score


def build_watchlist_promotion_reference_examples(opportunities: List[Opportunity], max_items: int = 6) -> List[Dict[str, Any]]:
    preferred = [
        opp for opp in opportunities
        if not _FINALIZER_FILECOIN_PROPOSAL_WATCH_RE.search(
            " ".join(
                [
                    opp.title or "",
                    opp.summary or "",
                    opp.event_titles or "",
                    extract_reason_text_from_reason_block(opp.reason) or "",
                    extract_evidence_snippet_from_reason(opp.reason) or "",
                ]
            )
        )
        and not _FINALIZER_APPLICANT_SIDE_WATCH_RE.search(
            " ".join(
                [
                    opp.title or "",
                    opp.summary or "",
                    opp.event_titles or "",
                    extract_reason_text_from_reason_block(opp.reason) or "",
                ]
            )
        )
    ]
    ranked = sorted(preferred or opportunities, key=lambda o: (-watchlist_promotion_priority(o), -float(o.confidence or 0.0), o.title or ""))
    return build_recovery_reference_examples(ranked[:max_items], max_items=max_items)


def weakest_keep_promotion_score(keep_opps: List[Opportunity]) -> float:
    if not keep_opps:
        return 0.0
    return min(watchlist_promotion_priority(opp) for opp in keep_opps)


def select_enrichment_source_events(opportunity: Opportunity, event_lookup: Dict[str, Event]) -> List[Event]:
    event_ids = [eid.strip() for eid in (opportunity.event_ids or "").split("|") if eid.strip()]
    evidence_snippet = extract_evidence_snippet_from_reason(opportunity.reason) or ""
    candidates: List[Tuple[int, int, int, Event]] = []

    for order_idx, eid in enumerate(event_ids):
        ev = event_lookup.get(eid)
        if not ev:
            continue
        haystack = " ".join([
            ev.title or "",
            ev.summary or "",
            ev.description or "",
        ])
        snippet_match = 1 if evidence_snippet and evidence_snippet in haystack else 0
        body_len = len((ev.description or "").strip())
        candidates.append((-snippet_match, -body_len, order_idx, ev))

    if not candidates:
        return []

    ordered = [item[3] for item in sorted(candidates)]
    chosen: List[Event] = []
    seen_signatures: Set[str] = set()

    for ev in ordered:
        signature = normalize_key((ev.title or "") + "|" + (ev.url or ""))
        context = build_enrichment_event_context(ev, evidence_snippet).get("body_context", "")
        if not context:
            continue
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        chosen.append(ev)
        if len(chosen) >= 2:
            break

    return chosen


_ENRICHMENT_SOFT_SIGNAL_RE = re.compile(
    r"\b("
    r"feedback welcome|questions? for the community|share your feedback|share your thoughts|"
    r"let us know|join (?:the )?(?:chat|discussion)|discuss|discussion|"
    r"looking for input|looking for ideas|if you spot any issue|if you have ideas|"
    r"testing requested|try it out|open[ -]source release|repository is now public|repo is now public"
    r")\b",
    re.IGNORECASE,
)

_ENRICHMENT_STRONG_ACTION_RE = re.compile(
    r"\b("
    r"apply|applications? open|submissions? open|reply to this thread|submit your|"
    r"rfp|rfq|rfi|grant(?:s)? program|onboarding teams|if your team is interested|"
    r"contact us privately to discuss terms|contact .* to discuss|"
    r"integrators? can|developers? can|you can (?:integrat|deploy|configure|set up|swap|use|build)|"
    r"operator onboarding|vendor|procurement|audit|security review|"
    r"partner intake|onboard(?:ing)?|co-build|implementation support|migration support"
    r")\b",
    re.IGNORECASE,
)

_ENRICHMENT_OVERCLAIM_RE = re.compile(
    r"\b("
    r"partnership opportunity|vendor opportunity|procurement|commercial opportunity|"
    r"apply now|application|onboarding|partner(?:ship)?|vendor|co-build|distribution|"
    r"collaboration opportunity|integration opportunity|engage .* for partnership|"
    r"open bd channel|actionable opportunity|clear opportunity"
    r")\b",
    re.IGNORECASE,
)


def enrichment_source_haystack(opportunity: Opportunity, source_events: Optional[List[Event]] = None) -> str:
    parts: List[str] = [
        clean_text(opportunity.title) or "",
        clean_text(opportunity.summary) or "",
        clean_text(opportunity.reason) or "",
        clean_text(opportunity.suggested_outreach_angle) or "",
    ]
    for ev in source_events or []:
        parts.extend(
            [
                clean_text(ev.title) or "",
                clean_text(ev.summary) or "",
                smart_truncate(clean_text(ev.description) or "", 2200),
            ]
        )
    return " ".join(p for p in parts if p).strip()


def enrichment_needs_soft_framing(opportunity: Opportunity, source_events: Optional[List[Event]] = None) -> bool:
    haystack = enrichment_source_haystack(opportunity, source_events)
    return bool(_ENRICHMENT_SOFT_SIGNAL_RE.search(haystack)) and not bool(
        _ENRICHMENT_STRONG_ACTION_RE.search(haystack)
    )


def validate_enriched_keep_output(
    original: Opportunity,
    candidate: Dict[str, Any],
    source_events: Optional[List[Event]] = None,
) -> bool:
    title = normalize_title_text(candidate.get("title"))
    summary = normalize_summary_text(
        candidate.get("summary"),
        [
            candidate.get("reason"),
            candidate.get("suggested_outreach_angle"),
            original.reason,
            original.summary,
        ],
    )
    reason = normalize_reason_text(candidate.get("reason"))
    outreach = normalize_outreach_text(candidate.get("suggested_outreach_angle"), original.suggested_outreach_angle)

    if not title or not summary or not reason or not outreach:
        return False
    if len(summary) > 600 or _word_count(summary) < 35:
        return False
    sent_count = len([s for s in _SENT_SPLIT_RE.split(summary) if s.strip()])
    if sent_count < 3 or sent_count > 4:
        return False
    if enrichment_needs_soft_framing(original, source_events):
        candidate_text = " ".join([title, summary, reason, outreach]).strip()
        if _ENRICHMENT_OVERCLAIM_RE.search(candidate_text):
            return False
    return True


_RECOVERY_GENERIC_CONTACT_ONLY_RE = re.compile(
    r"\b("
    r"if you have any questions|feel free to reach out|reach out through our social channels|"
    r"reach out via|contact us|contact via|social channels|get in touch"
    r")\b",
    re.IGNORECASE,
)

_RECOVERY_EXPLICIT_MOTION_RE = re.compile(
    r"\b("
    r"integrat(?:e|ion|or)|partner(?:ship)?s?|listing(?:s)?|audit|vendor|rfp|rfq|rfi|"
    r"grant(?:s)? program|accepting proposals|seeking (?:partners|vendors|integrators|auditors|distributors)|"
    r"distribution|onboard(?:ing)?|operator|co-build|implementation support|migration support|"
    r"webhook|sdk|api|plugin"
    r")\b",
    re.IGNORECASE,
)


def recovery_important_field_issue(candidate: Dict[str, Any], row: Dict[str, Any]) -> Optional[str]:
    raw_title = clean_text(candidate.get("title"))
    raw_summary = clean_text(candidate.get("summary"))
    raw_reason = clean_text(candidate.get("reason"))
    raw_outreach = clean_text(candidate.get("suggested_outreach_angle"))
    raw_target = normalize_target_company(candidate.get("target_company") or row.get("target_company"))

    if not raw_title:
        return "missing_title"
    if not raw_summary:
        return "missing_summary"
    if not raw_reason:
        return "missing_reason"
    if not raw_outreach:
        return "missing_outreach"
    if not raw_target:
        return "missing_target_company"

    title = normalize_title_text(raw_title)
    summary = normalize_summary_text(
        raw_summary,
        [
            raw_reason,
            raw_outreach,
            row.get("reason_text"),
            row.get("summary"),
            row.get("title"),
        ],
    )
    reason = normalize_reason_text(raw_reason)
    outreach = normalize_outreach_text(raw_outreach)

    if not title or title == "Opportunity":
        return "weak_title"
    if not summary or _word_count(summary) < 35:
        return "weak_summary"
    sent_count = len([s for s in _SENT_SPLIT_RE.split(summary) if s.strip()])
    if sent_count < 3 or sent_count > 4:
        return "weak_summary_structure"
    if not reason or _word_count(reason) < 18:
        return "weak_reason"
    if not outreach or _word_count(outreach) < 8:
        return "weak_outreach"

    evidence_type = clean_text(row.get("evidence_type")) or clean_text(candidate.get("evidence_type")) or ""
    evidence_snippet = enforce_verbatim_snippet(row.get("evidence_snippet") or candidate.get("evidence_snippet")) or ""
    if not evidence_type or not evidence_snippet:
        return "missing_evidence"

    source_context = " ".join(
        [
            evidence_snippet,
            clean_text(row.get("body_preview")) or "",
            clean_text(row.get("summary")) or "",
            clean_text(row.get("reason_text")) or "",
            clean_text(row.get("feedback")) or "",
        ]
    ).strip()

    if (
        evidence_type in {"partnership_or_integration", "support_added_with_bd_action"}
        and _RECOVERY_GENERIC_CONTACT_ONLY_RE.search(evidence_snippet)
        and not _RECOVERY_EXPLICIT_MOTION_RE.search(source_context)
    ):
        return "generic_contact_only"

    return None


def call_keep_enrichment_agent(
    client: OpenAI,
    model: str,
    opportunity: Opportunity,
    source_events: List[Event],
    max_output_tokens: int,
    tracker: Optional[TokenTracker] = None,
) -> Dict[str, Optional[str]]:
    evidence_snippet = extract_evidence_snippet_from_reason(opportunity.reason) or ""
    evidence_type = extract_evidence_type_from_reason(opportunity.reason) or ""
    events_payload = [build_enrichment_event_context(ev, evidence_snippet) for ev in source_events]
    soft_framing_mode = enrichment_needs_soft_framing(opportunity, source_events)
    opp_payload = {
        "title": opportunity.title,
        "summary": opportunity.summary,
        "reason": extract_reason_text_from_reason(opportunity.reason),
        "suggested_outreach_angle": opportunity.suggested_outreach_angle,
        "categories": opportunity.categories,
        "target_company": opportunity.target_company,
        "evidence_type": evidence_type,
        "evidence_snippet": evidence_snippet,
    }

    system_prompt = f"""You are a deterministic opportunity enrichment agent for final KEEP opportunities.

Your job is to improve clarity, conciseness, and BD usefulness while preserving all factual claims.

You must:
- Rewrite only title, summary, reason, suggested_outreach_angle, and opportunity_details.
- Preserve the same target company, evidence basis, categories, and BD motion.
- Keep all claims grounded in the provided opportunity payload and source events.
- Keep the strength of the wording aligned to the source text. If the source says feedback, testing, discussion, proposal, or early exploration, keep that softer framing.
- Calibrate actionability using evidence_type:
  - If evidence_type is partnership_or_integration or support_added_with_bd_action, prefer action-oriented phrasing (evaluate, integrate, adopt, request access) only when the source explicitly provides a concrete action path.
  - If no explicit intake/contact/action path exists in the source, explicitly label it as signal-stage and say to monitor updates.
  - For input/feedback sources with a repo/tool/SDK/testnet, you may frame as evaluation or integration testing, but still keep the feedback posture.
- Classify the source posture before writing. Use the strongest fitting posture and phrase everything accordingly:
  - open application / grant / RFP / program window
  - commercial intake / quote / contact path
  - live integration or onboarding path
  - technical evaluation / public repo / testable release
  - input / feedback / discussion / community request
  - migration / update / service transition
  - pending governance proposal / contingent future path
  - self-intro / ecosystem signal / soft visibility post
- Rebuild the key fields around the actual opportunity, not around general event narration.
- Make the opportunity itself unmistakable:
  - title should name the target company and use the weakest accurate phrasing for the source posture
  - summary should explain the opportunity surface, why it is actionable, and what the BD motion is without upgrading the source posture
  - reason should justify why this is a valid opportunity now
  - suggested_outreach_angle should directly support the real source ask, not give generic networking advice
- Favor opportunity-centric wording over article-centric wording. Do not merely summarize the news/event.
- Write summary as 3-4 sentences, at least 35 words, and no more than 600 characters.
- Make suggested_outreach_angle a single crisp BD motion.
- Write opportunity_details as an internal-only field: 6-15 sentences, at least 70 words, and no more than 1600 total characters including spaces and punctuation.
- Always return opportunity_details for KEEP opportunities. If source material is thin, write the best grounded concise version that still fits the limits.
- Write fields according to source posture:
  - for application/program items, prefer wording like apply, open application, grant window, program intake
  - for contact/quote items, prefer wording like contact path, custom quote, commercial intake, pricing path
  - for technical evaluation items, prefer wording like public repo, early evaluation path, technical review, testable release
  - for input/discussion items, prefer wording like input request, feedback request, technical discussion, builder input
  - for migration/update items, prefer wording like migration path, endpoint transition, update required, service move
  - for pending governance items, prefer wording like proposed program, pending proposal, contingent incentives path
- Use different wording discipline by field:
  - title must be the shortest honest framing
  - summary must state only what is open now
  - opportunity_details must preserve stage, constraints, and any conditionality
  - suggested_outreach_angle must match the real ask: apply, request introduction, request quote, offer technical feedback, evaluate the tool, migrate, or discuss pending fit
  - when intake is not open, include an explicit signal-stage line (e.g., "No official intake yet; monitor updates.")

You must NOT:
- Introduce new entities, claims, timelines, or speculation.
- Upgrade soft source language into a stronger BD claim. Do not turn feedback/testing/community discussion into partnership, procurement, or open vendor intake unless the source explicitly supports that.
- Change evidence basis or invent a different action surface.
- Invent or imply a concrete action step when the source does not provide one.
- Rewrite categories, target_company, or evidence_snippet.
- Drift into generic polishing that weakens the opportunity framing.
- Use inflated labels like 'partnership opportunity', 'collaboration opportunity', 'integration opportunity', 'open BD channel', 'clear opportunity', or 'actionable opportunity' unless the source explicitly supports that stronger wording.

{"SOFT-SIGNAL MODE:\n- The source reads more like feedback/testing/community discussion than a strong intake signal.\n- Rephrase conservatively and keep the wording modest.\n- Do not label this as a partnership, vendor, procurement, application, onboarding, or commercial opportunity unless the source text explicitly says so.\n- Prefer wording like feedback, evaluation, technical discussion, early integration exploration, or builder input when that better matches the source.\n" if soft_framing_mode else ""}
{"WORDING CALIBRATION:\n- Use the weakest accurate phrasing that still preserves BD usefulness.\n- If the source is asking for feedback, ideas, comments, discussion, community input, or technical suggestions, mirror that exact posture instead of upgrading it to collaboration or partnership.\n- If the source only exposes a contact path or introductory note, keep the output as an introduction, support inquiry, or exploratory discussion - not a formal opportunity claim.\n- Good examples of modest wording: input request, feedback channel, early evaluation path, builder support inquiry, technical discussion, integration exploration.\n- Avoid inflated title patterns like 'Partnership Opportunity', 'Integration Opportunity', 'Collaboration Opportunity', or 'Open BD Channel' unless the source explicitly supports them.\n" if soft_framing_mode else ""}

If the current wording is already clear, return a cleaner but semantically equivalent version.
Return JSON only.
"""

    user_prompt = f"""CURRENT OPPORTUNITY:
{json.dumps(opp_payload, ensure_ascii=False, separators=(",", ":"))}

SOURCE EVENTS:
{json.dumps(events_payload, ensure_ascii=False, separators=(",", ":"))}

Rewrite and return only:
- title
- summary
- reason
- suggested_outreach_angle
- opportunity_details

Focus on making the output support the opportunity directly:
- What is the BD opportunity?
- Why is it actionable now?
- What should outreach lead with?
- Match the source's level of certainty and immediacy exactly; polish, but do not overstate.
- If the source mainly asks for feedback, ideas, comments, discussion, or technical suggestions, the title and summary must use that softer posture rather than partnership or integration-opportunity language.
- If the source mainly announces a repo, tool, SDK, endpoint, or release, frame it as evaluation/adoption/migration unless the source explicitly invites external teams to integrate, onboard, or apply now.
- If the source is pending governance approval, make the conditionality explicit in title, summary, and opportunity_details.
- If there is no explicit intake/contact/action path in the source, add a clear signal-stage line such as: "No official intake yet; monitor updates."
- Final self-check before answering:
  - Did you convert feedback into partnership?
  - Did you convert a repo launch into a BD lead?
  - Did you convert a pending proposal into a live open program?
  - If yes, downgrade the wording.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": openai_keep_enrichment_schema_v1()},
        temperature=0,
        max_completion_tokens=max_output_tokens,
    )

    if tracker:
        tracker.add_usage(resp.usage, model, agent="Enrichment")

    raw = (resp.choices[0].message.content or "").strip()
    parsed = json.loads(raw)
    details = normalize_details_text(parsed.get("opportunity_details"))
    if not details:
        details = build_fallback_opportunity_details(opportunity, source_events)

    return {
        "title": normalize_title_text(parsed.get("title")),
        "summary": normalize_summary_text(
            parsed.get("summary"),
            [
                parsed.get("reason"),
                parsed.get("suggested_outreach_angle"),
                opportunity.reason,
                opportunity.summary,
            ],
        ),
        "reason": normalize_reason_text(parsed.get("reason")) or normalize_reason_text(opportunity.reason),
        "suggested_outreach_angle": normalize_outreach_text(parsed.get("suggested_outreach_angle"), opportunity.suggested_outreach_angle),
        "opportunity_details": details,
    }


def normalize_source_event_ids(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split("|") if v.strip()]
    return []


def near_miss_effective_confidence(row: Dict[str, Any]) -> float:
    raw_conf = row.get("confidence")
    try:
        if raw_conf not in (None, "", "NULL"):
            return float(raw_conf)
    except Exception:
        pass

    sub = row.get("sub_scores") or {}
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except Exception:
            sub = {}
    if isinstance(sub, dict):
        return confidence_from_subscores(sub)
    return 0.0


def near_miss_review_haystack(row: Dict[str, Any]) -> str:
    return " ".join(
        [
            clean_text(row.get("title")) or "",
            clean_text(row.get("summary")) or "",
            clean_text(row.get("evidence_snippet")) or "",
            clean_text(row.get("feedback")) or "",
            clean_text(row.get("reason_text")) or "",
            clean_text(row.get("suggested_outreach_angle")) or "",
            smart_truncate(clean_text(row.get("body_preview")) or "", 1600),
        ]
    )


_RECOVERY_HINT_RE = re.compile(
    r"\b(application|applications|apply|application thread|intake|open grant|grant proposal|rfp|"
    r"admission|operator status|professional operator|trusted operator|renew|renewal|annual engagement|services proposal|"
    r"support/funding|seeking support|audit|cover an audit|appetite to add|add .* into the .* app|"
    r"integrat(?:e|ion)|integrator|partner program|early partner|webhooks?|sdk|api|data streams?|dashboard|"
    r"monitoring|validator|delegator|light node|stablecoin|wallet-as-a-service|wallet security|address poisoning|"
    r"scam(?:-| )detection|governance data|data api|cohort|bug bounty|"
    r"codebase|documentation|docs|openapi|swagger|open[ -]source|publicly available|repository|"
    r"token standard|privacy pool|live demo|demo api|tagged release|"
    r"co-design|co-fund|co-run|collaborators?|partner projects?|deploy|listing|list on|extend)\b",
    flags=re.IGNORECASE,
)

_STRONG_IMPLICIT_BD_RE = re.compile(
    r"\b("
    r"api|sdk|webhook|plugin|developer docs|documentation|gasless|smart account|wallet-as-a-service|"
    r"stablecoin|liquidity rail|settlement|cash[- ]?out|remittance|treasury|payments?|"
    r"audit|security review|remediation|production-ready|not production-ready|unaudited|implementation need|"
    r"integrat(?:e|ion)|integrator|distribution|channel partner|operator|institutional|regional|latam|"
    r"co-build|pilot|prototype|platform adoption|wallet and payment APIs|western union|usdpt|ussd"
    r")\b",
    flags=re.IGNORECASE,
)

_OBVIOUS_NONRECOVERY_RE = re.compile(
    r"\b("
    r"hackathon|contest|delegate race|nominations?|community translation|retrospective|final report|"
    r"governance discussion|temp check|rfc|comment period|monitoring only|watchdog|ops contact|"
    r"waitlist|sign up|signup|try it|visit the site|dashboard launch"
    r")\b",
    flags=re.IGNORECASE,
)


def is_strong_implicit_bd_candidate(row: Dict[str, Any]) -> bool:
    haystack = near_miss_review_haystack(row)
    evidence_type = (row.get("evidence_type") or "").strip()
    confidence = near_miss_effective_confidence(row)

    if confidence < 0.72:
        return False
    if _OBVIOUS_NONRECOVERY_RE.search(haystack):
        return False
    if re.search(r"\b(deadline passed|applications closed|submission period ended|closed on)\b", haystack, flags=re.IGNORECASE):
        return False
    if evidence_type not in {"partnership_or_integration", "support_added_with_bd_action", "grant_or_rfp_or_program_open", "vendor_or_procurement_need"}:
        return False
    if not _STRONG_IMPLICIT_BD_RE.search(haystack):
        return False
    return True


def should_review_near_miss(row: Dict[str, Any], min_confidence: float) -> bool:
    stage = (row.get("stage") or "").strip().lower()
    if stage not in {"finder", "critic"}:
        return False

    reason = (row.get("reason") or "").strip().lower()
    haystack = near_miss_review_haystack(row)
    confidence = near_miss_effective_confidence(row)

    # Recovery consideration qualification:
    # only review near-misses that are already reasonably close, while
    # still allowing the recovery agent to inspect the full candidate payload.
    if confidence < RECOVERY_MIN_CONFIDENCE:
        return False

    if reason == "below_min_confidence":
        return True

    if is_strong_implicit_bd_candidate(row):
        return True

    if (
        (row.get("evidence_type") or "").strip() == "vendor_or_procurement_need"
        and _RECOVERY_HINT_RE.search(haystack)
    ):
        return True

    if re.search(r"\b(api|sdk|webhook|plugin|developer docs|documentation|integrate now|integration path)\b", haystack, flags=re.IGNORECASE):
        return True

    if (
        (row.get("evidence_type") or "").strip() == "grant_or_rfp_or_program_open"
        and (
            re.search(r"\b(open grant|grant proposal|application thread|applications? open|rfp)\b", haystack, flags=re.IGNORECASE)
            or re.search(r"\b(proposer|github\.com/|forum thread|reply in-thread|ongoing through|program is ongoing|still open|accepting contributors)\b", haystack, flags=re.IGNORECASE)
        )
    ):
        return True

    if (
        (row.get("evidence_type") or "").strip() == "partnership_or_integration"
        and re.search(r"\b(admission|operator status|professional operator|trusted operator|audit|cover an audit|appetite to add|collaboration|pilot|prototype|distribution|co-build|incubator)\b", haystack, flags=re.IGNORECASE)
    ):
        return True

    if _RECOVERY_HINT_RE.search(haystack):
        return True

    return False


def recovery_priority(row: Dict[str, Any], min_confidence: float) -> float:
    confidence = near_miss_effective_confidence(row)
    haystack = near_miss_review_haystack(row)
    evidence_type = (row.get("evidence_type") or "").strip()

    score = confidence
    if is_strong_implicit_bd_candidate(row):
        score += 3.0
        if re.search(r"\b(api|sdk|webhook|plugin|wallet and payment APIs|smart account|gasless)\b", haystack, flags=re.IGNORECASE):
            score += 1.75
        if re.search(r"\b(stablecoin|liquidity rail|settlement|cash[- ]?out|remittance|treasury|usdpt|ussd)\b", haystack, flags=re.IGNORECASE):
            score += 1.75
        if re.search(r"\b(audit|security review|remediation|production-ready|unaudited|implementation need)\b", haystack, flags=re.IGNORECASE):
            score += 1.75
        if re.search(r"\b(distribution|channel partner|operator|institutional|regional|latam|pilot|co-build)\b", haystack, flags=re.IGNORECASE):
            score += 1.5
    if (row.get("reason") or "").strip().lower() == "below_min_confidence":
        score += 2.0
        if confidence >= min_confidence - 0.05:
            score += 0.75
    if evidence_type == "grant_or_rfp_or_program_open":
        if re.search(r"\b(open grant|grant proposal|application thread|applications? open|rfp)\b", haystack, flags=re.IGNORECASE):
            score += 2.25
        if re.search(r"\b(proposer|github\.com/|forum thread|reply in-thread|ongoing through|program is ongoing|still open|accepting contributors)\b", haystack, flags=re.IGNORECASE):
            score += 0.75
    if evidence_type == "partnership_or_integration":
        if re.search(r"\b(admission|operator status|professional operator|trusted operator)\b", haystack, flags=re.IGNORECASE):
            score += 2.0
        if re.search(r"\b(audit|cover an audit|appetite to add|add .* into the .* app|collaboration|pilot|prototype|distribution|co-build|incubator)\b", haystack, flags=re.IGNORECASE):
            score += 2.0
    if re.search(r"\b(application|applications|apply|application thread|rfp|open grant)\b", haystack, flags=re.IGNORECASE):
        score += 1.5
    if re.search(r"\b(api|sdk|webhook|plugin|developer docs|documentation|integrate now|integration path)\b", haystack, flags=re.IGNORECASE):
        score += 1.5
    if re.search(r"\b(admission|operator status|professional operator)\b", haystack, flags=re.IGNORECASE):
        score += 1.2
    if re.search(r"\b(renew|renewal|annual engagement|services proposal|support/funding)\b", haystack, flags=re.IGNORECASE):
        score += 1.0
    if re.search(r"\b(integrat(?:e|ion)|integrator|partner program|early partner|collaboration|pilot|prototype|distribution|co-build|incubator)\b", haystack, flags=re.IGNORECASE):
        score += 0.8
    return score


def select_recovery_source_events(row: Dict[str, Any], event_lookup: Dict[str, Event]) -> List[Event]:
    event_ids = normalize_source_event_ids(row.get("source_event_ids"))
    evidence_snippet = clean_text(row.get("evidence_snippet")) or ""
    candidates: List[Tuple[int, int, int, Event]] = []

    for order_idx, eid in enumerate(event_ids):
        ev = event_lookup.get(eid)
        if not ev:
            continue
        haystack = " ".join([ev.title or "", ev.summary or "", ev.description or ""])
        snippet_match = 1 if evidence_snippet and evidence_snippet in haystack else 0
        body_len = len((ev.description or "").strip())
        candidates.append((-snippet_match, -body_len, order_idx, ev))

    return [item[3] for item in sorted(candidates)[:3]]


def select_recovery_candidates(
    near_miss_rows: List[Dict[str, Any]],
    event_lookup: Dict[str, Event],
    min_confidence: float,
    cap: int = RECOVERY_MAX_CANDIDATES,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for row in near_miss_rows:
        if not should_review_near_miss(row, min_confidence):
            continue
        event_ids = normalize_source_event_ids(row.get("source_event_ids"))
        if not event_ids or not any(eid in event_lookup for eid in event_ids):
            continue
        dedupe_key = normalize_key(f"{row.get('title') or ''}|{'|'.join(event_ids)}|{row.get('reason') or ''}")
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        candidates.append(row)

    candidates.sort(
        key=lambda r: (
            -near_miss_effective_confidence(r),
            -recovery_priority(r, min_confidence),
            str(r.get("title") or ""),
        )
    )
    return candidates[:cap]


def opportunity_to_recovery_row(opp: Opportunity, stage: str = "finalizer_drop") -> Dict[str, Any]:
    evidence_type = opportunity_evidence_type(opp) or ""
    evidence_snippet = opportunity_evidence_snippet(opp) or ""
    supporting_snippet = opportunity_supporting_snippet(opp) or ""
    return {
        "stage": stage,
        "status": "DROP",
        "audit_label": opp.audit_label or "Cut",
        "reason": opp.reason,
        "title": opp.title,
        "summary": opp.summary,
        "body_preview": (clean_text(opp.opportunity_details or opp.summary or opp.reason or "") or "")[:1000],
        "target_company": opp.target_company,
        "suggested_outreach_angle": opp.suggested_outreach_angle,
        "categories": list(opp.categories or []),
        "confidence": float(opp.confidence or 0.0),
        "min_confidence": 0.0,
        "evidence_type": evidence_type,
        "evidence_snippet": evidence_snippet,
        "supporting_snippet": supporting_snippet,
        "reason_text": extract_reason_text_from_reason_block(opp.reason) or opp.reason,
        "bd_signal": opp.audit_label or "finalizer_cut",
        "feedback": "finalizer_drop_candidate",
        "sub_scores": {},
        "source_event_ids": normalize_source_event_ids(opp.event_ids),
        "_drop_opp_id": id(opp),
    }


def select_dropped_recovery_candidates(
    drop_opps: List[Opportunity],
    event_lookup: Dict[str, Event],
    min_confidence: float,
    cap: int = RECOVERY_MAX_CANDIDATES,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    review_floor = max(0.0, min(float(min_confidence or 0.0), DROP_RECOVERY_MIN_CONFIDENCE_FLOOR))

    for opp in drop_opps:
        row = opportunity_to_recovery_row(opp)
        event_ids = normalize_source_event_ids(row.get("source_event_ids"))
        if not event_ids or not any(eid in event_lookup for eid in event_ids):
            continue
        if near_miss_effective_confidence(row) < review_floor:
            continue
        dedupe_key = normalize_key(f"{row.get('title') or ''}|{'|'.join(event_ids)}|{row.get('reason') or ''}")
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        candidates.append(row)

    candidates.sort(
        key=lambda r: (
            -near_miss_effective_confidence(r),
            -recovery_priority(r, review_floor),
            str(r.get("title") or ""),
        )
    )
    return candidates[:cap]


def compute_drop_recovery_min_restored(keep_count: int, drop_count: int) -> Tuple[float, int]:
    if drop_count <= 0:
        return 0.0, 0

    if keep_count <= 0:
        return float("inf"), min(RECOVERY_MAX_RESTORED, drop_count)

    ratio = float(drop_count) / float(max(keep_count, 1))
    if ratio >= DROP_RECOVERY_RATIO_FORCE_THREE:
        return ratio, min(3, RECOVERY_MAX_RESTORED, drop_count)
    if ratio >= DROP_RECOVERY_RATIO_FORCE_TWO:
        return ratio, min(2, RECOVERY_MAX_RESTORED, drop_count)
    if ratio >= DROP_RECOVERY_RATIO_FORCE_ONE:
        return ratio, min(1, RECOVERY_MAX_RESTORED, drop_count)
    return ratio, 0


def clone_opportunity(opp: Opportunity) -> Opportunity:
    return Opportunity(
        opportunity_id=opp.opportunity_id,
        title=opp.title,
        summary=opp.summary,
        reason=opp.reason,
        who_to_contact=opp.who_to_contact,
        suggested_outreach_angle=opp.suggested_outreach_angle,
        categories=list(opp.categories or []),
        filter_chain=opp.filter_chain,
        filter_sector=opp.filter_sector,
        filter_seeking=opp.filter_seeking,
        time_found=opp.time_found,
        confidence=float(opp.confidence or 0.0),
        tags=list(opp.tags or []),
        target_company=opp.target_company,
        sources=list(opp.sources or []),
        event_ids=opp.event_ids,
        event_titles=opp.event_titles,
        event_url=opp.event_url,
        bd_weeks=opp.bd_weeks,
        evidence_type=opp.evidence_type,
        evidence_snippet=opp.evidence_snippet,
        supporting_snippet=opp.supporting_snippet,
        opportunity_details=opp.opportunity_details,
        audit_label=opp.audit_label,
        finalizer_reason=opp.finalizer_reason,
    )


def select_watchlist_promotion_candidates(
    watch_opps: List[Opportunity],
    cap: int = WATCHLIST_PROMOTION_MAX_CANDIDATES,
) -> List[Opportunity]:
    ranked = sorted(
        watch_opps,
        key=lambda o: (
            -watchlist_promotion_priority(o),
            -float(o.confidence or 0.0),
            -(1 if opportunity_evidence_snippet(o) else 0),
            o.title or "",
        ),
    )
    selected: List[Opportunity] = []
    seen_keys: Set[str] = set()
    for opp in ranked:
        dedupe_key = normalize_key(
            f"{opp.title or ''}|{opp.target_company or ''}|{opp.event_ids or ''}|{extract_reason_text_from_reason_block(opp.reason)}"
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        selected.append(opp)
        if len(selected) >= cap:
            break
    return selected


def build_watchlist_promotion_payload(
    watch_opps: List[Opportunity],
    event_lookup: Dict[str, Event],
    cap: int = WATCHLIST_PROMOTION_MAX_CANDIDATES,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    payload: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for opp in select_watchlist_promotion_candidates(watch_opps, cap=cap):
        event_ids = normalize_source_event_ids(opp.event_ids)
        if not event_ids:
            continue

        source_events = select_enrichment_source_events(opp, event_lookup)
        if not source_events:
            source_events = [event_lookup[eid] for eid in event_ids if eid in event_lookup][:2]
        if not source_events:
            continue

        evidence_snippet = extract_evidence_snippet_from_reason(opp.reason) or ""
        supporting_snippet = extract_supporting_snippet_from_reason(opp.reason) or ""
        reason_text = extract_reason_text_from_reason_block(opp.reason) or opp.summary or ""
        body_preview = smart_truncate(
            " ".join(
                [
                    source_events[0].summary or "",
                    source_events[0].description or "",
                ]
            ).strip(),
            400,
        ) if source_events else ""

        row = {
            "stage": "watchlist_finalizer",
            "decision": "watchlist",
            "title": opp.title,
            "summary": opp.summary,
            "target_company": opp.target_company,
            "categories": list(opp.categories or []),
            "suggested_outreach_angle": opp.suggested_outreach_angle,
            "confidence": float(opp.confidence or 0.0),
            "evidence_type": extract_evidence_type_from_reason(opp.reason) or "watchlist_signal",
            "evidence_snippet": evidence_snippet,
            "supporting_snippet": supporting_snippet,
            "source_event_ids": event_ids,
            "reason_text": reason_text,
            "body_preview": body_preview,
            "watch_opportunity": opp,
        }
        rows.append(row)
        payload.append(
            {
                "candidate": {
                    **{k: v for k, v in row.items() if k != "watch_opportunity"},
                },
                "source_events": [
                    {
                        **build_enrichment_event_context(ev, evidence_snippet),
                        "body_context": _polish_recovery_body_context(
                            build_enrichment_event_context(ev, evidence_snippet).get("body_context", "")
                        ),
                    }
                    for ev in source_events
                ],
            }
        )

    return payload, rows


def promote_watchlist_candidates(
    client: OpenAI,
    model: str,
    watch_opps: List[Opportunity],
    keep_opps: List[Opportunity],
    event_lookup: Dict[str, Event],
    category_catalog: List[Dict[str, str]],
    categories_map: Dict[str, str],
    category_definitions: Dict[str, str],
    max_output_tokens: int,
    tracker: Optional[TokenTracker],
    rejection_audit: List[Dict[str, Any]],
) -> Tuple[List[Opportunity], List[Opportunity]]:
    payload, rows = build_watchlist_promotion_payload(watch_opps, event_lookup, cap=WATCHLIST_PROMOTION_MAX_CANDIDATES)
    if not payload:
        return keep_opps, watch_opps

    print(f"[OpportunityMatcher] Running WATCHLIST Promotion Agent on {len(payload)} candidates...")
    accepted_examples = build_watchlist_promotion_reference_examples(keep_opps, max_items=6)
    weakest_keep_score = weakest_keep_promotion_score(keep_opps)
    clear_upgrade_margin = 0.75

    promoted_opps: List[Opportunity] = []
    promoted_ids: Set[str] = set()

    def _row_upgrade_score(row: Dict[str, Any]) -> float:
        original = row.get("watch_opportunity")
        if not isinstance(original, Opportunity):
            return float("-inf")
        return watchlist_promotion_priority(original)

    def _row_is_clear_upgrade(row: Dict[str, Any]) -> bool:
        score = _row_upgrade_score(row)
        if score == float("-inf"):
            return False
        return score >= (weakest_keep_score + clear_upgrade_margin)

    def _apply_promotion_decisions(decisions: List[Dict[str, Any]]) -> None:
        nonlocal promoted_opps, promoted_ids
        for decision in decisions:
            if len(promoted_opps) >= WATCHLIST_PROMOTION_MAX_PROMOTED:
                break
            if not isinstance(decision, dict) or not decision.get("recover"):
                continue
            idx = int(decision.get("index", -1))
            if idx < 0 or idx >= len(rows):
                continue
            row = rows[idx]
            original = row.get("watch_opportunity")
            if not isinstance(original, Opportunity):
                continue
            if original.opportunity_id in promoted_ids:
                continue

            ro = decision.get("opportunity")
            if isinstance(ro, list):
                ro = ro[0] if ro else None

            built: Optional[Opportunity] = None
            if isinstance(ro, dict) and ro.get("is_opportunity", False):
                ro["source_event_ids"] = normalize_source_event_ids(row.get("source_event_ids"))
                ro["evidence_type"] = row.get("evidence_type")
                ro["evidence_snippet"] = enforce_verbatim_snippet(row.get("evidence_snippet")) or ""
                ro["supporting_snippet"] = enforce_verbatim_snippet(row.get("supporting_snippet"), max_words=30) or ""
                field_issue = recovery_important_field_issue(ro, row)
                if field_issue:
                    rejection_audit.append({
                        "stage": "watchlist_promotion",
                        "decision": "watchlist",
                        "reason": f"promotion_field_issue:{field_issue}",
                        "title": row.get("title", ""),
                        "target_company": row.get("target_company"),
                        "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                        "feedback": decision.get("rationale", ""),
                    })
                    continue
                conf = confidence_from_subscores(ro.get("sub_scores", {}) or {})
                ro["confidence"] = max(conf, float(original.confidence or 0.0))
                built, _ = build_opportunity_from_raw_candidate(
                    ro,
                    event_lookup,
                    categories_map,
                    category_definitions,
                )

            if not built:
                built = clone_opportunity(original)

            if any(existing.event_ids == built.event_ids and normalize_key(existing.title) == normalize_key(built.title) for existing in keep_opps + promoted_opps):
                continue

            promoted_opps.append(built)
            promoted_ids.add(original.opportunity_id)
            rejection_audit.append({
                "stage": "watchlist_promotion",
                "decision": "promoted",
                "title": built.title,
                "target_company": built.target_company,
                "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                "feedback": decision.get("rationale", ""),
            })

    def _apply_deterministic_upgrades(required_min: int = 2) -> None:
        nonlocal promoted_opps, promoted_ids
        ranked_rows = sorted(
            rows,
            key=lambda r: (-_row_upgrade_score(r), -float((r.get("confidence") or 0.0)), str(r.get("title") or "")),
        )
        clear_candidates = [r for r in ranked_rows if _row_is_clear_upgrade(r)]

        target_count = min(required_min, WATCHLIST_PROMOTION_MAX_PROMOTED, len(clear_candidates))
        if target_count <= len(promoted_opps):
            return

        for row in clear_candidates:
            if len(promoted_opps) >= target_count or len(promoted_opps) >= WATCHLIST_PROMOTION_MAX_PROMOTED:
                break
            original = row.get("watch_opportunity")
            if not isinstance(original, Opportunity):
                continue
            if original.opportunity_id in promoted_ids:
                continue
            built = clone_opportunity(original)
            if any(existing.event_ids == built.event_ids and normalize_key(existing.title) == normalize_key(built.title) for existing in keep_opps + promoted_opps):
                continue
            promoted_opps.append(built)
            promoted_ids.add(original.opportunity_id)
            rejection_audit.append({
                "stage": "watchlist_promotion",
                "decision": "promoted",
                "title": built.title,
                "target_company": built.target_company,
                "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                "feedback": "deterministic_upgrade_vs_weakest_keep",
            })

    def _apply_ranked_floor(required_min: int = 2) -> None:
        nonlocal promoted_opps, promoted_ids
        if len(promoted_opps) >= required_min:
            return

        ranked_rows = sorted(
            rows,
            key=lambda r: (-_row_upgrade_score(r), -float((r.get("confidence") or 0.0)), str(r.get("title") or "")),
        )

        target_count = min(required_min, WATCHLIST_PROMOTION_MAX_PROMOTED, len(ranked_rows))
        for row in ranked_rows:
            if len(promoted_opps) >= target_count or len(promoted_opps) >= WATCHLIST_PROMOTION_MAX_PROMOTED:
                break
            original = row.get("watch_opportunity")
            if not isinstance(original, Opportunity):
                continue
            if original.opportunity_id in promoted_ids:
                continue
            built = clone_opportunity(original)
            if any(existing.event_ids == built.event_ids and normalize_key(existing.title) == normalize_key(built.title) for existing in keep_opps + promoted_opps):
                continue
            promoted_opps.append(built)
            promoted_ids.add(original.opportunity_id)
            rejection_audit.append({
                "stage": "watchlist_promotion",
                "decision": "promoted",
                "title": built.title,
                "target_company": built.target_company,
                "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                "feedback": "ranked_floor_to_minimum_three",
            })

    try:
        decisions = call_recovery_agent(
            client=client,
            model=model,
            candidates=payload,
            category_catalog=category_catalog,
            accepted_examples=accepted_examples,
            max_output_tokens=max_output_tokens,
            tracker=tracker,
            prefer_early_signals=True,
        )
        _apply_promotion_decisions(decisions)

        if not promoted_opps:
            fallback_decisions = call_recovery_agent(
                client=client,
                model=model,
                candidates=payload[: min(len(payload), WATCHLIST_PROMOTION_MAX_CANDIDATES)],
                category_catalog=category_catalog,
                accepted_examples=accepted_examples,
                max_output_tokens=max_output_tokens,
                tracker=tracker,
                force_one=True,
                prefer_early_signals=True,
            )
            _apply_promotion_decisions(fallback_decisions)

        _apply_deterministic_upgrades(required_min=2)
        _apply_ranked_floor(required_min=2)
    except Exception as e:
        print(f"[OpportunityMatcher] WARNING: WATCHLIST Promotion Agent failed: {e}")

    if not promoted_opps and rows:
        fallback_row = rows[0]
        fallback_opp = fallback_row.get("watch_opportunity")
        if isinstance(fallback_opp, Opportunity):
            promoted_opps = [clone_opportunity(fallback_opp)]
            promoted_ids.add(fallback_opp.opportunity_id)
            rejection_audit.append({
                "stage": "watchlist_promotion",
                "decision": "promoted",
                "title": fallback_opp.title,
                "target_company": fallback_opp.target_company,
                "source_event_ids": normalize_source_event_ids(fallback_row.get("source_event_ids")),
                "feedback": "forced_best_watchlist_candidate",
            })

    if promoted_opps:
        promoted_opps = dedupe_opportunities(promoted_opps)[:WATCHLIST_PROMOTION_MAX_PROMOTED]
        keep_opps = dedupe_opportunities(keep_opps + promoted_opps)
        watch_opps = [o for o in watch_opps if o.opportunity_id not in promoted_ids]
        print(
            f"[OpportunityMatcher] WATCHLIST Promotion Agent promoted {len(promoted_opps)} opportunities "
            f"(cap={WATCHLIST_PROMOTION_MAX_PROMOTED})"
        )
        print(f"[OpportunityMatcher] WATCHLIST Promotion IDs: {[o.opportunity_id for o in promoted_opps]}")
    else:
        print(
            f"[OpportunityMatcher] WATCHLIST Promotion Agent promoted 0 opportunities "
            f"(cap={WATCHLIST_PROMOTION_MAX_PROMOTED})"
        )

    return keep_opps, watch_opps


def recovery_skip_reasons(
    candidate_count: int,
    current_opp_count: int,
    min_candidates_to_run: int = RECOVERY_MIN_CANDIDATES_TO_RUN,
) -> List[str]:
    reasons: List[str] = []
    if candidate_count < min_candidates_to_run:
        reasons.append(
            f"candidate_count={candidate_count} < {min_candidates_to_run}"
        )
    if current_opp_count >= RECOVERY_SKIP_IF_PRE_DEDUPE_COUNT_AT_LEAST:
        reasons.append(
            f"pre_dedupe_opportunities={current_opp_count} >= {RECOVERY_SKIP_IF_PRE_DEDUPE_COUNT_AT_LEAST}"
        )
    return reasons


def _polish_recovery_body_context(text: Optional[str], max_chars: int = 2200) -> str:
    cleaned = re.sub(r"\s+", " ", (clean_text(text) or "").strip()).strip()
    return smart_truncate(cleaned, max_chars) if cleaned else ""


def should_override_critic_discard(row: Dict[str, Any], feedback: Dict[str, Any], min_confidence: float) -> bool:
    if not is_strong_implicit_bd_candidate(row):
        return False

    haystack = near_miss_review_haystack(
        {
            **row,
            "feedback": (feedback or {}).get("feedback") or (feedback or {}).get("critic_notes") or "",
        }
    )
    if _OBVIOUS_NONRECOVERY_RE.search(haystack):
        return False
    if re.search(r"\b(no action surface|wrong actor|deadline passed|applications closed|self-serve signup|waitlist)\b", haystack, flags=re.IGNORECASE):
        return False

    confidence = near_miss_effective_confidence(row)
    return confidence >= max(min_confidence, 0.72)


def critic_discard_is_hard_exclusion(row: Dict[str, Any], feedback: Dict[str, Any], min_confidence: float) -> bool:
    if should_override_critic_discard(row, feedback, min_confidence):
        return False

    haystack = near_miss_review_haystack(
        {
            **row,
            "feedback": (feedback or {}).get("feedback") or (feedback or {}).get("critic_notes") or "",
        }
    )

    if re.search(
        r"\b(deadline passed|applications closed|submission period ended|closed on|expired|window is already closed)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return True

    if re.search(
        r"\b(hackathon|contest|delegate race|nominations?|community translation|emergency contact|ops contact|admin registration|"
        r"generic contributor intake|security council candidacy|security council reelection)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return True

    if re.search(
        r"\b(waitlist|sign up|signup|try it|visit the site|self-serve|generic trial|dashboard launch)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return True

    if re.search(
        r"\b(retrospective|final report|monitoring only|governance discussion|temp check|rfc|comment period|roadmap call|"
        r"informational update|policy signal|macro|price move)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return True

    if re.search(
        r"\b(wrong actor|wrong role|no action surface|no explicit intake|applicant-side ask|seeking funding for itself)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return True

    return False


_GATEKEEPER_POSTPASS_WEAK_SIGNAL_RE = re.compile(
    r"\b("
    r"open grant proposal|grant proposal|grant request|grant application|devgrant proposal|"
    r"application to become|node operator application|operator application|professional operator|"
    r"ambassador (?:application|referral program)|ceo search application|"
    r"agenda suggestions?|add agenda(?: items?)?|community feedback|seeking feedback|requests? feedback|"
    r"discussion of|governance proposal|rfc|claim(?:ing)? (?:a )?profile|claim pre-?created profile|"
    r"meetup(?: registration)?|registration open|committee applications?|recognized delegate|delegate program|"
    r"topic submissions?|input sought|feedback request|draft erc|proposal review(?: service)?|"
    r"contributions welcome|open[ -]source(?: repo)?(?: available)?|forum discussion|public comment|"
    r"nightly release|status update|self-serve(?: product)?"
    r")\b",
    flags=re.IGNORECASE,
)

_GATEKEEPER_POSTPASS_APPLICANT_SIDE_RE = re.compile(
    r"\b("
    r"grant proposal|grant request|grant application|devgrant proposal|filecoin devgrant proposal|"
    r"proposal seeking|seeking technical sponsor|seeking ecosystem input|proposal category|"
    r"application to become|committee applications?|operator application|professional operator application"
    r")\b",
    flags=re.IGNORECASE,
)

_GATEKEEPER_POSTPASS_GOVERNANCE_FLOW_RE = re.compile(
    r"\b("
    r"governance (?:committee|proposal|call|forum)|recognized delegate|delegate program|delegate race|"
    r"committee applications?|topic submissions?|agenda suggestions?|nominations?|elections?|"
    r"proposal review(?: service)?|forum thread|snapshot"
    r")\b",
    flags=re.IGNORECASE,
)

_GATEKEEPER_POSTPASS_FEEDBACK_ONLY_RE = re.compile(
    r"\b("
    r"feedback request|requests? feedback|seeking feedback|input sought|comments? welcome|"
    r"discussion(?: thread)?|forum discussion|draft erc|rfc|proposal discussion|public comment"
    r")\b",
    flags=re.IGNORECASE,
)

_GATEKEEPER_POSTPASS_FILECOIN_DEVGRANT_RE = re.compile(
    r"\b("
    r"filecoin devgrant|filecoin devgrants|devgrant proposal|open grant proposal|filecoin open grants?|"
    r"retrieval category|fvm category"
    r")\b",
    flags=re.IGNORECASE,
)

_GATEKEEPER_POSTPASS_FILECOIN_BUYER_INTAKE_RE = re.compile(
    r"\b("
    r"request for proposals|requests for proposals|rfp|rfq|rfi|applications? open|apply now|"
    r"submit (?:a )?proposal|call for proposals|grant program (?:is )?(?:live|open|accepting)|"
    r"accepting applications|accepting proposals|seeking projects|program managers? invite|"
    r"official intake|partner intake"
    r")\b",
    flags=re.IGNORECASE,
)

_GATEKEEPER_POSTPASS_STRONG_COUNTERSIGNAL_RE = re.compile(
    r"\b("
    r"onboarding|partner onboarding|partner intake|intake form|applications? open|apply now|"
    r"vendor|procurement|request (?:quotes?|a quote)|routes quotes|rfp|rfq|rfi|"
    r"audit(?: providers?)?|security providers?|migrat(?:e|ion)|upgrade immediately|"
    r"integrat(?:e now|ion path|ors?)|implementation support|talk to us|contact sales|book a demo"
    r")\b",
    flags=re.IGNORECASE,
)

_GATEKEEPER_POSTPASS_ROLE_INVERTED_RE = re.compile(
    r"\b("
    r"proposer:|proposal category|application to become|seeking funding|requests? funding|"
    r"grant request|grant proposal|devgrant proposal|ambassador referral program|ceo search application"
    r")\b",
    flags=re.IGNORECASE,
)

_FINALIZER_GOVERNANCE_WATCH_RE = re.compile(
    r"\b("
    r"governance|forum|discussion|feedback|input sought|draft erc|rfc|committee|delegate|"
    r"proposal review|topic submissions?|agenda"
    r")\b",
    flags=re.IGNORECASE,
)

_FINALIZER_SIGNAL_STAGE_WATCH_RE = re.compile(
    r"\b("
    r"signal-stage|no official intake yet|monitor updates|under review|pending approval|pending dao approval|"
    r"if (?:the )?grant is approved|if approved|not yet open|forthcoming|future intake|contingent"
    r")\b",
    flags=re.IGNORECASE,
)

_FINALIZER_APPLICANT_SIDE_WATCH_RE = re.compile(
    r"\b("
    r"open grant proposal|devgrant proposal|grant proposal|technical sponsor|proposer\b|"
    r"grant issue|proposal stage|submitted (?:an )?open"
    r")\b",
    flags=re.IGNORECASE,
)

_FINALIZER_FILECOIN_PROPOSAL_WATCH_RE = re.compile(
    r"\b("
    r"filecoin(?:\s+devgrants?)?|devgrants?|open grants?"
    r")\b.*\b("
    r"open grant proposal|grant proposal|devgrant proposal|proposal|technical sponsor|under review|github issue"
    r")\b|"
    r"\b("
    r"open grant proposal|grant proposal|devgrant proposal"
    r")\b.*\b("
    r"filecoin(?:\s+devgrants?)?|devgrants?"
    r")\b",
    flags=re.IGNORECASE,
)

_FINALIZER_COMMERCIAL_EARLY_KEEP_RE = re.compile(
    r"\b("
    r"migration|migrate|replace tally|replace .*provider|vendor|pricing|deployment|hosted service|"
    r"self-hosted|self hosted|hosted vs self-hosted|hosted vs self hosted|"
    r"implementation support|implementation partner|deployment support|"
    r"on-ramp|payment rail|treasury rail|settlement rail|fiat access|fiat on-ramp|frontend-only integration|integrat(?:e|ion|or)|"
    r"audit program|security audits?|subsid(?:ized|y)|provider discussion|technical review service|"
    r"operator collaboration|technical evaluation|public repo|open-source|playground|developer tooling|"
    r"\bjoin\b [^.!?\n]{0,40}\bplatform\b|"
    r"\bjoin\b [^.!?\n]{0,40}\becosystem\b|"
    r"\blaunch\b [^.!?\n]{0,40}\bon\b [^.!?\n]{0,30}\bplatform\b|"
    r"\bintegrate\b [^.!?\n]{0,40}\bwith\b"
    r")\b",
    flags=re.IGNORECASE,
)

_WATCHLIST_PROMOTION_COMMERCIAL_RE = re.compile(
    r"\b("
    r"migration|migrate|replace tally|replace .*provider|vendor|pricing|deployment|hosted service|"
    r"self-hosted|self hosted|hosted vs self-hosted|hosted vs self hosted|"
    r"implementation support|implementation partner|deployment support|"
    r"on-ramp|payment rail|treasury rail|settlement rail|fiat access|fiat on-ramp|frontend-only integration|integrat(?:e|ion|or)|"
    r"audit program|security audits?|subsid(?:ized|y)|provider discussion|implementation support|"
    r"migration process|technical review service|operator collaboration|"
    r"\bjoin\b [^.!?\n]{0,40}\bplatform\b|"
    r"\bjoin\b [^.!?\n]{0,40}\becosystem\b|"
    r"\bintegrate\b [^.!?\n]{0,40}\bwith\b|"
    r"\blaunch\b [^.!?\n]{0,40}\bon\b [^.!?\n]{0,30}\bplatform\b|"
    r"\buse\b [^.!?\n]{0,40}\bplatform\b"
    r")\b",
    flags=re.IGNORECASE,
)

_WATCHLIST_PROMOTION_DEPRIORITIZE_RE = re.compile(
    r"\b("
    r"filecoin(?:\s+devgrants?)?|open grant proposal|devgrant proposal|grant proposal|"
    r"technical sponsor|committee seats?|delegate|draft erc|gitter|feedback window"
    r")\b",
    flags=re.IGNORECASE,
)

_WATCHLIST_PROMOTION_PUBLIC_RE = re.compile(
    r"\b("
    r"permissionless|teams wanted|teams building|would implement|implementers?|"
    r"apps?, wallets?,? and communities|can participate|projects can submit|submit proposals?|"
    r"intake channel|integration requests?|pool listings?|revenue share|"
    r"subgraph owners?|owners? or maintainers?|must upgrade|migration guide|"
    r"custom assets? for listing|request .*listing"
    r")\b",
    flags=re.IGNORECASE,
)

_WATCHLIST_PROMOTION_INVERTED_RE = re.compile(
    r"\b("
    r"looking to sync with|looking to connect with|seeking to connect with|"
    r"requests? integration discussion|would love to get your thoughts|"
    r"engage early|please message me if youd like to chat|please message me if you'd like to chat|"
    r"feel free to message me|found your repo on github|"
    r"for (?:our|their) clients|complimentary security review|free security review|"
    r"serve as a primary execution partner"
    r")\b",
    flags=re.IGNORECASE,
)

_FINALIZER_FORMAL_INTAKE_KEEP_RE = re.compile(
    r"\b("
    r"apply now|applications? open|open application|call for proposals|requests? for proposals|"
    r"rfp|rfq|rfi|submit a proposal|partner intake|vendor intake|rolling grant program|"
    r"projects? in [a-z0-9 ._-]+ can join|join and launch|matrix channel|telegram\s*@|"
    r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
    r")\b",
    flags=re.IGNORECASE,
)

_FINALIZER_TECHNICAL_EVAL_KEEP_RE = re.compile(
    r"\b("
    r"public repo|open-source|testnet|playground|reference implementations?|documentation|docs|"
    r"technical evaluation|prototype integrations?|evaluate and prototype|developer tooling"
    r")\b",
    flags=re.IGNORECASE,
)

_FINALIZER_NOISE_CUT_RE = re.compile(
    r"\b("
    r"unconditional tld gifts?|claim by dming|gifted tlds?|art & social experiments|cultural practitioners|"
    r"social experiment concepts?"
    r")\b",
    flags=re.IGNORECASE,
)

_MEDIA_ROLLOUT_SOURCE_POSITIVE_RE = re.compile(
    r"(cointelegraph|coindesk|cryptobriefing|theblock|the block|cryptopotato|decrypt|blockworks|"
    r"beincrypto|bitcoinist|\bnews\b|\bblog\b|\bmedia\b)",
    flags=re.IGNORECASE,
)

_MEDIA_ROLLOUT_SOURCE_NEGATIVE_RE = re.compile(
    r"\b(forum|governance|github|gitlab|snapshot|magicians|community)\b",
    flags=re.IGNORECASE,
)

_MEDIA_ROLLOUT_PHRASE_RE = re.compile(
    r"\b(now live|expands to|instant conversion|seller acceptance|onchain for first time|auto-enables|enables|joins)\b|"
    r"plugs? [^.!?\n]{0,60}\binto\b|"
    r"bring(?:s)? [^.!?\n]{0,60}\bonchain\b",
    flags=re.IGNORECASE,
)

_MEDIA_ROLLOUT_DOMAIN_RE = re.compile(
    r"\b(payment stack|payments?|wallet|merchant(?:s)?|stablecoin(?: rails?)?|onchain index|"
    r"index onchain|integrat(?:e|ed|ion)|checkout|treasur(?:y|ies) index)\b",
    flags=re.IGNORECASE,
)

_MEDIA_ROLLOUT_RELATION_RE = re.compile(
    r"\b(joins?|with|into|to|and|plugs?|brings?|seller acceptance|sellers?|merchant(?:s)?|payments?)\b",
    flags=re.IGNORECASE,
)

_MEDIA_ROLLOUT_REJECT_RE = re.compile(
    r"\b(roadmap|proposal|grant|temp check|application|feedback request|discussion|vote|governance|"
    r"rfc|comment period|retrospective|builder'?s journal|pending dao approval|pending governance approval)\b",
    flags=re.IGNORECASE,
)


def _media_rollout_rescue_snippets(ev: Event) -> tuple[Optional[str], Optional[str]]:
    candidates: List[str] = []
    for part in [clean_text(ev.title), clean_text(ev.summary), clean_text(ev.description)]:
        if not part:
            continue
        if part not in candidates:
            candidates.append(part)
        for sentence in _SENT_SPLIT_RE.split(part):
            normalized = clean_text(sentence)
            if normalized and normalized not in candidates:
                candidates.append(normalized)

    evidence: Optional[str] = None
    supporting: Optional[str] = None
    for text in candidates:
        if _MEDIA_ROLLOUT_REJECT_RE.search(text):
            continue
        if _MEDIA_ROLLOUT_PHRASE_RE.search(text) and _MEDIA_ROLLOUT_DOMAIN_RE.search(text):
            evidence = enforce_verbatim_snippet(text, max_words=38)
            break

    if evidence:
        for text in candidates:
            if text == evidence or _MEDIA_ROLLOUT_REJECT_RE.search(text):
                continue
            if _MEDIA_ROLLOUT_DOMAIN_RE.search(text) or _MEDIA_ROLLOUT_PHRASE_RE.search(text):
                supporting = enforce_verbatim_snippet(text, max_words=22)
                if supporting:
                    break

    return evidence, supporting


def maybe_rescue_gatekeeper_media_rollout(ev: Event, gr: GateResult) -> Optional[GateResult]:
    """Very narrow deterministic rescue for recurring media rollout false negatives.

    This runs only on hard-excluded Gatekeeper rows and uses cheap local rules so
    cost and latency stay effectively unchanged.
    """
    if not gr.hard_exclusion:
        return None

    source_blob_parts: List[str] = [clean_text(ev.source) or ""]
    try:
        host = (urlparse(ev.url or "").netloc or "").lower()
    except Exception:
        host = ""
    if host:
        source_blob_parts.append(host)
    if ev.url:
        source_blob_parts.append(ev.url)
    source_blob = " ".join(p for p in source_blob_parts if p)
    if not _MEDIA_ROLLOUT_SOURCE_POSITIVE_RE.search(source_blob):
        return None
    if _MEDIA_ROLLOUT_SOURCE_NEGATIVE_RE.search(source_blob) and not _MEDIA_ROLLOUT_SOURCE_POSITIVE_RE.search(source_blob):
        return None

    headline = " ".join(
        p for p in [clean_text(ev.title) or "", clean_text(ev.summary) or ""] if p
    )
    if not headline:
        return None
    if _MEDIA_ROLLOUT_REJECT_RE.search(headline):
        return None
    if not _MEDIA_ROLLOUT_PHRASE_RE.search(headline):
        return None
    if not _MEDIA_ROLLOUT_DOMAIN_RE.search(headline):
        return None
    if not _MEDIA_ROLLOUT_RELATION_RE.search(headline):
        return None

    evidence_snippet, supporting_snippet = _media_rollout_rescue_snippets(ev)
    if not evidence_snippet:
        return None

    return GateResult(
        event_id=ev.event_id,
        hard_exclusion=False,
        hard_exclusion_reason=None,
        evidence_type="partnership_or_integration",
        evidence_snippet=evidence_snippet,
        supporting_snippet=supporting_snippet,
    )


def gatekeeper_postpass_veto_reason(ev: Event, gr: GateResult) -> Optional[str]:
    """Deterministic veto for recurring weak-signal false positives.

    Keep this intentionally narrow: only veto when a passed Gatekeeper item
    matches recurring junk patterns AND lacks a stronger buyer-side counter-signal.
    """
    if not gr.passes():
        return None

    haystack = " ".join(
        part
        for part in [
            clean_text(ev.title) or "",
            clean_text(ev.summary) or "",
            clean_text(ev.description) or "",
            clean_text(ev.source) or "",
            clean_text(gr.evidence_snippet) or "",
            clean_text(gr.supporting_snippet) or "",
        ]
        if part
    )

    if not haystack:
        return None

    if _GATEKEEPER_POSTPASS_FILECOIN_DEVGRANT_RE.search(haystack):
        if not _GATEKEEPER_POSTPASS_FILECOIN_BUYER_INTAKE_RE.search(haystack):
            return "filecoin_devgrant_without_buyer_intake"

    if _GATEKEEPER_POSTPASS_STRONG_COUNTERSIGNAL_RE.search(haystack):
        return None

    if _GATEKEEPER_POSTPASS_APPLICANT_SIDE_RE.search(haystack):
        return "applicant_side_grant_or_proposal"

    if _GATEKEEPER_POSTPASS_GOVERNANCE_FLOW_RE.search(haystack):
        return "governance_or_community_flow_without_buyer_path"

    if _GATEKEEPER_POSTPASS_FEEDBACK_ONLY_RE.search(haystack):
        return "feedback_or_discussion_without_execution_surface"

    if _GATEKEEPER_POSTPASS_ROLE_INVERTED_RE.search(haystack):
        return "weak_signal_without_buyer_path"

    if _GATEKEEPER_POSTPASS_WEAK_SIGNAL_RE.search(haystack):
        return "weak_signal_without_buyer_path"

    return None


def finder_obvious_guideline_cut_reason(row: Dict[str, Any]) -> Optional[str]:
    """Small pre-critic guardrail for obvious junk only.

    Keep this intentionally conservative so we do not lose strong/good candidates.
    """
    if is_strong_implicit_bd_candidate(row):
        return None

    haystack = near_miss_review_haystack(row)

    if re.search(
        r"\b(deadline passed|applications closed|submission period ended|closed on|expired|window is already closed)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return "closed_expired"

    if re.search(
        r"\b(waitlist|sign up|signup|try it|visit the site|self-serve|generic trial)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return "generic_signup"

    if re.search(
        r"\b(hackathon|contest|delegate race|nominations?|community translation|security council candidacy|"
        r"security council reelection|emergency contact|ops contact|admin registration|generic contributor intake)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return "admin_or_participation_only"

    if re.search(
        r"\b(retrospective|final report|monitoring only|governance discussion|temp check|comment period|roadmap call|"
        r"informational update|policy signal|macro|price move)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return "informational_or_governance_only"

    if re.search(
        r"\b(wrong actor|wrong role|applicant-side ask|seeking funding for itself|seeking support for itself)\b",
        haystack,
        flags=re.IGNORECASE,
    ):
        return "wrong_role"

    return None


def critic_feedback_should_go_to_near_miss(feedback: Dict[str, Any]) -> bool:
    audit_bucket = (feedback or {}).get("audit_bucket")
    reason_code = ((feedback or {}).get("reason_code") or "").strip().lower()
    if audit_bucket == "manual_review":
        return True
    if reason_code in {"ambiguous_but_salvageable", "inference_only", "media_recap_no_intake"}:
        return True
    return False


_CRITIC_SOFT_LIFT_EVIDENCE_TYPES = {
    "partnership_or_integration",
    "grant_or_rfp",
    "grant_or_rfp_or_program_open",
    "support_added_with_bd_action",
}


def evidence_type_matches_allowed(evidence_type: str, allowed: set[str]) -> bool:
    et = (evidence_type or "").strip()
    if not et:
        return False
    if et in allowed:
        return True
    # Treat composite evidence types like "grant_or_rfp_or_program_open" as a set of tokens.
    parts = [p.strip() for p in et.split("_or_") if p.strip()]
    return any(part in allowed for part in parts)


def should_soft_lift_critic_near_miss(row: Dict[str, Any], feedback: Dict[str, Any], min_confidence: float) -> bool:
    reason_code = ((feedback or {}).get("reason_code") or "").strip().lower()
    if reason_code != "ambiguous_but_salvageable":
        return False
    evidence_type = (row.get("evidence_type") or "").strip()
    if not evidence_type_matches_allowed(evidence_type, _CRITIC_SOFT_LIFT_EVIDENCE_TYPES):
        return False
    confidence = near_miss_effective_confidence(row)
    return confidence >= max(min_confidence, 0.6)


def append_signal_stage_note(text: Optional[str]) -> str:
    note = "Signal-stage; monitor for intake updates."
    base = (text or "").strip()
    if not base:
        return note
    if note.lower() in base.lower():
        return base
    if base[-1] not in ".!?":
        base += "."
    return f"{base} {note}"


def call_recovery_agent(
    client: OpenAI,
    model: str,
    candidates: List[Dict[str, Any]],
    category_catalog: List[Dict[str, str]],
    max_output_tokens: int,
    accepted_examples: Optional[List[Dict[str, Any]]] = None,
    tracker: Optional[TokenTracker] = None,
    force_one: bool = False,
    prefer_early_signals: bool = False,
    stronger_rescue_bias: bool = False,
    target_remaining: int = 0,
) -> List[Dict[str, Any]]:
    audit_today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    system_prompt = f"""You are a deterministic Dropped-Items Recovery Agent.
Today is {audit_today}. Do not recover anything whose window is already closed as of that date.

Your job is to review a SMALL set of borderline dropped candidates and rescue only the strongest real BD opportunities.

Recovery bar:
- Recover only when the evidence supports a real external action surface within {TIME_HORIZON_STR}.
- If unsure, do not recover.
- Prefer precision over creativity, but rescue obvious false negatives.
- Use all relevant provided fields when ranking the best rescues: title, summary, reason, body_preview,
  target_company, suggested_outreach_angle, evidence_type, evidence_snippet, supporting_snippet, reason_text,
  categories, bd_signal, feedback, sub_scores, source_event_ids, and source_events.
- Pick the best dropped candidates by total grounded strength, not by a single strong phrase in isolation.

    Patterns that may be recovered when explicitly grounded:
    - open application / intake threads
    - operator or provider admission / onboarding paths
    - paid services proposal / renewal / annual engagement surfaces
    - concrete integration partner / integrator build surfaces
    - audit / security review / implementation support / migration support / integration help requests
    - partner / distribution / incubator / co-build surfaces
    - named API / SDK / webhook / plugin / docs / platform adoption paths tied to a concrete business use case and clearly
      meant for external adopters, even without a formal intake email
    - stablecoin / liquidity rail / settlement rail / treasury / remittance adoption surfaces where the point is clearly
      external integration by wallets, apps, bridges, or payment operators
    - regional or institutional distribution / operator partnership surfaces, even if phrased as ecosystem expansion
    - ongoing grants/programs or recap posts that explicitly state the program remains live/ongoing now
    - public grant proposals under review ONLY when an external infra / audit / implementation / integration partner can
      realistically engage the proposing team now

    Strong rescue bias:
    - If a candidate clearly looks like a JAW.id / Crossmint / Safe module audit / Sonic USSD type of surface, prefer recover=true.
    - Do not require a formal intake email when the text clearly invites external integration, adoption, audit, distribution,
      or operator collaboration.

    Never recover these unless the text also exposes a clear commercial counterparty motion:
    - hackathons, contests, contributor programs, delegate races, community participation flows
    - emergency/admin registration or ops contact forms
    - applicant-side asks where the posting entity wants funding/approval/support for itself from the target ecosystem
    - governance/RFC/comment threads with no explicit funded or operational intake
    - generic signup, self-serve product trial, waitlist, or "visit the site and try it" language with no explicit
      enterprise, integration, partner, distributor, operator, auditor, or vendor path
    - closed or expired deadlines, ended submission periods, or retrospective-only surfaces

    Comparative rule:
- If a borderline candidate is at least as strong as the accepted_examples provided for this run, prefer recovery.
- Use accepted_examples as calibration for what already counts as a valid final opportunity.
- Do not demand stricter evidence from borderline candidates than from accepted_examples.

{"Early-signal promotion rule:\n- Strong early signals are allowed when they already expose a commercially legible path for an external builder, integrator, auditor, operator, distributor, or service provider.\n- Do not require a fully formal intake if the source clearly signals active buyer-side interest, technical evaluation, integration readiness, partner exploration, or implementation demand within the current time horizon.\n- Prefer recovery when the item is not publishable as a closed-form RFP yet, but is still strong enough to be surfaced now as a credible BD lead.\n- Do NOT use this to recover vague monitoring posts, generic discussions, or applicant-side asks with no external counterparty motion." if prefer_early_signals else ""}
{"Priority early-signal patterns:\n- buyer-side migration or replacement need (for example replacing an incumbent governance/vendor UI)\n- frontend or app-layer fiat on-ramp / payment / settlement integration opportunities\n- subsidized audit / security / provider programs that create a near-term vendor path\n- hosted service, deployment, pricing, or implementation discussions that imply an external vendor or integrator can act now\n- concrete integration or deployment discussions that already name the operational next step\n- broadly publishable public opportunities that multiple outside teams could act on should outrank named-party-only outreach or seller-led offers to one specific target when both are otherwise decent\n- When these are present, they should outrank applicant-side grant proposals, named-target-only outreach, and generic standards-feedback threads." if prefer_early_signals else ""}
{"Elevated drop-pressure rule:\n- This run has a higher-than-desired DROP/KEEP ratio, so use a stronger rescue bias for clear false negatives.\n- If a candidate is reasonably defensible, evidence-grounded, and at least strong enough to surface as an Early Signal, prefer recovery over a conservative cut.\n- Break close calls toward recovery when the evidence shows a real buyer-side need, active integration path, migration need, vendor surface, or early partner motion.\n- Do not use this rule to rescue obvious junk, applicant-side asks, or generic discussion threads.\n- Remaining recovery target for this pass: " + str(target_remaining) if stronger_rescue_bias else ""}

Rules:
- Use the provided evidence_snippet verbatim.
- Keep evidence_type grounded to the provided candidate.
- Preserve source_event_ids exactly as provided for the candidate.
- Do not invent new facts, entities, deadlines, or categories.
- If any important output field would be weak, generic, or unsupported (title, summary, reason, target_company,
  suggested_outreach_angle), do not recover.
- Generic "reach out / social channels / contact us" language is NOT enough on its own. Recover only when the source
  separately shows a concrete partner, integration, vendor, audit, operator, distribution, or grant motion.
- Recovery is a promotion step only.
- If recover=false, return opportunity=null.
- If recover=true, return a fully compliant opportunity object with is_opportunity=true.

{"If at least one candidate is reasonably defensible and evidence-grounded, recover the single strongest one. Do not recover more than one in this pass." if force_one else ""}

Return JSON only."""

    user_prompt = f"""CANDIDATES:
{json.dumps(candidates, ensure_ascii=False, default=_json_safe, separators=(",", ":"))}

ACCEPTED_EXAMPLES:
{json.dumps(accepted_examples or [], ensure_ascii=False, default=_json_safe, separators=(",", ":"))}

AVAILABLE CATEGORIES (use only category_text_id values):
{json.dumps(category_catalog, ensure_ascii=False, separators=(",", ":"))}
"""

    create_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_schema", "json_schema": openai_recovery_schema_v1()},
        "max_completion_tokens": max_output_tokens,
    }
    if model not in {"o3", "o3-mini"}:
        create_kwargs["temperature"] = 0

    resp = client.chat.completions.create(**create_kwargs)

    if tracker:
        tracker.add_usage(resp.usage, model, agent="Recovery")

    raw = (resp.choices[0].message.content or "").strip()
    parsed = json.loads(raw) if raw else {}
    return parsed.get("decisions", []) or []


def write_near_miss_watchlist_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """CSV companion for near-miss manual review."""
    fieldnames = [
        "stage",
        "decision",
        "reason",
        "batch_num",
        "title",
        "summary",
        "body_preview",
        "target_company",
        "categories",
        "suggested_outreach_angle",
        "confidence",
        "min_confidence",
        "evidence_type",
        "evidence_snippet",
        "supporting_snippet",
        "source_event_ids",
        "reason_text",
        "bd_signal",
        "feedback",
        "sub_scores",
        "decision_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = dict(r)
            if isinstance(row.get("source_event_ids"), list):
                row["source_event_ids"] = "|".join(str(x) for x in row["source_event_ids"])
            if isinstance(row.get("categories"), list):
                row["categories"] = ",".join(str(x) for x in row["categories"])
            if isinstance(row.get("sub_scores"), dict):
                row["sub_scores"] = json.dumps(row["sub_scores"], ensure_ascii=False)
            if isinstance(row.get("bd_signal"), dict):
                row["bd_signal"] = json.dumps(row["bd_signal"], ensure_ascii=False)
            row["decision_reason"] = clean_text(row.get("reason")) or ""
            w.writerow({k: row.get(k, "") for k in fieldnames})


def build_opportunity_from_raw_candidate(
    ro: Dict[str, Any],
    event_lookup: Dict[str, Event],
    categories_map: Dict[str, str],
    category_definitions: Dict[str, str],
) -> Tuple[Optional[Opportunity], Optional[str]]:
    event_ids_list = normalize_source_event_ids(ro.get("source_event_ids"))
    if not event_ids_list:
        return None, "missing_source_event_ids"

    source_events = [event_lookup[eid] for eid in event_ids_list if eid in event_lookup]
    if not source_events:
        return None, "missing_source_events"

    sources: List[str] = []
    event_titles: List[str] = []
    event_urls: List[str] = []
    for ev in source_events:
        if ev.url:
            sources.append(f"{ev.source}: {ev.url}" if ev.source else ev.url)
            event_urls.append(ev.url)
        elif ev.source:
            sources.append(ev.source)
        if ev.title:
            event_titles.append(ev.title)

    if not event_titles:
        return None, "missing_event_titles"

    evidence_type = ro.get("evidence_type")
    evidence_snippet = enforce_verbatim_snippet(ro.get("evidence_snippet")) or ""
    supporting_snippet = enforce_verbatim_snippet(ro.get("supporting_snippet"), max_words=30) or ""
    title = (ro.get("title") or "").strip()
    target = (ro.get("target_company") or "").strip()

    mega = is_mega_counterparty(target) or is_mega_counterparty(title)
    if mega and not has_open_motion(evidence_type, evidence_snippet, title=title, target_company=target):
        return None, "mega_counterparty_without_open_motion"

    validated_categories = validate_categories(
        ro.get("categories", []),
        evidence_type,
        evidence_snippet,
        categories_map,
        category_definitions,
    )
    if not validated_categories:
        return None, "no_valid_categories_after_guardrails"

    reason_block = build_reason_block(
        normalize_reason_text(clean_text(ro.get("reason", "")) or ""),
        evidence_type,
        clean_text(evidence_snippet) or "",
        clean_text(supporting_snippet) or "",
    )
    recommended_action = clean_text(ro.get("recommended_action") or "") or ""
    suggested_outreach_angle = clean_text(ro.get("suggested_outreach_angle") or "") or ""
    tc = normalize_target_company(ro.get("target_company"))
    if not tc:
        tc = derive_target_company(event_urls, sources)
    tc = clean_text(tc)

    opport = Opportunity(
        opportunity_id=str(uuid.uuid4()),
        title=normalize_title_text(ro.get("title", "")),
        summary=normalize_summary_text(
            sanitize_opportunity_field_text(ro.get("summary", "")),
            [
                sanitize_opportunity_field_text(ro.get("reason", "")),
                sanitize_opportunity_field_text(evidence_snippet),
                sanitize_opportunity_field_text(supporting_snippet),
            ],
        ),
        opportunity_details=None,
        reason=reason_block,
        who_to_contact=sanitize_opportunity_field_text(ro.get("who_to_contact")),
        suggested_outreach_angle=normalize_outreach_text(suggested_outreach_angle or recommended_action),
        categories=validated_categories,
        filter_chain="chain-oth",
        filter_sector="sect-oth",
        filter_seeking="seek-oth",
        time_found=datetime.now(timezone.utc).date().isoformat(),
        confidence=float(ro.get("confidence", 0)),
        tags=[sanitize_opportunity_field_text(t) for t in (ro.get("tags", []) or []) if sanitize_opportunity_field_text(t)],
        target_company=sanitize_opportunity_field_text(tc),
        sources=[sanitize_opportunity_field_text(s) for s in sources if sanitize_opportunity_field_text(s)],
        event_ids="|".join(event_ids_list),
        event_titles="|".join(sanitize_opportunity_field_text(t) or "" for t in event_titles),
        event_url="|".join(event_urls),
        bd_weeks=int(TIME_HORIZON_WEEKS),
        evidence_type=normalize_key(evidence_type),
        evidence_snippet=clean_text(evidence_snippet) or "",
        supporting_snippet=clean_text(supporting_snippet) or "",
    )
    return opport, None


# ---------------------------
# Agent: Filter

# ---------------------------

def call_filter_agent(
    client: OpenAI,
    model: str,
    opp_batch: List[Opportunity],
    event_map: Dict[str, Event],
    chains: Dict[str, str],
    sectors: Dict[str, str],
    seeking: Dict[str, str],
    tracker: Optional[TokenTracker] = None,
):
    payload = []

    for o in opp_batch:
        first_event_id = (o.event_ids or "").split("|")[0]
        ev = event_map.get(first_event_id)
        body_slice = slice_body_for_finder(ev.description) if ev else ""

        payload.append({
            "opportunity_id": o.opportunity_id,
            "title": o.title,
            "summary": o.summary,
            "reason": o.reason,
            "suggested_outreach_angle": o.suggested_outreach_angle,
            "event_context": body_slice,
        })

    system_prompt = """
You are a strict filter classification agent for business development targeting.
Return JSON ONLY. No explanations.

Ignore any bd_signal metadata if present; classify only from opportunity text and event_context.

TASK:
For each opportunity, classify into:
  - filter_chain: list of chain IDs
  - filter_sector: list of sector IDs
  - filter_seeking: list of seeking IDs

ALLOWED IDS (use ONLY these; never invent new IDs):
filter_chain: ["chain-evm","chain-sol","chain-bnb","chain-cosmos","chain-bit","chain-multi","chain-oth"]
filter_sector: ["sect-defi","sect-infra","sect-pay","sect-rwa","sect-zk","sect-ai","sect-oth"]
filter_seeking: ["seek-infra","seek-data","seek-bridge","seek-dex","seek-wallet","seek-zk","seek-ai","seek-rwa","seek-secur","seek-maker","seek-launch","seek-oth"]

DEFINITIONS:
- filter_chain = which ecosystem the opportunity primarily operates on or requires compatibility with.
- filter_sector = what domain the TARGET company operates in (their core business identity). This is stable and does not change opportunity to opportunity.
- filter_seeking = which partner/vendor capability is most relevant for BD outreach based on the specific evidence in this opportunity. This is situational.

STRICT RULES:

1) Allowed IDs only.
   Never return an ID not listed above. If uncertain, use the appropriate Other ID (chain-oth, sect-oth, seek-oth).

2) chain-multi usage.
   Use chain-multi ONLY if the opportunity explicitly involves cross-chain, bridge, or interoperability requirements,
   OR explicitly states compatibility with 2+ named ecosystems as a requirement.
   If the company is broadly multi-chain but this opportunity is ecosystem-specific, tag the primary ecosystem only.
   If the primary ecosystem is unclear and it is not a cross-chain motion, use chain-oth.

3) Sector vs Seeking separation.
   Sector = what the target company IS.
   Seeking = what this specific opportunity requires for BD outreach.
   These must answer different questions.
   If you are using the same reasoning for both, one of them is wrong.
   Tag sect-ai ONLY if the target company's primary business is AI.
   Incidental AI mentions do not qualify.

4) Evidence threshold for filter_seeking.
   Assign a seeking tag ONLY if the opportunity text contains at least one concrete cue supporting that partner category.
   Do not infer seeking tags from what the company might broadly need.
   The cue must be in the opportunity text.
   If no concrete cue exists, use seek-oth.

5) Tag caps.
   filter_chain: max 1 (max 2 only if chain-multi is included).
   filter_sector: max 1 (max 2 only if genuinely dual-domain with clear evidence for both).
   filter_seeking: max 2 (max 3 only if the opportunity text explicitly describes multiple distinct BD needs).

OUTPUT SCHEMA:
{
  "filter_chain": ["chain-evm"],
  "filter_sector": ["sect-defi"],
  "filter_seeking": ["seek-dex"]
}
"""

    user_prompt = f"""
OPPORTUNITIES:
{json.dumps(payload, default=_json_safe)}

FILTER_CHAINS:
{json.dumps(chains)}

FILTER_SECTORS:
{json.dumps(sectors)}

FILTER_SEEKING:
{json.dumps(seeking)}

Return JSON.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": openai_filter_schema()},
        temperature=0,
        max_completion_tokens=2000,
    )

    if tracker:
        tracker.add_usage(resp.usage, model, agent="Filter")

    return json.loads(resp.choices[0].message.content).get("results", [])



# ---------------------------
# Post-processing
# ---------------------------

def normalize_key(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s



def normalize_categories_list(raw_categories: Any, categories_map: Dict[str, str]) -> List[str]:
    """Normalize a list of category tokens into canonical category_text_id values.

    IMPORTANT (fail-closed):
    - Only return sectors that exist in category_master-derived categories_map values.
    - Any unknown/unmappable token is DROPPED (prevents LLM category hallucinations).

    Notes:
    - categories_map is expected to map {category_id_or_alias: category_text_id}
      (e.g., "11" -> "tech", "Infrastructure ..." -> "tech", "tech" -> "tech").
    """
    if not raw_categories:
        return []

    # Ensure list-like
    if isinstance(raw_categories, str):
        cats = [raw_categories]
    else:
        try:
            cats = list(raw_categories)
        except Exception:
            cats = [str(raw_categories)]

    # Precompute normalized maps for robust matching
    norm_key_to_sector: Dict[str, str] = {normalize_key(k): v for k, v in (categories_map or {}).items() if k}

    # Canonical sectors (normalized) -> canonical sector
    norm_sector_to_sector: Dict[str, str] = {}
    for v in (categories_map or {}).values():
        vv = (v or "").strip()
        if vv:
            norm_sector_to_sector[normalize_key(vv)] = vv

    allowed_sectors = set(norm_sector_to_sector.values())

    out: List[str] = []
    seen: set = set()
    dropped: List[str] = []

    for tok in cats:
        s = str(tok or "").strip()
        if not s:
            continue

        mapped: Optional[str] = None

        # 1) Direct key match (exact)
        if categories_map:
            mapped = categories_map.get(s)

        # 2) Normalized key match
        if not mapped:
            mapped = norm_key_to_sector.get(normalize_key(s))

        # 3) Curated fallback aliases for common non-canonical LLM labels
        if not mapped:
            mapped = CATEGORY_TOKEN_ALIASES.get(normalize_key(s))

        # 4) If already a sector, keep canonical casing; otherwise DROP (fail-closed)
        if not mapped:
            mapped = norm_sector_to_sector.get(normalize_key(s))

        mapped = (mapped or "").strip()

        # Fail-closed: only allow known sectors
        if not mapped or mapped not in allowed_sectors:
            dropped.append(s)
            continue

        if mapped not in seen:
            seen.add(mapped)
            out.append(mapped)

    # Log once if model emitted unknown tokens (keeps pipeline deterministic)
    if dropped:
        print(f"[OpportunityMatcher] Dropped unknown category tokens from LLM output: {sorted(set(dropped))}")

    return out


def _category_def_overlap_score(definition: str, evidence_snippet: str) -> int:
    """Simple definition-evidence alignment score.

    We intentionally keep this cheap/deterministic: token overlap count.
    """
    def_toks = set(re.sub(r"[^a-z0-9 ]+", " ", normalize_key(definition or "")).split())
    snip_toks = set(re.sub(r"[^a-z0-9 ]+", " ", normalize_key(evidence_snippet or "")).split())
    if not def_toks or not snip_toks:
        return 0
    return len(def_toks & snip_toks)


def validate_categories(
    raw_categories: Any,
    evidence_type: str,
    evidence_snippet: str,
    categories_map: Dict[str, str],
    category_definitions: Dict[str, str],
) -> List[str]:
    """Apply category guardrails:

    1) Normalize + fail-closed unknown drop (via normalize_categories_list)
    2) Evidence-type alignment (ALLOWED_BY_EVIDENCE)
    3) Strict 'gener' rule
    4) Definition-consistency enforcement using overlap score
    5) Cardinality guardrail (keep top MAX_CATEGORIES_PER_OPPORTUNITY by score)
    """
    cats = normalize_categories_list(raw_categories, categories_map)

    if not cats:
        return []

    ev = (evidence_type or "").strip()

    # (2) Evidence-type alignment
    allowed = ALLOWED_BY_EVIDENCE.get(ev)
    if allowed:
        cats = [c for c in cats if c in allowed]
        if not cats:
            return []

    # (3) Strict gener
    if "gener" in cats and ev not in STRICT_GENER_ALLOWED_EVIDENCE:
        cats = [c for c in cats if c != "gener"]
        if not cats:
            return []

    # (4) Definition consistency enforcement:
    # Require a minimum overlap with the category definition (when definition exists).
    scored: List[Tuple[int, str]] = []
    for c in cats:
        definition = (category_definitions.get(c) or "").strip()
        if not definition:
            # If definition is missing, be conservative: allow but score 0 (will lose ties).
            score = 0
        else:
            score = _category_def_overlap_score(definition, evidence_snippet)
            if score <= 0:
                # Fail-closed when a definition exists but we cannot align to the evidence snippet.
                continue
        scored.append((score, c))

    if not scored:
        # Last resort: only allow gener if explicitly permitted.
        if "gener" in cats and ev in STRICT_GENER_ALLOWED_EVIDENCE:
            return ["gener"]
        return []

    # Stable sort: higher score first, then deterministic lexicographic tiebreak.
    scored.sort(key=lambda x: (-x[0], x[1]))

    # (5) Cardinality: keep top N
    out = []
    for _, c in scored:
        if c not in out:
            out.append(c)
        if len(out) >= int(MAX_CATEGORIES_PER_OPPORTUNITY):
            break

    # If nothing remains, optional fallback to gener only if allowed
    if not out and ev in STRICT_GENER_ALLOWED_EVIDENCE:
        out = ["gener"]

    return out

def normalize_primary_category(raw_categories: Any, categories_map: Dict[str, str]) -> str:
    """Return the first (primary) category as a canonical name, or empty string."""
    cats = normalize_categories_list(raw_categories, categories_map)
    return cats[0] if cats else ""

EVIDENCE_TYPE_LINE_RE = re.compile(r"^Evidence type:\s*(.+)$", re.MULTILINE)
EVIDENCE_SNIPPET_LINE_RE = re.compile(r"^Evidence snippet:\s*(.+)$", re.MULTILINE)
SUPPORTING_SNIPPET_LINE_RE = re.compile(r"^Supporting snippet:\s*(.+)$", re.MULTILINE)
REASON_LINE_RE = re.compile(r"^Reason:\s*(.+)$", re.MULTILINE)


def extract_evidence_type_from_reason(reason_block: str) -> str:
    """Extract evidence type from the structured reason block (best-effort)."""
    m = EVIDENCE_TYPE_LINE_RE.search(reason_block or "")
    if not m:
        return ""
    return normalize_key(m.group(1))


def extract_evidence_snippet_from_reason(reason_block: str) -> str:
    """Extract evidence snippet from the structured reason block (best-effort)."""
    m = EVIDENCE_SNIPPET_LINE_RE.search(reason_block or "")
    if not m:
        return ""
    return (m.group(1) or "").strip()


def extract_supporting_snippet_from_reason(reason_block: str) -> str:
    """Extract supporting snippet from the structured reason block (best-effort)."""
    m = SUPPORTING_SNIPPET_LINE_RE.search(reason_block or "")
    if not m:
        return ""
    return (m.group(1) or "").strip()


def extract_reason_text_from_reason_block(reason_block: str) -> str:
    """Extract the human-readable reason from the structured reason block."""
    m = REASON_LINE_RE.search(reason_block or "")
    if m:
        return (m.group(1) or "").strip()
    return (reason_block or "").strip()


def opportunity_evidence_type(opp: Opportunity) -> str:
    stored = normalize_key(getattr(opp, "evidence_type", "") or "")
    if stored:
        return stored
    return extract_evidence_type_from_reason(opp.reason)


def opportunity_evidence_snippet(opp: Opportunity) -> str:
    stored = enforce_verbatim_snippet(getattr(opp, "evidence_snippet", ""))
    if stored:
        return stored
    return enforce_verbatim_snippet(extract_evidence_snippet_from_reason(opp.reason)) or ""


def opportunity_supporting_snippet(opp: Opportunity) -> str:
    stored = enforce_verbatim_snippet(getattr(opp, "supporting_snippet", ""), max_words=30)
    if stored:
        return stored
    return enforce_verbatim_snippet(extract_supporting_snippet_from_reason(opp.reason), max_words=30) or ""


def _normalize_string_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, tuple):
        values = list(raw)
    elif isinstance(raw, str):
        if "|" in raw:
            values = raw.split("|")
        elif "," in raw:
            values = raw.split(",")
        else:
            values = [raw]
    else:
        values = [raw]

    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        text = sanitize_opportunity_field_text(str(v) if v is not None else None)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def enrich_rejection_audit_rows(
    rows: List[Dict[str, Any]],
    events: List[Event],
    gate_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fill common audit context so the JSON is useful for manual review."""
    if not rows:
        return []

    event_lookup = {ev.event_id: ev for ev in events if ev.event_id}
    gate_lookup = {
        str(row.get("event_id")): row
        for row in (gate_results or [])
        if isinstance(row, dict) and row.get("event_id")
    }

    enriched: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row or {})

        event_id = clean_text(item.get("event_id"))
        source_event_ids = _normalize_string_list(item.get("source_event_ids"))
        if not source_event_ids:
            source_event_ids = _normalize_string_list(item.get("event_ids"))
        if not source_event_ids and event_id:
            source_event_ids = [event_id]
        if not event_id and source_event_ids:
            event_id = source_event_ids[0]
        if event_id:
            item["event_id"] = event_id
        item["source_event_ids"] = source_event_ids

        resolved_events = [event_lookup[eid] for eid in source_event_ids if eid in event_lookup]
        source_titles = [sanitize_opportunity_field_text(ev.title) for ev in resolved_events if sanitize_opportunity_field_text(ev.title)]
        source_urls = [sanitize_opportunity_field_text(ev.url) for ev in resolved_events if sanitize_opportunity_field_text(ev.url)]
        source_sources = [sanitize_opportunity_field_text(ev.source) for ev in resolved_events if sanitize_opportunity_field_text(ev.source)]

        if not clean_text(item.get("event_title")) and source_titles:
            item["event_title"] = source_titles[0]
        if not clean_text(item.get("title")):
            item["title"] = clean_text(item.get("event_title")) or (source_titles[0] if source_titles else "")

        item["source_event_titles"] = source_titles
        item["source_urls"] = source_urls
        item["source_sources"] = source_sources

        categories = _normalize_string_list(item.get("categories"))
        if not categories:
            categories = _normalize_string_list(item.get("raw_categories"))
        item["categories"] = categories

        gate_row = gate_lookup.get(event_id or "")
        if not gate_row and source_event_ids:
            gate_row = gate_lookup.get(source_event_ids[0])
        if gate_row:
            if not clean_text(item.get("evidence_type")):
                item["evidence_type"] = clean_text(gate_row.get("evidence_type")) or ""
            if not clean_text(item.get("evidence_snippet")):
                item["evidence_snippet"] = clean_text(gate_row.get("evidence_snippet")) or ""
            if not clean_text(item.get("supporting_snippet")):
                item["supporting_snippet"] = clean_text(gate_row.get("supporting_snippet")) or ""
            if not clean_text(item.get("hard_exclusion_reason")):
                item["hard_exclusion_reason"] = normalize_hard_exclusion_reason(
                    gate_row.get("hard_exclusion_reason"),
                    clean_text(gate_row.get("evidence_type")) or "none",
                )

        reason_summary = clean_text(item.get("reason_summary"))
        if not reason_summary:
            for candidate in [
                item.get("reason"),
                item.get("feedback"),
                item.get("hard_exclusion_reason"),
                item.get("decision"),
                item.get("stage"),
            ]:
                reason_summary = clean_text(candidate)
                if reason_summary and normalize_key(reason_summary) in {"true", "false", "hard_exclusion=true", "hard_exclusion=false"}:
                    reason_summary = normalize_hard_exclusion_reason(
                        item.get("hard_exclusion_reason"),
                        item.get("evidence_type"),
                    )
                if reason_summary:
                    break
        item["reason_summary"] = smart_truncate(reason_summary or "", 320)
        item["final_status"] = clean_text(item.get("final_status")) or clean_text(item.get("decision")) or "rejected"
        item["review_status"] = clean_text(item.get("review_status")) or clean_text(item.get("stage")) or ""

        enriched.append(item)

    return enriched


def dedupe_opportunities(opps: List[Opportunity]) -> List[Opportunity]:
    """Deduplicate opportunities.

    Client feedback: duplicates erode pilot credibility. We dedupe more aggressively and later (post-structuring):
    1) Exact/near-exact duplicates by overlapping event_ids (any overlap) â†’ keep highest confidence.
    2) Fallback key by (target_company + normalized title) â†’ keep highest confidence.
    3) Strong key by (target_company + primary category + evidence_type + normalized action) â†’ keep highest confidence.
    """
    if not opps:
        return []

    # 1) event_id overlap dedupe
    kept: List[Opportunity] = []
    seen_event_sets: List[set] = []

    for o in sorted(opps, key=lambda x: x.confidence, reverse=True):
        o_set = set((o.event_ids or "").split("|")) if o.event_ids else set()
        is_dup = False
        for s in seen_event_sets:
            if o_set and s and (o_set & s):
                is_dup = True
                break
        if not is_dup:
            kept.append(o)
            seen_event_sets.append(o_set)

    # 2) target_company + normalized title fallback
    best_title: Dict[str, Opportunity] = {}
    for o in kept:
        key = f"{normalize_key(o.target_company or '')}||{normalize_key(o.title)}"
        if key not in best_title or o.confidence > best_title[key].confidence:
            best_title[key] = o

    # 3) target_company + category + evidence_type + action (late-stage structured dedup)
    best_structured: Dict[str, Opportunity] = {}
    for o in best_title.values():
        primary_cat = normalize_key((o.categories or [""])[0])
        ev_type = opportunity_evidence_type(o)
        action = normalize_key(o.suggested_outreach_angle or "")
        if len(action) > 120:
            action = action[:120]
        key = f"{normalize_key(o.target_company or '')}||{primary_cat}||{ev_type}||{action}"
        if key not in best_structured or o.confidence > best_structured[key].confidence:
            best_structured[key] = o

    return list(best_structured.values())


# ---------------------------
# Finalizer (post-processing) â€” keep 5-agent logic intact
# ---------------------------

def _tokenize_simple(text: str) -> List[str]:
    t = normalize_key(text or "")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    stop = {"the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with", "by", "from", "as", "at", "into", "via"}
    return [x for x in t.split() if x and x not in stop]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _merge_pipe_fields(a: str, b: str) -> str:
    """Union two pipe-separated strings preserving order (best-effort)."""
    seen: set = set()
    out: List[str] = []
    for part in (a or "").split("|"):
        p = part.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    for part in (b or "").split("|"):
        p = part.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "|".join(out)


def _merge_sources(a: List[str], b: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for s in (a or []):
        ss = (s or "").strip()
        if ss and ss not in seen:
            seen.add(ss)
            out.append(ss)
    for s in (b or []):
        ss = (s or "").strip()
        if ss and ss not in seen:
            seen.add(ss)
            out.append(ss)
    return out


def _consolidate_similar_opportunities(opps: List[Opportunity]) -> List[Opportunity]:
    """Consolidate near-duplicates after the main dedupe step.

    Goal: merge cases like 'Newrez' where two distinct URLs/events describe the same opportunity.
    We only merge within the same (target_company + primary_category) bucket and require
    meaningful similarity between the recommended actions or titles.
    """
    if not opps:
        return []

    buckets: Dict[str, List[Opportunity]] = {}
    for o in opps:
        primary_cat = normalize_key((o.categories or [""])[0])
        key = f"{normalize_key(o.target_company or '')}||{primary_cat}"
        buckets.setdefault(key, []).append(o)

    consolidated: List[Opportunity] = []
    for _, items in buckets.items():
        # Sort high confidence first; we keep the strongest record as the base.
        items = sorted(items, key=lambda x: x.confidence, reverse=True)
        used = [False] * len(items)
        for i, base in enumerate(items):
            if used[i]:
                continue
            used[i] = True
            base_action_toks = _tokenize_simple(base.suggested_outreach_angle)
            base_title_toks = _tokenize_simple(base.title)
            base_ev_type = extract_evidence_type_from_reason(base.reason)
            base_ev_snip = extract_evidence_snippet_from_reason(base.reason)

            merged = base
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                cand = items[j]

                # Only merge if evidence types match OR the action similarity is high.
                cand_ev_type = extract_evidence_type_from_reason(cand.reason)
                action_sim = _jaccard(base_action_toks, _tokenize_simple(cand.suggested_outreach_angle))
                title_sim = _jaccard(base_title_toks, _tokenize_simple(cand.title))

                should_merge = False
                if base_ev_type and cand_ev_type and base_ev_type == cand_ev_type and (action_sim >= 0.45 or title_sim >= 0.55):
                    should_merge = True
                elif action_sim >= 0.60:
                    should_merge = True
                elif title_sim >= 0.75:
                    should_merge = True

                if not should_merge:
                    continue

                #& Merge cand into merged (keep merged's title/action; union sources/event ids/titles)
                merged = Opportunity(
                    opportunity_id=merged.opportunity_id,
                    title=merged.title or cand.title,
                    summary=merged.summary or cand.summary,
                    reason=_merge_reason_blocks(merged.reason, cand.reason),
                    who_to_contact=merged.who_to_contact or cand.who_to_contact,
                    suggested_outreach_angle=merged.suggested_outreach_angle or cand.suggested_outreach_angle,
                    categories=list(dict.fromkeys((merged.categories or []) + (cand.categories or []))),
                    #~ >>> ADD THESE THREE LINES <<<
                    filter_chain=merged.filter_chain or cand.filter_chain,
                    filter_sector=merged.filter_sector or cand.filter_sector,
                    filter_seeking=merged.filter_seeking or cand.filter_seeking,
                    time_found=merged.time_found,
                    confidence=max(merged.confidence, cand.confidence),
                    tags=list(dict.fromkeys((merged.tags or []) + (cand.tags or []))),
                    target_company=merged.target_company or cand.target_company,
                    sources=_merge_sources(merged.sources, cand.sources),
                    event_ids=_merge_pipe_fields(merged.event_ids, cand.event_ids),
                    event_titles=_merge_pipe_fields(merged.event_titles, cand.event_titles),
                    event_url=_merge_pipe_fields(merged.event_url, cand.event_url),
                    bd_weeks=merged.bd_weeks or cand.bd_weeks,
                    evidence_type=merged.evidence_type or cand.evidence_type,
                    evidence_snippet=merged.evidence_snippet or cand.evidence_snippet,
                    supporting_snippet=merged.supporting_snippet or cand.supporting_snippet,
                    opportunity_details=merged.opportunity_details or cand.opportunity_details,
                    audit_label=merged.audit_label or cand.audit_label,
                )
                used[j] = True

            consolidated.append(merged)

    # Preserve a stable order for downstream: confidence desc
    consolidated = sorted(consolidated, key=lambda x: x.confidence, reverse=True)
    return consolidated


def _merge_reason_blocks(a: str, b: str, max_extra_chars: int = 600) -> str:
    """Merge two structured reason blocks without bloating.

    We keep the 'a' block intact and append a small 'Additional evidence' section if b contains
    a different evidence snippet or source detail.
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a

    a_snip = extract_evidence_snippet_from_reason(a)
    b_snip = extract_evidence_snippet_from_reason(b)
    if b_snip and b_snip != a_snip and b_snip not in a:
        extra = f"\nAdditional evidence snippet: {enforce_verbatim_snippet(b_snip, max_words=22) or b_snip}"
        if len(extra) > max_extra_chars:
            extra = extra[:max_extra_chars]
        return a + extra
    return a


def finalize_opportunities(opps: List[Opportunity]) -> Tuple[List[Opportunity], List[Opportunity], List[Opportunity]]:
    """Post-processing Finalizer.

    - Consolidate near-duplicates (e.g., Newrez showing twice across different sources).
    - Apply a deterministic retain-or-cut policy without modifying any agent logic.
    - All retained rows stay in KEEP and carry an audit label for later manual review.

    Returns: (keep, watchlist, dropped)
    """
    opps = _consolidate_similar_opportunities(opps)

    keep: List[Opportunity] = []
    watch: List[Opportunity] = []
    drop: List[Opportunity] = []

    def _set_audit_label(opp: Opportunity, label: str) -> Opportunity:
        opp.audit_label = label
        if label != "Cut":
            opp.finalizer_reason = None
        return opp

    def _set_drop_reason(opp: Opportunity, reason_code: str) -> Opportunity:
        opp.audit_label = "Cut"
        opp.finalizer_reason = reason_code
        return opp

    def _retained_label(
        *,
        has_formal_intake: bool,
        has_technical_eval: bool,
        has_commercial_early_keep: bool,
        has_inverted_outreach: bool,
        is_signal_stage: bool,
        is_applicant_side: bool,
        governance_like: bool,
        filecoin_like: bool,
    ) -> str:
        if has_inverted_outreach:
            return "Inverted Outreach"
        if is_signal_stage or is_applicant_side or filecoin_like:
            return "Early Signal"
        if governance_like or has_technical_eval or has_commercial_early_keep:
            return "Keep & Reframe"
        if has_formal_intake:
            return "Keep"
        return "Keep & Reframe"

    for o in opps:
        primary_cat = (o.categories or [""])[0]
        cat_norm = normalize_key(primary_cat)
        ev_type = opportunity_evidence_type(o)
        ev_snip = opportunity_evidence_snippet(o)
        reason_text = extract_reason_text_from_reason_block(o.reason)
        watch_haystack = " ".join(
            part
            for part in [
                o.title or "",
                o.summary or "",
                o.event_titles or "",
                reason_text,
                ev_snip,
                o.suggested_outreach_angle or "",
                o.target_company or "",
            ]
            if part
        )
        has_formal_intake = bool(_FINALIZER_FORMAL_INTAKE_KEEP_RE.search(watch_haystack))
        has_technical_eval = bool(_FINALIZER_TECHNICAL_EVAL_KEEP_RE.search(watch_haystack))
        has_commercial_early_keep = bool(_FINALIZER_COMMERCIAL_EARLY_KEEP_RE.search(watch_haystack))
        has_inverted_outreach = bool(_WATCHLIST_PROMOTION_INVERTED_RE.search(watch_haystack))
        is_signal_stage = bool(_FINALIZER_SIGNAL_STAGE_WATCH_RE.search(watch_haystack))
        is_applicant_side = bool(_FINALIZER_APPLICANT_SIDE_WATCH_RE.search(watch_haystack))
        has_real_surface = has_formal_intake or has_technical_eval or has_open_motion(
            ev_type, ev_snip, title=o.title, target_company=o.target_company or ""
        )
        strong_actionable_ev = ev_type in {"vendor_or_procurement_need", "partnership_or_integration"}
        target_text = clean_text(o.target_company or "")
        has_named_target = bool(
            target_text
            and len(target_text) >= 4
            and not re.search(
                r"\b(unnamed|unknown|community|ecosystem|forum|contributors?|users?)\b",
                target_text,
                flags=re.IGNORECASE,
            )
        )
        has_public_icp = bool(_WATCHLIST_PROMOTION_PUBLIC_RE.search(watch_haystack)) or bool(
            re.search(
                r"\b(builders?|integrators?|wallets?|providers?|operators?|maintainers?|subgraph owners?|projects?|teams?|apps?)\b",
                watch_haystack,
                flags=re.IGNORECASE,
            )
        )
        has_clear_buyer = has_named_target or has_public_icp or has_inverted_outreach
        weak_surface = not (
            has_real_surface
            or has_commercial_early_keep
            or strong_actionable_ev
            or has_inverted_outreach
        )
        # Product policy: KEEP is the main retained bucket and should include
        # strong early signals and private/inverted-outreach opportunities too.
        # WATCHLIST is intentionally narrower: use it only for plausible-but-still
        # weak/ambiguous rows that need more evidence before we retain them as opportunities.
        allow_consolidated_keep = has_real_surface and (
            has_formal_intake
            or has_commercial_early_keep
            or has_inverted_outreach
            or strong_actionable_ev
            or is_signal_stage
            or is_applicant_side
        )

        # Clear non-report items should be cut rather than reviewed.
        if _FINALIZER_NOISE_CUT_RE.search(watch_haystack):
            drop.append(_set_drop_reason(o, "noise_pattern_cut"))
            continue

        # Drop policy-heavy items that lack a concrete, near-term BD hook.
        if ("regulat" in cat_norm or "policy" in cat_norm or "compliance" in cat_norm or ev_type == "regulatory_or_compliance_update"):
            mega = is_mega_counterparty(o.target_company or "") or is_mega_counterparty(o.title)
            if mega and not has_open_motion(ev_type, ev_snip, title=o.title, target_company=o.target_company or ""):
                drop.append(_set_drop_reason(o, "policy_mega_no_open_motion"))
                continue

        # Funding-only items without a real external surface should be cut.
        if ("fund" in cat_norm or "investment" in cat_norm) and not allow_consolidated_keep and not has_open_motion(ev_type, ev_snip, title=o.title, target_company=o.target_company or ""):
            drop.append(_set_drop_reason(o, "funding_without_surface"))
            continue

        # Vertical-dependent items with thin surfaces should be cut rather than retained.
        if any(k in cat_norm for k in ["ticket", "entertain", "gaming", "nft"]) and not allow_consolidated_keep:
            drop.append(_set_drop_reason(o, "vertical_without_surface"))
            continue

        # Governance / forum / discussion surfaces can still belong in KEEP when
        # they expose a concrete early signal or private outreach path.
        if _FINALIZER_GOVERNANCE_WATCH_RE.search(watch_haystack):
            if not allow_consolidated_keep:
                drop.append(_set_drop_reason(o, "governance_without_surface"))
                continue

        # Early-stage items should remain in KEEP when they already have a real
        # action surface; otherwise cut them.
        if is_signal_stage:
            if not allow_consolidated_keep:
                drop.append(_set_drop_reason(o, "signal_stage_without_surface"))
                continue

        governance_like = bool(_FINALIZER_GOVERNANCE_WATCH_RE.search(watch_haystack))
        filecoin_like = bool(_FINALIZER_FILECOIN_PROPOSAL_WATCH_RE.search(watch_haystack))

        # Applicant-side and Filecoin-style grant proposals are retained in KEEP
        # when they represent a real named-counterparty or early-signal motion.
        if _FINALIZER_FILECOIN_PROPOSAL_WATCH_RE.search(watch_haystack):
            if allow_consolidated_keep:
                keep.append(_set_audit_label(o, _retained_label(
                    has_formal_intake=has_formal_intake,
                    has_technical_eval=has_technical_eval,
                    has_commercial_early_keep=has_commercial_early_keep,
                    has_inverted_outreach=has_inverted_outreach,
                    is_signal_stage=is_signal_stage,
                    is_applicant_side=is_applicant_side,
                    governance_like=governance_like,
                    filecoin_like=filecoin_like,
                )))
                continue
            drop.append(_set_drop_reason(o, "filecoin_like_without_surface"))
            continue

        if ev_type == "grant_or_rfp_or_program_open":
            if is_applicant_side:
                if allow_consolidated_keep:
                    keep.append(_set_audit_label(o, _retained_label(
                        has_formal_intake=has_formal_intake,
                        has_technical_eval=has_technical_eval,
                        has_commercial_early_keep=has_commercial_early_keep,
                        has_inverted_outreach=has_inverted_outreach,
                        is_signal_stage=is_signal_stage,
                        is_applicant_side=is_applicant_side,
                        governance_like=governance_like,
                        filecoin_like=filecoin_like,
                    )))
                else:
                    drop.append(_set_drop_reason(o, "grant_applicant_without_surface"))
                continue
            if not has_formal_intake and not allow_consolidated_keep and (weak_surface or not has_clear_buyer):
                drop.append(_set_drop_reason(o, "grant_without_clear_buyer_surface"))
                continue
            keep.append(_set_audit_label(o, _retained_label(
                has_formal_intake=has_formal_intake,
                has_technical_eval=has_technical_eval,
                has_commercial_early_keep=has_commercial_early_keep,
                has_inverted_outreach=has_inverted_outreach,
                is_signal_stage=is_signal_stage,
                is_applicant_side=is_applicant_side,
                governance_like=governance_like,
                filecoin_like=filecoin_like,
            )))
            continue

        if ev_type in {"partnership_or_integration", "support_added_with_bd_action"}:
            if not has_formal_intake and not has_technical_eval and not has_open_motion(
                ev_type, ev_snip, title=o.title, target_company=o.target_company or ""
            ) and not has_commercial_early_keep and not has_inverted_outreach and (weak_surface or not has_clear_buyer):
                drop.append(_set_drop_reason(o, "partnership_without_surface"))
                continue
            keep.append(_set_audit_label(o, _retained_label(
                has_formal_intake=has_formal_intake,
                has_technical_eval=has_technical_eval,
                has_commercial_early_keep=has_commercial_early_keep,
                has_inverted_outreach=has_inverted_outreach,
                is_signal_stage=is_signal_stage,
                is_applicant_side=is_applicant_side,
                governance_like=governance_like,
                filecoin_like=filecoin_like,
            )))
            continue

        if ev_type == "vendor_or_procurement_need":
            if not has_formal_intake and not has_open_motion(
                ev_type, ev_snip, title=o.title, target_company=o.target_company or ""
            ) and not has_commercial_early_keep and not has_inverted_outreach and (weak_surface or not has_clear_buyer):
                drop.append(_set_drop_reason(o, "vendor_without_surface"))
                continue
            keep.append(_set_audit_label(o, _retained_label(
                has_formal_intake=has_formal_intake,
                has_technical_eval=has_technical_eval,
                has_commercial_early_keep=has_commercial_early_keep,
                has_inverted_outreach=has_inverted_outreach,
                is_signal_stage=is_signal_stage,
                is_applicant_side=is_applicant_side,
                governance_like=governance_like,
                filecoin_like=filecoin_like,
            )))
            continue

        if weak_surface and not has_clear_buyer:
            drop.append(_set_drop_reason(o, "weak_surface_no_clear_buyer"))
            continue

        keep.append(_set_audit_label(o, _retained_label(
            has_formal_intake=has_formal_intake,
            has_technical_eval=has_technical_eval,
            has_commercial_early_keep=has_commercial_early_keep,
            has_inverted_outreach=has_inverted_outreach,
            is_signal_stage=is_signal_stage,
            is_applicant_side=is_applicant_side,
            governance_like=governance_like,
            filecoin_like=filecoin_like,
        )))

    return keep, watch, drop


def write_opportunities_csv(path: str, opps: List[Opportunity], status: str) -> None:
    """Write an opportunities CSV for operator review.

    This is a lightweight reporting artifact; it does not affect DB writes.
    """
    fieldnames = [
        "status",
        "opportunity_id",
        "audit_label",
        "confidence",
        "title",
        "summary",
        "reason_summary",
        "reason",
        "opportunity_details",
        "target_company",
        "primary_category",
        "categories",
        "suggested_outreach_angle",
        "evidence_type",
        "evidence_snippet",
        "supporting_snippet",
        "sources",
        "event_url",
        "bd_weeks",
        "event_ids",
        "event_titles",
        "decision_reason",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for o in sorted(opps, key=lambda x: x.confidence, reverse=True):
            primary_cat = (o.categories or [""])[0]
            ev_type = opportunity_evidence_type(o)
            ev_snip = opportunity_evidence_snippet(o)
            support_snip = opportunity_supporting_snippet(o)
            reason_text = extract_reason_text_from_reason_block(o.reason)
            w.writerow({
                "status": status,
                "opportunity_id": o.opportunity_id,
                "audit_label": o.audit_label or ("Cut" if status == "DROP" else "Keep"),
                "confidence": f"{o.confidence:.3f}",
                "title": o.title,
                "summary": o.summary or "",
                "reason_summary": smart_truncate(reason_text, 320),
                "reason": o.reason or "",
                "opportunity_details": o.opportunity_details or "",
                "target_company": o.target_company or "",
                "primary_category": primary_cat,
                "categories": ", ".join(o.categories or []),
                "suggested_outreach_angle": o.suggested_outreach_angle or "",
                "evidence_type": ev_type,
                "evidence_snippet": ev_snip,
                "supporting_snippet": support_snip,
                "sources": " | ".join(o.sources or []),
                "event_url": o.event_url or "",
                "bd_weeks": o.bd_weeks,
                "event_ids": o.event_ids or "",
                "event_titles": o.event_titles or "",
                "decision_reason": o.finalizer_reason or "",
            })

def build_reason_block(reason: str, evidence_type: str, evidence_snippet: str, supporting_snippet: str = "") -> str:
    support_line = f"Supporting snippet: {supporting_snippet}\n" if supporting_snippet else ""
    return (
        f"Evidence type: {evidence_type}\n"
        f"Evidence snippet: {evidence_snippet}\n"
        f"{support_line}"
        f"Reason: {reason}"
    )


# ---------------------------
# Main Processing
# ---------------------------

def process_events_batch(
    db: DatabaseManager,
    client: ThreadLocalOpenAIClientPool,
    finder_model: str,
    gatekeeper_model: str,
    critic_model: str,
    enrichment_model: str,
    refiner_model: str,
    events: List[Event],
    categories_map: dict,
    category_definitions: Dict[str, str],
    category_catalog: List[Dict[str, str]],
    chains_map: Dict[str, Any],
    sectors_map: Dict[str, Any],
    seeking_map: Dict[str, Any],
    max_events_per_batch: int,
    max_output_tokens: int,
    min_confidence: float,
    tracker: Optional[TokenTracker] = None,
) -> int:
    return _process_events_batch_threaded(
        db=db,
        client=client,
        finder_model=finder_model,
        gatekeeper_model=gatekeeper_model,
        critic_model=critic_model,
        enrichment_model=enrichment_model,
        refiner_model=refiner_model,
        events=events,
        categories_map=categories_map,
        category_definitions=category_definitions,
        category_catalog=category_catalog,
        chains_map=chains_map,
        sectors_map=sectors_map,
        seeking_map=seeking_map,
        max_events_per_batch=max_events_per_batch,
        max_output_tokens=max_output_tokens,
        min_confidence=min_confidence,
        tracker=tracker,
    )


def _process_events_batch_threaded(
    db: DatabaseManager,
    client: ThreadLocalOpenAIClientPool,
    finder_model: str,
    gatekeeper_model: str,
    critic_model: str,
    enrichment_model: str,
    refiner_model: str,
    events: List[Event],
    categories_map: dict,
    category_definitions: Dict[str, str],
    category_catalog: List[Dict[str, str]],
    chains_map: Dict[str, Any],
    sectors_map: Dict[str, Any],
    seeking_map: Dict[str, Any],
    max_events_per_batch: int,
    max_output_tokens: int,
    min_confidence: float,
    tracker: Optional[TokenTracker] = None,
) -> int:
    all_opps: List[Opportunity] = []
    all_gate_results: List[Dict[str, Any]] = []
    rejection_audit: List[Dict[str, Any]] = []
    near_miss_watchlist: List[Dict[str, Any]] = []

    total_events = len(events)
    print(f"[OpportunityMatcher] Processing {total_events} events...")

    print("[OpportunityMatcher] [Chunker] Creating fixed-size batches (no event dropping)...")
    intelligent_batches = [
        events[i : i + max_events_per_batch]
        for i in range(0, len(events), max_events_per_batch)
    ]
    total_in_batches = sum(len(b) for b in intelligent_batches)
    print(
        f"[OpportunityMatcher] [Chunker] Created {len(intelligent_batches)} batches | total_in_batches={total_in_batches} (expected={len(events)})\n"
    )

    event_lookup = {ev.event_id: ev for ev in events}

    def _gatekeeper_error_meta(err: Exception, batch_size: int) -> Dict[str, Any]:
        meta = {
            "type": type(err).__name__,
            "batch_size": batch_size,
            "err_len": len(str(err)) if err else 0,
        }
        for key in ("completion_tokens", "max_tokens", "raw_len", "prompt_len", "payload_len"):
            if hasattr(err, key):
                meta[key] = getattr(err, key)
        return meta

    def _run_one_batch(batch_num: int, batch_events: List[Event]) -> Dict[str, Any]:
        local_client = client.get_client()
        thread_name = pretty_thread_name()
        local_opps: List[Opportunity] = []
        local_gate_results: List[Dict[str, Any]] = []
        local_rejection_audit: List[Dict[str, Any]] = []
        local_near_miss_watchlist: List[Dict[str, Any]] = []

        try:
            gate_map = call_gatekeeper(local_client, gatekeeper_model, batch_events, tracker)
        except Exception as e:
            meta = _gatekeeper_error_meta(e, len(batch_events))
            print(f"  WARNING: Gatekeeper failed (details suppressed) | {meta}")
            gate_map = {}
            if len(batch_events) > 1:
                mid = len(batch_events) // 2
                halves = [batch_events[:mid], batch_events[mid:]]
                for idx, half in enumerate(halves, 1):
                    try:
                        gate_map.update(call_gatekeeper(local_client, gatekeeper_model, half, tracker))
                    except Exception as e2:
                        meta2 = _gatekeeper_error_meta(e2, len(half))
                        print(f"  WARNING: Gatekeeper failed (split {idx}/2, details suppressed) | {meta2}")
            if not gate_map:
                return {"batch_num": batch_num, "thread_name": thread_name, "opps": [], "gate": [], "rejects": [], "near_miss": []}

        for event_id, gate_result in gate_map.items():
            event_obj = next((e for e in batch_events if e.event_id == event_id), None)
            local_gate_results.append({
                "batch_num": batch_num,
                "event_id": event_id,
                "event_title": event_obj.title if event_obj else "Unknown",
                "hard_exclusion": gate_result.hard_exclusion,
                "hard_exclusion_reason": gate_result.hard_exclusion_reason,
                "evidence_type": gate_result.evidence_type,
                "evidence_snippet": gate_result.evidence_snippet,
                "supporting_snippet": gate_result.supporting_snippet,
                "passes_gate": gate_result.passes(),
            })

        gated: List[Tuple[Event, GateResult]] = []
        excluded = 0
        rescued = 0
        for ev in batch_events:
            gr = gate_map.get(ev.event_id)
            if not gr:
                excluded += 1
                local_rejection_audit.append({
                    "stage": "gatekeeper",
                    "event_id": ev.event_id,
                    "title": ev.title or "",
                    "event_title": ev.title or "",
                    "source_event_ids": [ev.event_id],
                    "decision": "excluded",
                    "reason": "missing_gate_result",
                    "reason_summary": "Gatekeeper returned no result for this event.",
                    "final_status": "rejected",
                    "review_status": "gatekeeper",
                })
                continue
            rescued_gr = maybe_rescue_gatekeeper_media_rollout(ev, gr)
            if rescued_gr:
                gate_map[ev.event_id] = rescued_gr
                gr = rescued_gr
                rescued += 1
            if gr.passes():
                veto_reason = gatekeeper_postpass_veto_reason(ev, gr)
                if veto_reason:
                    gr.hard_exclusion = True
                    gr.hard_exclusion_reason = veto_reason
                    excluded += 1
                    local_rejection_audit.append({
                        "stage": "gatekeeper",
                        "event_id": ev.event_id,
                        "title": ev.title or "",
                        "event_title": ev.title or "",
                        "source_event_ids": [ev.event_id],
                        "decision": "excluded",
                        "reason": veto_reason,
                        "reason_summary": veto_reason.replace("_", " "),
                        "hard_exclusion": True,
                        "evidence_type": gr.evidence_type,
                        "evidence_snippet": gr.evidence_snippet,
                        "supporting_snippet": gr.supporting_snippet,
                        "final_status": "rejected",
                        "review_status": "gatekeeper",
                    })
                    continue
                gated.append((ev, gr))
            else:
                reject_reason = (
                    normalize_hard_exclusion_reason(gr.hard_exclusion_reason, gr.evidence_type)
                    if gr.hard_exclusion
                    else ("no_qualifying_evidence" if gr.evidence_type == "none" else "no_verbatim_evidence")
                )
                excluded += 1
                local_rejection_audit.append({
                    "stage": "gatekeeper",
                    "event_id": ev.event_id,
                    "title": ev.title or "",
                    "event_title": ev.title or "",
                    "source_event_ids": [ev.event_id],
                    "decision": "excluded",
                    "reason": reject_reason,
                    "reason_summary": reject_reason.replace("_", " "),
                    "hard_exclusion": gr.hard_exclusion,
                    "evidence_type": gr.evidence_type,
                    "evidence_snippet": gr.evidence_snippet,
                    "supporting_snippet": gr.supporting_snippet,
                    "final_status": "rejected",
                    "review_status": "gatekeeper",
                })

        for row in local_gate_results:
            updated = gate_map.get(row["event_id"])
            if not updated:
                continue
            row["hard_exclusion"] = updated.hard_exclusion
            row["hard_exclusion_reason"] = updated.hard_exclusion_reason
            row["evidence_type"] = updated.evidence_type
            row["evidence_snippet"] = updated.evidence_snippet
            row["supporting_snippet"] = updated.supporting_snippet
            row["passes_gate"] = updated.passes()

        rescue_suffix = f" | Rescued: {rescued}" if rescued else ""
        print(f"  [Gatekeeper- Batch{batch_num}] Passed: {len(gated)} | Excluded: {excluded}{rescue_suffix}")
        if not gated:
            return {"batch_num": batch_num, "thread_name": thread_name, "opps": [], "gate": local_gate_results, "rejects": local_rejection_audit, "near_miss": local_near_miss_watchlist}

        try:
            raw_opps = call_finder(local_client, finder_model, gated, categories_map, category_catalog, max_output_tokens, tracker)
        except OpenAIRequestTooLarge as e:
            print(f"  WARNING: Request too large: {e}")
            return {"batch_num": batch_num, "thread_name": thread_name, "opps": [], "gate": local_gate_results, "rejects": local_rejection_audit, "near_miss": local_near_miss_watchlist}
        except Exception as e:
            print(f"  WARNING: Finder failed: {e}")
            return {"batch_num": batch_num, "thread_name": thread_name, "opps": [], "gate": local_gate_results, "rejects": local_rejection_audit, "near_miss": local_near_miss_watchlist}

        filtered: List[Dict[str, Any]] = []
        filtered_reasons = {"not_opportunity": 0, "low_confidence": 0, "obvious_cut": 0}
        batch_title_map = {ev.event_id: (ev.title or "").strip() for ev in batch_events}
        batch_event_map = {ev.event_id: ev for ev in batch_events}

        def _near_miss_context(o: Dict[str, Any]) -> Dict[str, Any]:
            event_ids = [str(eid) for eid in (o.get("source_event_ids", []) or []) if str(eid)]
            source_events = [batch_event_map[eid] for eid in event_ids if eid in batch_event_map]
            summary = clean_text(o.get("summary")) or ""
            body_preview = clean_text(o.get("opportunity_details")) or clean_text(o.get("body_preview")) or ""
            bd_signal = {}
            supporting_snippet = clean_text(o.get("supporting_snippet")) or clean_text(o.get("evidence_snippet")) or ""
            derived_target_company = clean_text(o.get("target_company")) or ""
            reason_text = clean_text(o.get("reason")) or ""
            outreach = (
                clean_text(o.get("suggested_outreach_angle"))
                or clean_text(o.get("recommended_action"))
                or ""
            )
            if source_events:
                ev = source_events[0]
                summary = summary or clean_text(ev.summary) or clean_text(ev.title) or ""
                desc = clean_text(ev.description) or ""
                body_preview = body_preview or desc[:3000] or summary
                bd_signal = ev.bd_signal or {}

                why_parts: List[str] = []
                if isinstance(ev.why_it_matters, list):
                    for item in ev.why_it_matters:
                        item_text = clean_text(item)
                        if item_text:
                            why_parts.append(item_text)
                why_text = " ".join(why_parts).strip()
                reason_text = reason_text or why_text or summary
                outreach = outreach or clean_text(ev.recommended_action) or ""

                if not derived_target_company:
                    derived_target_company = derive_target_company(
                        [e.url for e in source_events if e.url],
                        [e.source for e in source_events if e.source],
                    ) or ""

                if not outreach and ev.url:
                    outreach = (
                        "Use the linked source thread to initiate a concrete outreach "
                        "about integration, implementation, or service support."
                    )

            return {
                "summary": summary,
                "body_preview": body_preview,
                "target_company": derived_target_company,
                "categories": o.get("categories", []) or [],
                "reason_text": reason_text,
                "suggested_outreach_angle": outreach,
                "supporting_snippet": supporting_snippet,
                "bd_signal": bd_signal,
            }

        def _debug_title(o: Dict[str, Any]) -> str:
            title = (o.get("title") or "").strip()
            if title:
                return title
            for eid in o.get("source_event_ids", []) or []:
                mapped = batch_title_map.get(str(eid), "").strip()
                if mapped:
                    return mapped
            target = (o.get("target_company") or "").strip()
            return target or "Untitled"

        for o in raw_opps:
            if not o.get("is_opportunity", False):
                filtered_reasons["not_opportunity"] += 1
                dbg_title = _debug_title(o)
                print(f"    DEBUG: Filtered '{dbg_title[:50]}' - is_opportunity=False")
                local_rejection_audit.append({
                    "stage": "finder",
                    "decision": "filtered",
                    "reason": "is_opportunity_false",
                    "title": dbg_title,
                    "target_company": o.get("target_company"),
                    "source_event_ids": o.get("source_event_ids", []),
                })
                local_near_miss_watchlist.append({
                    "stage": "finder",
                    "decision": "filtered",
                    "reason": "is_opportunity_false",
                    "batch_num": batch_num,
                    "title": dbg_title,
                    "target_company": o.get("target_company"),
                    "confidence": None,
                    "min_confidence": float(min_confidence),
                    "evidence_type": o.get("evidence_type"),
                    "evidence_snippet": enforce_verbatim_snippet(o.get("evidence_snippet")),
                    "source_event_ids": o.get("source_event_ids", []),
                    "feedback": "",
                    "sub_scores": o.get("sub_scores", {}) or {},
                    **_near_miss_context(o),
                })
                continue
            sub = o.get("sub_scores", {}) or {}
            conf = confidence_from_subscores(sub)
            if conf < min_confidence:
                filtered_reasons["low_confidence"] += 1
                dbg_title = _debug_title(o)
                print(f"    DEBUG: Filtered '{dbg_title[:50]}' - confidence={conf:.3f} (sub_scores: {sub})")
                local_rejection_audit.append({
                    "stage": "finder",
                    "decision": "filtered",
                    "reason": "below_min_confidence",
                    "min_confidence": float(min_confidence),
                    "confidence": float(conf),
                    "sub_scores": sub,
                    "title": dbg_title,
                    "target_company": o.get("target_company"),
                    "source_event_ids": o.get("source_event_ids", []),
                })
                local_near_miss_watchlist.append({
                    "stage": "finder",
                    "decision": "filtered",
                    "reason": "below_min_confidence",
                    "batch_num": batch_num,
                    "title": dbg_title,
                    "target_company": o.get("target_company"),
                    "confidence": float(conf),
                    "min_confidence": float(min_confidence),
                    "evidence_type": o.get("evidence_type"),
                    "evidence_snippet": enforce_verbatim_snippet(o.get("evidence_snippet")),
                    "source_event_ids": o.get("source_event_ids", []),
                    "feedback": "",
                    "sub_scores": sub,
                    **_near_miss_context(o),
                })
                continue
            obvious_cut_reason = finder_obvious_guideline_cut_reason(o)
            if obvious_cut_reason:
                filtered_reasons["obvious_cut"] += 1
                dbg_title = _debug_title(o)
                print(f"    DEBUG: Filtered '{dbg_title[:50]}' - obvious_cut={obvious_cut_reason}")
                local_rejection_audit.append({
                    "stage": "finder",
                    "decision": "filtered",
                    "reason": obvious_cut_reason,
                    "title": dbg_title,
                    "target_company": o.get("target_company"),
                    "source_event_ids": o.get("source_event_ids", []),
                    "sub_scores": sub,
                })
                continue
            o["confidence"] = conf
            filtered.append(o)

        print(f"  [Finder- Batch{batch_num}] Produced: {len(raw_opps)} | Kept after gate+confidence: {len(filtered)}")
        if len(raw_opps) > len(filtered):
            print(
                f"    Filtered: {filtered_reasons['not_opportunity']} (is_opportunity=False), "
                f"{filtered_reasons['low_confidence']} (confidence<{min_confidence}), "
                f"{filtered_reasons['obvious_cut']} (obvious_guideline_cut)"
            )

        if not filtered:
            return {"batch_num": batch_num, "thread_name": thread_name, "opps": [], "gate": local_gate_results, "rejects": local_rejection_audit, "near_miss": local_near_miss_watchlist}

        print(f"  [Critic- Batch{batch_num}][{thread_name}]", end=" ")
        feedback = call_critic(local_client, critic_model, filtered, {ev.event_id: ev for ev in batch_events}, tracker)
        rating = normalize_critic_rating(feedback.get("overall_rating", 0))
        print(f"rated {float(rating):.1f}/10")

        final_items: List[Dict[str, Any]] = []
        fb_items = feedback.get("opportunity_feedback", []) or []
        critic_requested_reframe = False
        for fb in fb_items:
            if isinstance(fb, str):
                try:
                    fb = json.loads(fb)
                except Exception:
                    continue
            if not isinstance(fb, dict):
                continue
            idx = int(fb.get("index", -1))
            if idx < 0 or idx >= len(filtered):
                continue
            route_to_near_miss = critic_feedback_should_go_to_near_miss(fb)
            soft_lift = False
            if route_to_near_miss and should_soft_lift_critic_near_miss(filtered[idx], fb, min_confidence):
                route_to_near_miss = False
                soft_lift = True
            status = fb.get("status")
            if not status:
                audit_bucket = fb.get("audit_bucket")
                if audit_bucket == "keep":
                    status = "keep"
                elif audit_bucket in {"keep_but_reframe", "manual_review"}:
                    status = "reframe"
                elif audit_bucket == "cut":
                    status = "discard"
            if not status:
                rec = fb.get("recommendation")
                if rec == "keep":
                    status = "keep"
                elif rec in {"drop", "watchlist"}:
                    status = "discard"
            if status == "discard":
                if not critic_discard_is_hard_exclusion(filtered[idx], fb, min_confidence):
                    ro = fb.get("reframed_opportunity")
                    if isinstance(ro, list):
                        ro = ro[0] if ro else None
                    rescued = ro if isinstance(ro, dict) and ro.get("is_opportunity", False) else dict(filtered[idx])
                    if not rescued.get("evidence_type"):
                        rescued["evidence_type"] = filtered[idx].get("evidence_type")
                    if not rescued.get("evidence_snippet"):
                        rescued["evidence_snippet"] = filtered[idx].get("evidence_snippet")
                    if not rescued.get("supporting_snippet"):
                        rescued["supporting_snippet"] = filtered[idx].get("supporting_snippet")
                    corrected = fb.get("corrected_sub_scores") or rescued.get("sub_scores") or filtered[idx].get("sub_scores") or {}
                    rescued["sub_scores"] = corrected
                    rescued["confidence"] = confidence_from_subscores(corrected)
                    if rescued["confidence"] >= min_confidence:
                        final_items.append(rescued)
                        continue
                local_rejection_audit.append({
                    "stage": "critic",
                    "decision": "discard",
                    "title": filtered[idx].get("title", ""),
                    "target_company": filtered[idx].get("target_company"),
                    "source_event_ids": filtered[idx].get("source_event_ids", []),
                    "feedback": fb.get("feedback") or fb.get("critic_notes", ""),
                })
                local_near_miss_watchlist.append({
                    "stage": "critic",
                    "decision": "discard",
                    "reason": "critic_discard",
                    "batch_num": batch_num,
                    "title": filtered[idx].get("title", ""),
                    "target_company": filtered[idx].get("target_company"),
                    "confidence": float(filtered[idx].get("confidence", 0.0) or 0.0),
                    "min_confidence": float(min_confidence),
                    "evidence_type": filtered[idx].get("evidence_type"),
                    "evidence_snippet": enforce_verbatim_snippet(filtered[idx].get("evidence_snippet")),
                    "source_event_ids": filtered[idx].get("source_event_ids", []),
                    "feedback": fb.get("feedback") or fb.get("critic_notes", ""),
                    "sub_scores": filtered[idx].get("sub_scores", {}) or {},
                    **_near_miss_context(filtered[idx]),
                })
                continue
            if status == "keep":
                if route_to_near_miss:
                    local_near_miss_watchlist.append({
                        "stage": "critic",
                        "decision": "manual_review",
                        "reason": (fb.get("reason_code") or fb.get("audit_bucket") or "critic_manual_review"),
                        "batch_num": batch_num,
                        "title": filtered[idx].get("title", ""),
                        "target_company": filtered[idx].get("target_company"),
                        "confidence": float(filtered[idx].get("confidence", 0.0) or 0.0),
                        "min_confidence": float(min_confidence),
                        "evidence_type": filtered[idx].get("evidence_type"),
                        "evidence_snippet": enforce_verbatim_snippet(filtered[idx].get("evidence_snippet")),
                        "source_event_ids": filtered[idx].get("source_event_ids", []),
                        "feedback": fb.get("feedback") or fb.get("critic_notes", ""),
                        "sub_scores": filtered[idx].get("sub_scores", {}) or {},
                        **_near_miss_context(filtered[idx]),
                    })
                    continue
                if soft_lift:
                    filtered[idx]["reason"] = append_signal_stage_note(filtered[idx].get("reason"))
                corrected = fb.get("corrected_sub_scores")
                if corrected:
                    filtered[idx]["sub_scores"] = corrected
                    filtered[idx]["confidence"] = confidence_from_subscores(corrected)
                if not filtered[idx].get("supporting_snippet"):
                    filtered[idx]["supporting_snippet"] = filtered[idx].get("evidence_snippet")
                final_items.append(filtered[idx])
            if status == "reframe":
                critic_requested_reframe = True
                if route_to_near_miss:
                    local_near_miss_watchlist.append({
                        "stage": "critic",
                        "decision": "manual_review",
                        "reason": (fb.get("reason_code") or fb.get("audit_bucket") or "critic_manual_review"),
                        "batch_num": batch_num,
                        "title": filtered[idx].get("title", ""),
                        "target_company": filtered[idx].get("target_company"),
                        "confidence": float(filtered[idx].get("confidence", 0.0) or 0.0),
                        "min_confidence": float(min_confidence),
                        "evidence_type": filtered[idx].get("evidence_type"),
                        "evidence_snippet": enforce_verbatim_snippet(filtered[idx].get("evidence_snippet")),
                        "source_event_ids": filtered[idx].get("source_event_ids", []),
                        "feedback": fb.get("feedback") or fb.get("critic_notes", ""),
                        "sub_scores": filtered[idx].get("sub_scores", {}) or {},
                        **_near_miss_context(filtered[idx]),
                    })
                    continue
                ro = fb.get("reframed_opportunity")
                if isinstance(ro, list):
                    ro = ro[0] if ro else None
                if ro and ro.get("is_opportunity", False):
                    if soft_lift:
                        ro["reason"] = append_signal_stage_note(ro.get("reason"))
                    if not ro.get("evidence_type"):
                        ro["evidence_type"] = filtered[idx].get("evidence_type")
                    if not ro.get("evidence_snippet"):
                        ro["evidence_snippet"] = filtered[idx].get("evidence_snippet")
                    if not ro.get("supporting_snippet"):
                        ro["supporting_snippet"] = filtered[idx].get("supporting_snippet") or filtered[idx].get("evidence_snippet")
                    corrected = fb.get("corrected_sub_scores") or ro.get("sub_scores")
                    ro["sub_scores"] = corrected
                    ro["confidence"] = confidence_from_subscores(corrected)
                    if ro["confidence"] >= min_confidence:
                        final_items.append(ro)

        if critic_requested_reframe or rating < 7.0 or len(final_items) == 0:
            print(f"  [Refiner][Batch {batch_num}][{thread_name}]", end=" ")
            try:
                critic_approved_items = list(final_items)
                refined = call_refiner(
                    local_client,
                    refiner_model,
                    gated,
                    final_items or filtered,
                    feedback,
                    categories_map,
                    category_catalog,
                    max_output_tokens,
                    tracker,
                )
                refined_keep: List[Dict[str, Any]] = []
                for o in refined:
                    if not o.get("is_opportunity", False):
                        continue
                    conf = confidence_from_subscores(o.get("sub_scores", {}) or {})
                    if conf < min_confidence:
                        continue
                    o["confidence"] = conf
                    refined_keep.append(o)
                if refined_keep:
                    final_items = refined_keep
                    print(f"-> {len(final_items)}")
                else:
                    final_items = critic_approved_items
                    print(f"-> 0 (preserved {len(final_items)} critic-approved)")
            except Exception as e:
                print(f"FAILED ({e})")

        if not final_items:
            return {"batch_num": batch_num, "thread_name": thread_name, "opps": [], "gate": local_gate_results, "rejects": local_rejection_audit, "near_miss": local_near_miss_watchlist}

        reach_filtered: List[Dict[str, Any]] = []
        for o in final_items:
            target = (o.get("target_company") or "").strip()
            title = (o.get("title") or "").strip()
            et = (o.get("evidence_type") or "").strip()
            es = enforce_verbatim_snippet(o.get("evidence_snippet")) or ""
            o["evidence_snippet"] = es

            mega = is_mega_counterparty(target) or is_mega_counterparty(title)
            if mega and not has_open_motion(et, es, title=title, target_company=target):
                continue

            reach_filtered.append(o)

        final_items = reach_filtered
        if not final_items:
            return {"batch_num": batch_num, "thread_name": thread_name, "opps": [], "gate": local_gate_results, "rejects": local_rejection_audit, "near_miss": local_near_miss_watchlist}

        for ro in final_items:
            built, build_error = build_opportunity_from_raw_candidate(
                ro,
                event_lookup,
                categories_map,
                category_definitions,
            )
            if not built:
                if build_error == "no_valid_categories_after_guardrails":
                    local_rejection_audit.append({
                        "stage": "category_guardrails",
                        "decision": "discard",
                        "reason": "no_valid_categories_after_guardrails",
                        "title": ro.get("title", ""),
                        "target_company": ro.get("target_company"),
                        "evidence_type": ro.get("evidence_type"),
                        "evidence_snippet": enforce_verbatim_snippet(ro.get("evidence_snippet")),
                        "raw_categories": ro.get("categories", []),
                    })
                continue
            local_opps.append(built)

        return {"batch_num": batch_num, "thread_name": thread_name, "opps": local_opps, "gate": local_gate_results, "rejects": local_rejection_audit, "near_miss": local_near_miss_watchlist}

    results_by_batch: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=MAX_BATCH_WORKERS) as ex:
        futures = [ex.submit(_run_one_batch, i, b) for i, b in enumerate(intelligent_batches, 1)]
        for fut in as_completed(futures):
            res = fut.result()
            if isinstance(res, dict) and "batch_num" in res:
                results_by_batch[int(res["batch_num"])] = res
                print(
                    f"[OpportunityMatcher] Finished batch {res['batch_num']}/{len(intelligent_batches)} "
                    f"on {res.get('thread_name', 'unknown-thread')}"
                )

    for batch_num in sorted(results_by_batch.keys()):
        res = results_by_batch[batch_num]
        all_opps.extend(res.get("opps") or [])
        all_gate_results.extend(res.get("gate") or [])
        rejection_audit.extend(res.get("rejects") or [])
        near_miss_watchlist.extend(res.get("near_miss") or [])

    recovered_opps: List[Opportunity] = []

    print(f"\n[OpportunityMatcher] Total opportunities (pre-dedupe): {len(all_opps)}")
    if all_opps:
        dist = Counter()
        for o in all_opps:
            for c in (o.categories or []):
                dist[c] += 1
        print("\n[OpportunityMatcher] Category Distribution (Drift Monitor, pre-dedupe):")
        for cat, count in sorted(dist.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {cat}: {count}")

    all_opps = dedupe_opportunities(all_opps)
    print(f"[OpportunityMatcher] Total opportunities (post-dedupe): {len(all_opps)}")

    keep_opps, watch_opps, drop_opps = finalize_opportunities(all_opps)
    print(
        f"[OpportunityMatcher] Finalizer: KEEP={len(keep_opps)} | WATCHLIST={len(watch_opps)} | DROP={len(drop_opps)}"
    )

    event_lookup = {ev.event_id: ev for ev in events}
    if watch_opps:
        keep_opps, watch_opps = promote_watchlist_candidates(
            client=client.get_client(),
            model=DEFAULT_WATCHLIST_PROMOTION_MODEL,
            watch_opps=watch_opps,
            keep_opps=keep_opps,
            event_lookup=event_lookup,
            category_catalog=category_catalog,
            categories_map=categories_map,
            category_definitions=category_definitions,
            max_output_tokens=max_output_tokens,
            tracker=tracker,
            rejection_audit=rejection_audit,
        )
        print(
            f"[OpportunityMatcher] Post-WATCHLIST promotion: KEEP={len(keep_opps)} | "
            f"WATCHLIST={len(watch_opps)} | DROP={len(drop_opps)}"
        )

    drop_keep_ratio, drop_recovery_min_restored = compute_drop_recovery_min_restored(
        len(keep_opps),
        len(drop_opps),
    )

    drop_recovery_candidates = select_dropped_recovery_candidates(
        drop_opps,
        event_lookup,
        min_confidence=min_confidence,
        cap=RECOVERY_MAX_CANDIDATES,
    )
    drop_recovery_reasons = recovery_skip_reasons(
        len(drop_recovery_candidates),
        len(keep_opps),
        min_candidates_to_run=DROP_RECOVERY_MIN_CANDIDATES_TO_RUN,
    )
    if drop_recovery_candidates and drop_recovery_reasons:
        print(
            "[OpportunityMatcher] Skipping Dropped-Items Recovery Agent (low ROI): "
            + "; ".join(drop_recovery_reasons)
        )
    if drop_recovery_candidates and not drop_recovery_reasons:
        ratio_text = "inf" if math.isinf(drop_keep_ratio) else f"{drop_keep_ratio:.2f}"
        accepted_examples = build_recovery_reference_examples(keep_opps, max_items=4)
        recovery_payload: List[Dict[str, Any]] = []
        recovery_payload_rows: List[Dict[str, Any]] = []
        skipped_recovery_rows: List[Dict[str, Any]] = []
        for row in drop_recovery_candidates:
            source_events = select_recovery_source_events(row, event_lookup)
            if not source_events:
                skipped_recovery_rows.append(row)
                continue
            idx = len(recovery_payload_rows)
            recovery_payload_rows.append(row)
            recovery_payload.append(
                {
                    "index": idx,
                    "dropped_item": {
                        "stage": row.get("stage"),
                        "reason": row.get("reason"),
                        "title": row.get("title"),
                        "summary": row.get("summary"),
                        "body_preview": row.get("body_preview"),
                        "target_company": row.get("target_company"),
                        "suggested_outreach_angle": row.get("suggested_outreach_angle"),
                        "confidence": near_miss_effective_confidence(row),
                        "min_confidence": row.get("min_confidence"),
                        "evidence_type": row.get("evidence_type"),
                        "evidence_snippet": row.get("evidence_snippet"),
                        "supporting_snippet": row.get("supporting_snippet"),
                        "reason_text": row.get("reason_text"),
                        "bd_signal": row.get("bd_signal"),
                        "feedback": row.get("feedback"),
                        "sub_scores": row.get("sub_scores"),
                        "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                        "audit_label": row.get("audit_label"),
                    },
                    "source_events": [
                        {
                            **build_enrichment_event_context(ev, clean_text(row.get("evidence_snippet")) or ""),
                            "body_context": _polish_recovery_body_context(
                                build_enrichment_event_context(ev, clean_text(row.get("evidence_snippet")) or "").get("body_context", "")
                            ),
                        }
                        for ev in source_events
                    ],
                }
            )

        for row in skipped_recovery_rows:
            rejection_audit.append({
                "stage": "dropped_items_recovery_agent",
                "decision": "discard",
                "reason": "recover_false_reason:no_source_events_for_recovery",
                "title": row.get("title", ""),
                "target_company": row.get("target_company"),
                "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                "feedback": "",
            })

        if recovery_payload:
            print(
                f"[OpportunityMatcher] Running Dropped-Items Recovery Agent on "
                f"{len(recovery_payload)} dropped candidates..."
            )
            print(
                f"[OpportunityMatcher] Dropped-Items Recovery target: "
                f"ratio={ratio_text} | minimum_restore={drop_recovery_min_restored} | "
                f"cap={RECOVERY_MAX_RESTORED}"
            )
            recovered_row_ids: Set[int] = set()
            recovery_outcome_by_row: Dict[int, Dict[str, str]] = {}
            empty_chunk_count = 0

            def _set_drop_recovery_outcome(row: Dict[str, Any], reason: str, feedback: str = "") -> None:
                row_id = int(row.get("_drop_opp_id") or 0)
                recovery_outcome_by_row[row_id] = {
                    "reason": reason,
                    "feedback": feedback or "",
                }

            def _apply_drop_recovery_decisions(decisions: List[Dict[str, Any]], active_rows: List[Dict[str, Any]]) -> None:
                nonlocal recovered_opps, recovered_row_ids
                decisions_by_index: Dict[int, Dict[str, Any]] = {}
                for decision in decisions:
                    if not isinstance(decision, dict):
                        continue
                    try:
                        idx = int(decision.get("index", -1))
                    except Exception:
                        continue
                    if 0 <= idx < len(active_rows):
                        decisions_by_index[idx] = decision

                for idx, row in enumerate(active_rows):
                    decision = decisions_by_index.get(idx)
                    if not isinstance(decision, dict):
                        _set_drop_recovery_outcome(row, "recover_false_reason:no_decision_returned")
                        continue

                    rationale = clean_text(decision.get("rationale")) or ""
                    if not decision.get("recover"):
                        _set_drop_recovery_outcome(row, "recover_false_reason", rationale)
                        continue

                    ro = decision.get("opportunity")
                    if isinstance(ro, list):
                        ro = ro[0] if ro else None
                    if not isinstance(ro, dict) or not ro.get("is_opportunity", False):
                        _set_drop_recovery_outcome(
                            row,
                            "recover_false_reason:invalid_or_missing_opportunity",
                            rationale,
                        )
                        continue
                    ro["source_event_ids"] = normalize_source_event_ids(row.get("source_event_ids"))
                    ro["evidence_type"] = row.get("evidence_type")
                    ro["evidence_snippet"] = enforce_verbatim_snippet(row.get("evidence_snippet")) or ""
                    ro["supporting_snippet"] = enforce_verbatim_snippet(row.get("supporting_snippet"), max_words=30) or ""
                    # Recovery model output can miss required fields even when DROP row already has them.
                    # Fill missing essentials from the source row before strict validation.
                    if not clean_text(ro.get("title")):
                        ro["title"] = clean_text(row.get("title")) or ro.get("title")
                    if not clean_text(ro.get("summary")):
                        ro["summary"] = (
                            clean_text(row.get("summary"))
                            or clean_text(row.get("body_preview"))
                            or clean_text(row.get("reason_text"))
                            or ""
                        )
                    if not clean_text(ro.get("reason")):
                        ro["reason"] = (
                            clean_text(row.get("reason_text"))
                            or clean_text(row.get("reason"))
                            or ""
                        )
                    if not clean_text(ro.get("suggested_outreach_angle")):
                        ro["suggested_outreach_angle"] = (
                            clean_text(row.get("suggested_outreach_angle"))
                            or ""
                        )
                    if not normalize_target_company(ro.get("target_company")):
                        fallback_target = normalize_target_company(row.get("target_company"))
                        if fallback_target:
                            ro["target_company"] = fallback_target

                    field_issue = recovery_important_field_issue(ro, row)
                    if field_issue:
                        _set_drop_recovery_outcome(row, f"recovery_field_issue:{field_issue}", rationale)
                        continue
                    conf = confidence_from_subscores(ro.get("sub_scores", {}) or {})
                    if conf < min_confidence:
                        _set_drop_recovery_outcome(
                            row,
                            "confidence_gate_fail",
                            f"{rationale} | conf={conf:.3f} < min_confidence={min_confidence:.3f}",
                        )
                        continue
                    ro["confidence"] = conf
                    built, build_error = build_opportunity_from_raw_candidate(
                        ro,
                        event_lookup,
                        categories_map,
                        category_definitions,
                    )
                    if not built:
                        build_msg = clean_text(str(build_error or "unknown_build_failure")) or "unknown_build_failure"
                        _set_drop_recovery_outcome(
                            row,
                            "build_fail",
                            f"{rationale} | {build_msg}",
                        )
                        continue
                    if any(existing.event_ids == built.event_ids and existing.title == built.title for existing in keep_opps):
                        _set_drop_recovery_outcome(row, "recover_false_reason:duplicate_existing_keep", rationale)
                        continue
                    if any(existing.event_ids == built.event_ids and existing.title == built.title for existing in recovered_opps):
                        _set_drop_recovery_outcome(row, "recover_false_reason:duplicate_recovered", rationale)
                        continue
                    built.audit_label = "Early Signal"
                    recovered_opps.append(built)
                    recovered_row_ids.add(int(row.get("_drop_opp_id") or 0))
                    rejection_audit.append({
                        "stage": "dropped_items_recovery_agent",
                        "decision": "recovered",
                        "title": built.title,
                        "target_company": built.target_company,
                        "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                        "feedback": decision.get("rationale", ""),
                    })
                    if len(recovered_opps) >= RECOVERY_MAX_RESTORED:
                        break

            try:
                for chunk_start in range(0, len(recovery_payload), RECOVERY_CHUNK_SIZE):
                    if len(recovered_opps) >= RECOVERY_MAX_RESTORED:
                        break
                    if empty_chunk_count >= RECOVERY_MAX_EMPTY_CHUNKS:
                        print(
                            f"[OpportunityMatcher] Stopping Dropped-Items Recovery after {empty_chunk_count} empty chunks."
                        )
                        break
                    chunk_end = min(chunk_start + RECOVERY_CHUNK_SIZE, len(recovery_payload))
                    chunk_payload = recovery_payload[chunk_start:chunk_end]
                    chunk_rows = recovery_payload_rows[chunk_start:chunk_end]
                    if not chunk_payload:
                        continue

                    print(
                        f"[OpportunityMatcher] Dropped-Items Recovery pass on candidates "
                        f"{chunk_start + 1}-{chunk_end} of {len(recovery_payload)}..."
                    )
                    before_count = len(recovered_opps)
                    recovery_decisions = call_recovery_agent(
                        client=client.get_client(),
                        model=DEFAULT_RECOVERY_MODEL,
                        candidates=chunk_payload,
                        category_catalog=category_catalog,
                        accepted_examples=accepted_examples,
                        max_output_tokens=max_output_tokens,
                        tracker=tracker,
                        prefer_early_signals=True,
                        stronger_rescue_bias=drop_recovery_min_restored > 0,
                        target_remaining=max(0, drop_recovery_min_restored - len(recovered_opps)),
                    )
                    _apply_drop_recovery_decisions(recovery_decisions, chunk_rows)

                    if len(recovered_opps) >= RECOVERY_MAX_RESTORED:
                        break

                    if len(recovered_opps) == before_count:
                        fallback_payload = chunk_payload[: min(RECOVERY_FALLBACK_CANDIDATES, len(chunk_payload))]
                        fallback_rows = chunk_rows[: len(fallback_payload)]
                        fallback_decisions = call_recovery_agent(
                            client=client.get_client(),
                            model=DEFAULT_RECOVERY_MODEL,
                            candidates=fallback_payload,
                            category_catalog=category_catalog,
                            accepted_examples=accepted_examples,
                            max_output_tokens=max_output_tokens,
                            tracker=tracker,
                            force_one=True,
                            prefer_early_signals=True,
                            stronger_rescue_bias=drop_recovery_min_restored > len(recovered_opps),
                            target_remaining=max(0, drop_recovery_min_restored - len(recovered_opps)),
                        )
                        _apply_drop_recovery_decisions(fallback_decisions, fallback_rows)
                    if len(recovered_opps) == before_count:
                        empty_chunk_count += 1
                    else:
                        empty_chunk_count = 0

                if len(recovered_opps) < drop_recovery_min_restored:
                    remaining_rows = [
                        row for row in recovery_payload_rows
                        if int(row.get("_drop_opp_id") or 0) not in recovered_row_ids
                    ]
                    for row in remaining_rows:
                        if len(recovered_opps) >= drop_recovery_min_restored:
                            break
                        rescue_decisions = call_recovery_agent(
                            client=client.get_client(),
                            model=DEFAULT_RECOVERY_MODEL,
                            candidates=[
                                {
                                    "index": 0,
                                    "dropped_item": {
                                        "stage": row.get("stage"),
                                        "reason": row.get("reason"),
                                        "title": row.get("title"),
                                        "summary": row.get("summary"),
                                        "body_preview": row.get("body_preview"),
                                        "target_company": row.get("target_company"),
                                        "suggested_outreach_angle": row.get("suggested_outreach_angle"),
                                        "confidence": near_miss_effective_confidence(row),
                                        "min_confidence": row.get("min_confidence"),
                                        "evidence_type": row.get("evidence_type"),
                                        "evidence_snippet": row.get("evidence_snippet"),
                                        "supporting_snippet": row.get("supporting_snippet"),
                                        "reason_text": row.get("reason_text"),
                                        "bd_signal": row.get("bd_signal"),
                                        "feedback": row.get("feedback"),
                                        "sub_scores": row.get("sub_scores"),
                                        "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                                        "audit_label": row.get("audit_label"),
                                    },
                                    "source_events": [
                                        {
                                            **build_enrichment_event_context(ev, clean_text(row.get("evidence_snippet")) or ""),
                                            "body_context": _polish_recovery_body_context(
                                                build_enrichment_event_context(
                                                    ev,
                                                    clean_text(row.get("evidence_snippet")) or "",
                                                ).get("body_context", "")
                                            ),
                                        }
                                        for ev in select_recovery_source_events(row, event_lookup)
                                    ],
                                }
                            ],
                            category_catalog=category_catalog,
                            accepted_examples=accepted_examples,
                            max_output_tokens=max_output_tokens,
                            tracker=tracker,
                            force_one=True,
                            prefer_early_signals=True,
                            stronger_rescue_bias=True,
                            target_remaining=max(0, drop_recovery_min_restored - len(recovered_opps)),
                        )
                        _apply_drop_recovery_decisions(rescue_decisions, [row])
            except Exception as e:
                print(f"[OpportunityMatcher] WARNING: Dropped-Items Recovery Agent failed: {e}")
                recovered_opps = []
                recovered_row_ids = set()

            for row in recovery_payload_rows:
                row_id = int(row.get("_drop_opp_id") or 0)
                if row_id in recovered_row_ids:
                    continue
                outcome = recovery_outcome_by_row.get(row_id) or {
                    "reason": "recover_false_reason:no_final_decision",
                    "feedback": "",
                }
                rejection_audit.append({
                    "stage": "dropped_items_recovery_agent",
                    "decision": "discard",
                    "reason": outcome.get("reason") or "recover_false_reason",
                    "title": row.get("title", ""),
                    "target_company": row.get("target_company"),
                    "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                    "feedback": outcome.get("feedback", ""),
                })

            if recovered_opps:
                keep_opps.extend(recovered_opps)
                drop_opps = [o for o in drop_opps if id(o) not in recovered_row_ids]
                recovered_ids = [o.opportunity_id for o in recovered_opps]
                print(f"[OpportunityMatcher] Dropped-Items Recovery Agent restored {len(recovered_opps)} opportunities (cap={RECOVERY_MAX_RESTORED})")
                print(f"[OpportunityMatcher] Recovery IDs: {recovered_ids}")
            else:
                print(f"[OpportunityMatcher] Dropped-Items Recovery Agent restored {len(recovered_opps)} opportunities (cap={RECOVERY_MAX_RESTORED})")

    final_opps_for_db = keep_opps

    if recovered_opps:
        kept_ids = {o.opportunity_id for o in keep_opps}
        dropped_ids = [o.opportunity_id for o in recovered_opps if o.opportunity_id not in kept_ids]
        if dropped_ids:
            print(f"[OpportunityMatcher] WARNING: Recovered opportunities later dropped: {dropped_ids}")
        else:
            print("[OpportunityMatcher] All recovered opportunities retained after finalize.")

    if keep_opps:
        print("[OpportunityMatcher] Running Opportunity Enrichment Agent on KEEP opportunities...")
        for o in keep_opps:
            source_events = select_enrichment_source_events(o, event_lookup)
            if not source_events:
                continue
            try:
                enriched = call_keep_enrichment_agent(
                    client=client.get_client(),
                    model=enrichment_model,
                    opportunity=o,
                    source_events=source_events,
                    max_output_tokens=max_output_tokens,
                    tracker=tracker,
                )
                if validate_enriched_keep_output(o, enriched, source_events=source_events):
                    evidence_type = opportunity_evidence_type(o) or ""
                    evidence_snippet = opportunity_evidence_snippet(o) or ""
                    supporting_snippet = enforce_verbatim_snippet(enriched.get("supporting_snippet"), max_words=30) or opportunity_supporting_snippet(o)
                    o.title = enriched["title"]
                    o.summary = enriched["summary"]
                    o.reason = build_reason_block(
                        enriched["reason"],
                        evidence_type,
                        evidence_snippet,
                        supporting_snippet,
                    )
                    o.evidence_type = normalize_key(evidence_type)
                    o.evidence_snippet = evidence_snippet
                    o.supporting_snippet = supporting_snippet
                    o.suggested_outreach_angle = enriched["suggested_outreach_angle"]
                    o.opportunity_details = enriched.get("opportunity_details")
            except Exception as e:
                print(f"[OpportunityMatcher] WARNING: KEEP enrichment failed for {o.opportunity_id}: {e}")

            if not (o.opportunity_details or "").strip():
                o.opportunity_details = build_fallback_opportunity_details(o, source_events)

        print("[OpportunityMatcher] Running Filter Agent...")

        for i in range(0, len(keep_opps), FILTER_BATCH_SIZE):
            batch = keep_opps[i:i+FILTER_BATCH_SIZE]

            try:
                results = call_filter_agent(
                    client=client.get_client(),
                    model=DEFAULT_FILTER_MODEL,
                    opp_batch=batch,
                    event_map=event_lookup,
                    chains=chains_map,
                    sectors=sectors_map,
                    seeking=seeking_map,
                    tracker=tracker,
                )

                results_map = {r.get("opportunity_id"): r for r in results
                    if isinstance(r, dict) and r.get("opportunity_id")
                }

                for o in batch:
                    r = results_map.get(o.opportunity_id)
                    if not r:
                        o.filter_chain = "chain-oth"
                        o.filter_sector = "sect-oth"
                        o.filter_seeking = "seek-oth"
                        continue

                    vals = r.get("filter_chain") or []
                    if isinstance(vals, str):
                        vals = [vals]
                    valid = list(dict.fromkeys(v for v in vals if v in chains_map))
                    o.filter_chain = ",".join(valid) if valid else "chain-oth"

                    vals = r.get("filter_sector") or []
                    if isinstance(vals, str):
                        vals = [vals]
                    valid = list(dict.fromkeys(v for v in vals if v in sectors_map))
                    o.filter_sector = ",".join(valid) if valid else "sect-oth"

                    vals = r.get("filter_seeking") or []
                    if isinstance(vals, str):
                        vals = [vals]
                    valid = list(dict.fromkeys(v for v in vals if v in seeking_map))
                    o.filter_seeking = ",".join(valid) if valid else "seek-oth"
            except Exception as e:
                print(f"[FilterAgent] Batch failed: {e}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    write_opportunities_csv(str(OUTPUT_DIR / f"opportunities_KEEP_{ts}.csv"), keep_opps, "KEEP")
    write_opportunities_csv(str(OUTPUT_DIR / f"opportunities_WATCHLIST_{ts}.csv"), watch_opps, "WATCHLIST")
    write_opportunities_csv(str(OUTPUT_DIR / f"opportunities_DROP_{ts}.csv"), drop_opps, "DROP")

    if all_gate_results:
        gatekeeper_output_file = str(OUTPUT_DIR / "gatekeeper_results.json")
        with open(gatekeeper_output_file, "w") as f:
            json.dump(all_gate_results, f, indent=2, default=_json_safe)
        print(f"[OpportunityMatcher] Saved {len(all_gate_results)} gatekeeper results to {gatekeeper_output_file}")

    if rejection_audit:
        rejection_output_file = str(OUTPUT_DIR / "rejection_audit.json")
        with open(rejection_output_file, "w") as f:
            json.dump(rejection_audit, f, indent=2, default=_json_safe)
        print(f"[OpportunityMatcher] Saved {len(rejection_audit)} rejection audit rows to {rejection_output_file}")

    if near_miss_watchlist:
        ts_nm = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        near_miss_csv = str(OUTPUT_DIR / f"near_miss_watchlist_{ts_nm}.csv")
        try:
            write_near_miss_watchlist_csv(near_miss_csv, near_miss_watchlist)
            print(f"[OpportunityMatcher] Saved near-miss CSV to {near_miss_csv}")
        except Exception as e:
            print(f"[OpportunityMatcher] WARNING: Failed to write near-miss watchlist artifacts: {e}")

    if keep_opps:
        saved = db.save_opportunities(final_opps_for_db)
        print(f"[OpportunityMatcher] Saved {saved} opportunities")
        return saved

    print("[OpportunityMatcher] No opportunities found")
    return 0

def process_events_batch_legacy(
    db: DatabaseManager,
    client: OpenAI,
    finder_model: str,
    gatekeeper_model: str,
    critic_model: str,
    refiner_model: str,
    events: List[Event],
    categories_map: dict,
    category_definitions: Dict[str, str],
    category_catalog: List[Dict[str, str]],
    chains_map: Dict[str, Any],
    sectors_map: Dict[str, Any],
    seeking_map: Dict[str, Any],
    max_events_per_batch: int,
    max_output_tokens: int,
    min_confidence: float,
    tracker: Optional[TokenTracker] = None,
) -> int:
    all_opps: List[Opportunity] = []
    all_gate_results: List[Dict[str, Any]] = []  # Track all gatekeeper outputs
    rejection_audit: List[Dict[str, Any]] = []   # Track downstream rejects (coverage/recall diagnostics)

    total_events = len(events)
    print(f"[OpportunityMatcher] Processing {total_events} events...")

    # IMPORTANT: Do NOT use the LLM batcher for production runs.
    # It can return a strict subset of events (effectively dropping items).
    # We intentionally keep all events by using deterministic chunking.
    print("[OpportunityMatcher] [Chunker] Creating fixed-size batches (no event dropping)...")
    intelligent_batches = [
        events[i : i + max_events_per_batch]
        for i in range(0, len(events), max_events_per_batch)
    ]
    total_in_batches = sum(len(b) for b in intelligent_batches)
    print(
        f"[OpportunityMatcher] [Chunker] Created {len(intelligent_batches)} batches | total_in_batches={total_in_batches} (expected={len(events)})\n"
    )

    def _gatekeeper_error_meta(err: Exception, batch_size: int) -> Dict[str, Any]:
        meta = {
            "type": type(err).__name__,
            "batch_size": batch_size,
            "err_len": len(str(err)) if err else 0,
        }
        for key in ("completion_tokens", "max_tokens", "raw_len"):
            if hasattr(err, key):
                meta[key] = getattr(err, key)
        return meta

    for batch_num, batch_events in enumerate(intelligent_batches, 1):
        print(f"[OpportunityMatcher] Batch {batch_num}/{len(intelligent_batches)} ({len(batch_events)} events)...")

        # 1) Gatekeeper
        try:
            gate_map = call_gatekeeper(client, gatekeeper_model, batch_events, tracker)
        except Exception as e:
            meta = _gatekeeper_error_meta(e, len(batch_events))
            print(f"  WARNING: Gatekeeper failed (details suppressed) | {meta}")
            continue

        # Collect gatekeeper results for debugging
        for event_id, gate_result in gate_map.items():
            event_obj = next((e for e in batch_events if e.event_id == event_id), None)
            all_gate_results.append({
                "batch_num": batch_num,
                "event_id": event_id,
                "event_title": event_obj.title if event_obj else "Unknown",
                "hard_exclusion": gate_result.hard_exclusion,
                "hard_exclusion_reason": gate_result.hard_exclusion_reason,
                "evidence_type": gate_result.evidence_type,
                "evidence_snippet": gate_result.evidence_snippet,
                "supporting_snippet": gate_result.supporting_snippet,
                "passes_gate": gate_result.passes(),
            })

        gated: List[Tuple[Event, GateResult]] = []
        excluded = 0
        rescued = 0
        for ev in batch_events:
            gr = gate_map.get(ev.event_id)
            if not gr:
                excluded += 1
                rejection_audit.append({
                    "stage": "gatekeeper",
                    "event_id": ev.event_id,
                    "event_title": ev.title or "",
                    "decision": "excluded",
                    "reason": "missing_gate_result",
                })
                continue
            rescued_gr = maybe_rescue_gatekeeper_media_rollout(ev, gr)
            if rescued_gr:
                gate_map[ev.event_id] = rescued_gr
                gr = rescued_gr
                rescued += 1
            if gr.passes():
                veto_reason = gatekeeper_postpass_veto_reason(ev, gr)
                if veto_reason:
                    gr.hard_exclusion = True
                    gr.hard_exclusion_reason = veto_reason
                    excluded += 1
                    rejection_audit.append({
                        "stage": "gatekeeper",
                        "event_id": ev.event_id,
                        "event_title": ev.title or "",
                        "decision": "excluded",
                        "reason": veto_reason,
                        "hard_exclusion": True,
                        "evidence_type": gr.evidence_type,
                    })
                    continue
                gated.append((ev, gr))
            else:
                excluded += 1
                rejection_audit.append({
                    "stage": "gatekeeper",
                    "event_id": ev.event_id,
                    "event_title": ev.title or "",
                    "decision": "excluded",
                    "reason": (
                        normalize_hard_exclusion_reason(gr.hard_exclusion_reason, gr.evidence_type)
                        if gr.hard_exclusion
                        else ("no_qualifying_evidence" if gr.evidence_type == "none" else "no_verbatim_evidence")
                    ),
                    "hard_exclusion": gr.hard_exclusion,
                    "evidence_type": gr.evidence_type,
                })

        for row in all_gate_results:
            if row.get("batch_num") != batch_num:
                continue
            updated = gate_map.get(row["event_id"])
            if not updated:
                continue
            row["hard_exclusion"] = updated.hard_exclusion
            row["hard_exclusion_reason"] = updated.hard_exclusion_reason
            row["evidence_type"] = updated.evidence_type
            row["evidence_snippet"] = updated.evidence_snippet
            row["supporting_snippet"] = updated.supporting_snippet
            row["passes_gate"] = updated.passes()

        rescue_suffix = f" | Rescued: {rescued}" if rescued else ""
        print(f"  [Gatekeeper] Passed: {len(gated)} | Excluded: {excluded}{rescue_suffix}")
        if not gated:
            continue

        # 2) Finder
        try:
            raw_opps = call_finder(client, finder_model, gated, categories_map, category_catalog, max_output_tokens, tracker)
        except OpenAIRequestTooLarge as e:
            print(f"  WARNING: Request too large: {e}")
            continue
        except Exception as e:
            print(f"  WARNING: Finder failed: {e}")
            continue

        # Filter for is_opportunity and compute confidence deterministically
        filtered: List[Dict[str, Any]] = []
        filtered_reasons = {"not_opportunity": 0, "low_confidence": 0}
        
        for o in raw_opps:
            if not o.get("is_opportunity", False):
                filtered_reasons["not_opportunity"] += 1
                print(f"    DEBUG: Filtered '{o.get('title', 'Untitled')[:50]}' - is_opportunity=False")
                rejection_audit.append({
                    "stage": "finder",
                    "decision": "filtered",
                    "reason": "is_opportunity_false",
                    "title": o.get("title", ""),
                    "target_company": o.get("target_company"),
                    "source_event_ids": o.get("source_event_ids", []),
                })
                continue
            sub = o.get("sub_scores", {}) or {}
            conf = confidence_from_subscores(sub)
            if conf < min_confidence:
                filtered_reasons["low_confidence"] += 1
                print(f"    DEBUG: Filtered '{o.get('title', 'Untitled')[:50]}' - confidence={conf:.3f} (sub_scores: {sub})")
                rejection_audit.append({
                    "stage": "finder",
                    "decision": "filtered",
                    "reason": "below_min_confidence",
                    "min_confidence": float(min_confidence),
                    "confidence": float(conf),
                    "sub_scores": sub,
                    "title": o.get("title", ""),
                    "target_company": o.get("target_company"),
                    "source_event_ids": o.get("source_event_ids", []),
                })
                continue
            o["confidence"] = conf
            filtered.append(o)

        print(f"  [Finder] Produced: {len(raw_opps)} | Kept after gate+confidence: {len(filtered)}")
        if len(raw_opps) > len(filtered):
            print(f"    Filtered: {filtered_reasons['not_opportunity']} (is_opportunity=False), {filtered_reasons['low_confidence']} (confidence<{min_confidence})")

        if not filtered:
            continue

        # 3) Critic
        print("  [Critic]", end=" ")
        feedback = call_critic(client, critic_model, filtered, tracker)
        rating = normalize_critic_rating(feedback.get("overall_rating", 0))
        print(f"rated {float(rating):.1f}/10")

        # Apply critic feedback: keep/reframe/discard
        final_items: List[Dict[str, Any]] = []
        fb_items = feedback.get("opportunity_feedback", []) or []
        for fb in fb_items:
            # Defensive parsing: some models may emit strings; try to parse or skip
            if isinstance(fb, str):
                try:
                    fb = json.loads(fb)
                except Exception:
                    continue
            if not isinstance(fb, dict):
                continue
            idx = int(fb.get("index", -1))
            if idx < 0 or idx >= len(filtered):
                continue
            status = fb.get("status")
            if status == "discard":
                rejection_audit.append({
                    "stage": "critic",
                    "decision": "discard",
                    "title": filtered[idx].get("title", ""),
                    "target_company": filtered[idx].get("target_company"),
                    "source_event_ids": filtered[idx].get("source_event_ids", []),
                    "feedback": fb.get("feedback", ""),
                })
                continue
            if status == "keep":
                # optionally adjust subscores
                corrected = fb.get("corrected_sub_scores")
                if corrected:
                    filtered[idx]["sub_scores"] = corrected
                    filtered[idx]["confidence"] = confidence_from_subscores(corrected)
                final_items.append(filtered[idx])
            if status == "reframe":
                ro = fb.get("reframed_opportunity")
                if ro and ro.get("is_opportunity", False):
                    corrected = fb.get("corrected_sub_scores") or ro.get("sub_scores")
                    ro["sub_scores"] = corrected
                    ro["confidence"] = confidence_from_subscores(corrected)
                    if ro["confidence"] >= min_confidence:
                        final_items.append(ro)

        # 4) Refiner (only if critic rating is not excellent or if critic discarded most)
        if rating < 8.5 or len(final_items) == 0:
            print("  [Refiner]", end=" ")
            try:
                refined = call_refiner(
                    client,
                    refiner_model,
                    gated,
                    final_items or filtered,
                    feedback,
                    categories_map,
                    category_catalog,
                    max_output_tokens,
                    tracker,
                )
                # Apply same post-filters
                refined_keep: List[Dict[str, Any]] = []
                for o in refined:
                    if not o.get("is_opportunity", False):
                        continue
                    conf = confidence_from_subscores(o.get("sub_scores", {}) or {})
                    if conf < min_confidence:
                        continue
                    o["confidence"] = conf
                    refined_keep.append(o)
                final_items = refined_keep
                print(f"-> {len(final_items)}")
            except Exception as e:
                print(f"FAILED ({e})")

        if not final_items:
            continue

        
        # 5) Reachability / Realism enforcement (client feedback)
        # Discard "pitch mega-institution" items unless the event text shows an explicit open motion
        # (RFP/procurement, applications open, partner portal/program, or explicit named partnership/integration).
        reach_filtered: List[Dict[str, Any]] = []
        for o in final_items:
            target = (o.get("target_company") or "").strip()
            title = (o.get("title") or "").strip()
            et = (o.get("evidence_type") or "").strip()
            es = enforce_verbatim_snippet(o.get("evidence_snippet")) or ""
            o["evidence_snippet"] = es

            mega = is_mega_counterparty(target) or is_mega_counterparty(title)
            if mega and not has_open_motion(et, es, title=title, target_company=target):
                # Drop from strict output (pilot credibility)
                continue

            reach_filtered.append(o)

        final_items = reach_filtered
        if not final_items:
            continue
# Convert to DB Opportunity objects
        time_found = datetime.now(timezone.utc).date().isoformat()
        for ro in final_items:
            event_ids_list = ro.get("source_event_ids", [])
            if not event_ids_list:
                continue

            sources: List[str] = []
            event_titles: List[str] = []
            event_urls: List[str] = []
            # Map sources and titles from batch
            for eid in event_ids_list:
                for ev in batch_events:
                    if ev.event_id == eid:
                        if ev.url:
                            sources.append(f"{ev.source}: {ev.url}" if ev.source else ev.url)
                            event_urls.append(ev.url)
                        elif ev.source:
                            sources.append(ev.source)
                        if ev.title:
                            event_titles.append(ev.title)
                        break

            if not event_titles:
                continue

            evidence_type = ro.get("evidence_type")
            evidence_snippet = enforce_verbatim_snippet(ro.get("evidence_snippet")) or ""
            reason_block = build_reason_block(
                ro.get("reason", ""),
                evidence_type,
                evidence_snippet,
                enforce_verbatim_snippet(ro.get("supporting_snippet"), max_words=30) or "",
            )

            recommended_action = ro.get("recommended_action") or ""

            validated_categories = validate_categories(
                ro.get("categories", []),
                evidence_type,
                evidence_snippet,
                categories_map,
                category_definitions,
            )
            if not validated_categories:
                rejection_audit.append({
                    "stage": "category_guardrails",
                    "decision": "discard",
                    "reason": "no_valid_categories_after_guardrails",
                    "title": ro.get("title", ""),
                    "target_company": ro.get("target_company"),
                    "evidence_type": evidence_type,
                    "evidence_snippet": evidence_snippet,
                    "raw_categories": ro.get("categories", []),
                })
                continue
            tc = normalize_target_company(ro.get("target_company"))

            if not tc:
                tc = derive_target_company(event_urls, sources)

            tc = clean_text(tc)

            opport = Opportunity(
                opportunity_id=str(uuid.uuid4()),
                title=ro.get("title", "").strip(),
                summary=(ro.get("reason", "").strip()[:800] or "Opportunity"),
                reason=reason_block,
                who_to_contact=ro.get("who_to_contact"),
                suggested_outreach_angle=recommended_action,
                categories=validated_categories,
                # ðŸ‘‡ Set deterministic fallback here
                filter_chain="chain-oth",
                filter_sector="sect-oth",
                filter_seeking="seek-oth",
                time_found=time_found,
                confidence=float(ro.get("confidence", 0)),
                tags=ro.get("tags", []),
                target_company=tc,
                sources=sources,
                event_ids="|".join(event_ids_list),
                event_titles="|".join(event_titles),
                event_url="|".join(event_urls),
                bd_weeks=int(TIME_HORIZON_WEEKS),
                evidence_type=normalize_key(evidence_type),
                evidence_snippet=clean_text(evidence_snippet) or "",
                supporting_snippet=enforce_verbatim_snippet(ro.get("supporting_snippet"), max_words=30) or "",
            )
            all_opps.append(opport)

        time.sleep(1.5)

    print(f"\n[OpportunityMatcher] Total opportunities (pre-dedupe): {len(all_opps)}")
    # Category Drift Monitoring (pre-dedupe)
    if all_opps:
        dist = Counter()
        for o in all_opps:
            for c in (o.categories or []):
                dist[c] += 1
        print("\n[OpportunityMatcher] Category Distribution (Drift Monitor, pre-dedupe):")
        for cat, count in sorted(dist.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {cat}: {count}")

    all_opps = dedupe_opportunities(all_opps)
    print(f"[OpportunityMatcher] Total opportunities (post-dedupe): {len(all_opps)}")

    # Finalizer step (post-processing): consolidate near-duplicates and apply KEEP/WATCHLIST/DROP policy.
    keep_opps, watch_opps, drop_opps = finalize_opportunities(all_opps)
    print(
        f"[OpportunityMatcher] Finalizer: KEEP={len(keep_opps)} | WATCHLIST={len(watch_opps)} | DROP={len(drop_opps)}"
    )

    event_lookup = {ev.event_id: ev for ev in events}
    if watch_opps:
        keep_opps, watch_opps = promote_watchlist_candidates(
            client=client,
            model=DEFAULT_WATCHLIST_PROMOTION_MODEL,
            watch_opps=watch_opps,
            keep_opps=keep_opps,
            event_lookup=event_lookup,
            category_catalog=category_catalog,
            categories_map=categories_map,
            category_definitions=category_definitions,
            max_output_tokens=max_output_tokens,
            tracker=tracker,
            rejection_audit=rejection_audit,
        )
        print(
            f"[OpportunityMatcher] Post-WATCHLIST promotion: KEEP={len(keep_opps)} | "
            f"WATCHLIST={len(watch_opps)} | DROP={len(drop_opps)}"
        )

    # Finalizer enforcement: DROP items are never saved as opportunities for MVP.
    final_opps_for_db = keep_opps

# ---------------------------
# FILTER AGENT (Batch Mode)
# ---------------------------

##^ commented on 24-Feb-> chains_map, sectors_map, seeking_map = load_filter_master(str(FILTER_MASTER_JSON_PATH))

    if keep_opps:
        print("[OpportunityMatcher] Running Filter Agent...")

        for i in range(0, len(keep_opps), FILTER_BATCH_SIZE):
            batch = keep_opps[i:i+FILTER_BATCH_SIZE]

            try:
                results = call_filter_agent(
                    client=client,
                    model=DEFAULT_FILTER_MODEL,
                    opp_batch=batch,
                    event_map=event_lookup,
                    chains=chains_map,
                    sectors=sectors_map,
                    seeking=seeking_map,
                    tracker=tracker,
                )

                results_map = {r.get("opportunity_id"): r for r in results
                    if isinstance(r, dict) and r.get("opportunity_id")
                }

                for o in batch:
                    r = results_map.get(o.opportunity_id)
                    if not r:
                        o.filter_chain = "chain-oth"
                        o.filter_sector = "sect-oth"
                        o.filter_seeking = "seek-oth"
                        continue

                    # ----- CHAIN -----
                    vals = r.get("filter_chain") or []
                    if isinstance(vals, str):
                        vals = [vals]
                    valid = list(dict.fromkeys(v for v in vals if v in chains_map))
                    o.filter_chain = ",".join(valid) if valid else "chain-oth"

                    # ----- SECTOR -----
                    vals = r.get("filter_sector") or []
                    if isinstance(vals, str):
                        vals = [vals]
                    valid = list(dict.fromkeys(v for v in vals if v in sectors_map))
                    o.filter_sector = ",".join(valid) if valid else "sect-oth"

                    # ----- SEEKING -----
                    vals = r.get("filter_seeking") or []
                    if isinstance(vals, str):
                        vals = [vals]                    
                    valid = list(dict.fromkeys(v for v in vals if v in seeking_map))
                    o.filter_seeking = ",".join(valid) if valid else "seek-oth"
            except Exception as e:
                print(f"[FilterAgent] Batch failed: {e}")

# ---------------------------
# END  OF  FILTER AGENT (Batch Mode)
# ---------------------------

    # Write operator-friendly CSVs (does not affect DB writes).
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    write_opportunities_csv(str(OUTPUT_DIR / f"opportunities_KEEP_{ts}.csv"), keep_opps, "KEEP")
    write_opportunities_csv(str(OUTPUT_DIR / f"opportunities_WATCHLIST_{ts}.csv"), watch_opps, "WATCHLIST")
    write_opportunities_csv(str(OUTPUT_DIR / f"opportunities_DROP_{ts}.csv"), drop_opps, "DROP")

    rejection_audit = enrich_rejection_audit_rows(rejection_audit, events, all_gate_results)

    # Save Gatekeeper results to JSON for auditing
    if all_gate_results:
        gatekeeper_output_file = str(OUTPUT_DIR / "gatekeeper_results.json")
        with open(gatekeeper_output_file, "w") as f:
            json.dump(all_gate_results, f, indent=2, default=_json_safe)
        print(f"[OpportunityMatcher] Saved {len(all_gate_results)} gatekeeper results to {gatekeeper_output_file}")

    # Save rejection audit for recall/coverage diagnostics (does not affect DB writes)
    if rejection_audit:
        rejection_output_file = str(OUTPUT_DIR / "rejection_audit.json")
        with open(rejection_output_file, "w") as f:
            json.dump(rejection_audit, f, indent=2, default=_json_safe)
        print(f"[OpportunityMatcher] Saved {len(rejection_audit)} rejection audit rows to {rejection_output_file}")

    if keep_opps:
        saved = db.save_opportunities(final_opps_for_db)
        print(f"[OpportunityMatcher] Saved {saved} opportunities")
        return saved

    print("[OpportunityMatcher] No opportunities found")
    return 0


# ---------------------------
# MAIN  Entry Point
# ---------------------------

def main() -> int:
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description="Event-centric opportunity classifier with client-gate enforcement")
    parser.add_argument("--finder-model", default=DEFAULT_MODEL, help="Model for Finder")
    parser.add_argument("--gatekeeper-model", default=DEFAULT_GATEKEEPER_MODEL, help="Model for Gatekeeper")
    parser.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL, help="Model for Critic")
    parser.add_argument("--enrichment-model", default=DEFAULT_ENRICHMENT_MODEL, help="Model for KEEP Opportunity Enrichment")
    parser.add_argument("--refiner-model", default=DEFAULT_REFINER_MODEL, help="Model for Refiner")
    parser.add_argument("--days-back", type=int, default=30, help="Days of events to analyze")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Optional JSONL input path. Default: Final_events.jsonl in the Data folder.",
    )
    parser.add_argument("--max-events-per-batch", type=int, default=DEFAULT_MAX_EVENTS_PER_BATCH, help="Max events per batch")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Max output tokens")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE, help="Minimum confidence threshold")
    parser.add_argument(
        "--time-horizon-weeks",
        type=int,
        default=12,
        choices=[2, 4, 5, 10, 12, 15, 20, 25, 30],
        help="Time horizon (in weeks) enforced across ALL agent prompts (1â€“N weeks). Allowed: 2, 4, 5, 10, 12, 15, 20, 25, 30.",
    )
    parser.add_argument(
        "--candidate-mode",
        default="expanded",
        choices=["strict", "expanded"],
        help="strict=client-gate only; expanded=more candidates via lower thresholds/prompting",
    )

    args = parser.parse_args()

    # Time horizon configuration (applies to ALL agents via prompt templating)
    horizon_weeks = int(args.time_horizon_weeks)
    global TIME_HORIZON_WEEKS, TIME_HORIZON_STR
    TIME_HORIZON_WEEKS = horizon_weeks
    TIME_HORIZON_STR = f"1â€“{horizon_weeks} weeks"

    # Candidate mode tuning
    if args.candidate_mode == "expanded":
        # Keep confidence strict; only increase completion budget
        if args.max_output_tokens < 2800:
            args.max_output_tokens = 2800

    print(f"[OpportunityMatcher] Script started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("[OpportunityMatcher] Mode: EVENT-CENTRIC (client-gate compliant)")
    print(
        f"[OpportunityMatcher] Models: Finder={args.finder_model}, Gatekeeper={args.gatekeeper_model}, Critic={args.critic_model}, Enrichment={args.enrichment_model}, Refiner={args.refiner_model}, Recovery={DEFAULT_RECOVERY_MODEL}, WatchlistPromotion={DEFAULT_WATCHLIST_PROMOTION_MODEL}, Filter={getattr(args, 'filter_model', DEFAULT_FILTER_MODEL)}"
    )

    if load_dotenv:
        load_dotenv()

    api_keys = load_openai_api_keys()
    if not api_keys:
        print("[OpportunityMatcher] ERROR: no OpenAI API keys found. Set OPENAI_API_KEY_1..3 or OPENAI_API_KEY.")
        return 1
    print(f"[OpportunityMatcher] OpenAI key pool size: {len(api_keys)}")

    if OpenAI is None:
        print("[OpportunityMatcher] ERROR: openai package is not installed")
        return 1

    try:
        client = ThreadLocalOpenAIClientPool(api_keys)
    except Exception as e:
        print(f"[OpportunityMatcher] ERROR: Failed to initialize OpenAI client: {e}")
        return 1

    try:
        config_path = SCRIPT_DIR / "Database_Connection.csv"
        db_config = _read_database_config(str(config_path))
    except Exception as e:
        print(f"[OpportunityMatcher] ERROR reading database config: {e}")
        return 1

    db_schema = db_config["DB_SCHEMA"]
    db = DatabaseManager(
        db_config["DB_HOST"],
        int(db_config["DB_PORT"]),
        db_config["DB_USER"],
        db_config["DB_PASSWORD"],
        db_config["DB_NAME"],
        schema=db_schema,
    )

    print(
        "[OpportunityMatcher] Connected to database: "
        f"host={db_config['DB_HOST']} port={db_config['DB_PORT']} "
        f"dbname={db_config['DB_NAME']} schema={db_schema} user={db_config['DB_USER']}"
    )
    # Helpful diagnostics; especially useful when the script unexpectedly loads 0 events/categories.
    db.debug_db_snapshot()

    tracker = TokenTracker()
    # Load events from JSONL (Model2_v12 behavior)
    events_path = str(FINAL_EVENTS_JSONL_PATH) if not str(args.input).strip() else str(args.input).strip()
    try:
        events = load_events_from_jsonl(events_path, days_back=args.days_back)
    except Exception as e:
        print(f"[OpportunityMatcher] ERROR: Failed to load events from JSONL: {e}")
        return 1
    print(f"[OpportunityMatcher] Loaded {len(events)} events from {events_path}")

    # --------------------------------------------------
    # HARD DEDUPE BY event_id (pre-batching integrity)
    # --------------------------------------------------

    unique_map = {}
    for e in events:
        key = (e.event_id or "").strip().lower()
        if key and key not in unique_map:
            unique_map[key] = e

    events = list(unique_map.values())

    print(f"[OpportunityMatcher] After event_id dedupe: {len(events)} events")

    # Load categories from category_master.json; fall back to DB categories if file missing/empty
    categories_map, category_definitions, category_names = load_categories_map(str(CATEGORY_MASTER_JSON_PATH))
    if categories_map:
        print(f"[OpportunityMatcher] Loaded {len(categories_map)} category aliases from {CATEGORY_MASTER_JSON_PATH}")
    else:
        # DB returns {category_text_id: category_name}; build a compatible aliases map
        db_cats = db.get_categories()
        categories_map = {}
        category_names = {}
        category_definitions = {}
        for text_id, name in (db_cats or {}).items():
            text_id = str(text_id or "").strip()
            name = str(name or "").strip()
            if not text_id:
                continue
            categories_map[text_id] = text_id
            if name:
                categories_map[name] = text_id
                category_names[text_id] = name
        print(f"[OpportunityMatcher] Loaded {len(categories_map)} category aliases from DB (no definitions)")

    # Build catalog for prompts: list of {category_text_id, category_name, definition}
    # Only include canonical ids (keys from category_names or category_definitions or self-mapped ids)
    canonical_ids = set()
    canonical_ids.update(category_names.keys())
    canonical_ids.update(category_definitions.keys())
    for k, v in (categories_map or {}).items():
        if k == v:
            canonical_ids.add(v)

    category_catalog = []
    for cid in sorted(canonical_ids):
        category_catalog.append({
            "category_text_id": cid,
            "category_name": category_names.get(cid, ""),
            "definition": category_definitions.get(cid, ""),
        })

    # --------------------------------------------------
    # Load Filter Master (startup configuration)
    # --------------------------------------------------

    print("[OpportunityMatcher] Loading filter master configuration...")

    chains_map, sectors_map, seeking_map = load_filter_master(str(FILTER_MASTER_JSON_PATH))

    print( f"[OpportunityMatcher] Filter Master Ready | "
        f"chains={len(chains_map)} | "
        f"sectors={len(sectors_map)} | "
        f"seeking={len(seeking_map)}"
    )

    total = process_events_batch(
        db=db,
        client=client,
        finder_model=args.finder_model,
        gatekeeper_model=args.gatekeeper_model,
        critic_model=args.critic_model,
        enrichment_model=args.enrichment_model,
        refiner_model=args.refiner_model,
        events=events,
        categories_map=categories_map,
        category_definitions=category_definitions,
        category_catalog=category_catalog,
        chains_map=chains_map,
        sectors_map=sectors_map,
        seeking_map=seeking_map,
        max_events_per_batch=args.max_events_per_batch,
        max_output_tokens=args.max_output_tokens,
        min_confidence=args.min_confidence,
        tracker=tracker,
    )

    elapsed = time.time() - script_start_time
    print(f"\n[OpportunityMatcher] COMPLETE Total opportunities saved: {total}")
    print(f"[OpportunityMatcher] Execution time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")

    tracker.print_summary()
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

