#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
The DB schema in this script (opportunities_filter) does not include
separate columns for is_opportunity/evidence_snippet/recommended_action.
To keep the script deployable without migrations, this implementation maps:
- reason: stored as a structured multi-line string including evidence + reason
- suggested_outreach_angle: stored as recommended_action (only for TRUE)

If you add DB columns later, update Opportunity + save_opportunities accordingly.
"""

from __future__ import annotations
from urllib.parse import urlparse

import argparse
import csv
import threading
import json
import os
import re
import time
import traceback
import unicodedata
import uuid
from difflib import SequenceMatcher
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


def normalize_target_company(raw: Optional[str]) -> Optional[str]:
    """
    Enforce single canonical target company.
    If multiple companies are provided, keep the first non-empty token.
    """
    if not raw:
        return None

    text = clean_text(raw) or ""
    if not text:
        return None

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\-\:\;\,\.\s]+|[\-\:\;\,\.\s]+$", "", text)
    text = re.sub(r"\b(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\((?:[^()]|\([^()]*\))*\)\s*$", "", text).strip()
    text = re.sub(
        r"\s+(?:team|foundation|dao|community|ecosystem|governance|risk committee|treasury subdao)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Split only on stronger company separators, not on "/" which appears in legitimate names.
    parts = re.split(r"\s*(?:\||;|,|(?:\s+(?:and|&)\s+))\s*", text)

    for p in parts:
        candidate = re.sub(r"\s+", " ", (p or "").strip()).strip(" -:;,.")
        if candidate and candidate.lower() not in _NULL_LIKE:
            return candidate

    return None


def clean_text(v: str | None) -> str | None:
    if v is None:
        return None
    s = str(v)

    # Attempt a lightweight mojibake repair for common UTF-8->latin1 artifacts.
    if any(bad in s for bad in ("Ã¢", "Ãƒ", "Ã‚")):
        try:
            repaired = s.encode("latin1").decode("utf-8")
            if repaired:
                before_bad = sum(s.count(ch) for ch in ("Ã¢", "Ãƒ", "Ã‚"))
                after_bad = sum(repaired.count(ch) for ch in ("Ã¢", "Ãƒ", "Ã‚"))
                if after_bad < before_bad:
                    s = repaired
        except Exception:
            pass

    # Normalize Unicode form to reduce mixed glyph variants.
    s = unicodedata.normalize("NFKC", s)

    # Replace common mojibake and smart punctuation with plain ASCII.
    replacements = {
        "Ã¢â‚¬â€œ": "-",
        "Ã¢â‚¬â€": "-",
        "Ã¢â‚¬Ëœ": "'",
        "Ã¢â‚¬â„¢": "'",
        "Ã¢â‚¬Å“": '"',
        "Ã¢â‚¬\x9d": '"',
        "Ã¢â‚¬Â¦": "...",
        "Ãƒâ€”": "x",
        "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“": "-",
        "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢": "'",
        "ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ": '"',
        "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â": '"',
        "Ã‚": "",
        "â€“": "-",
        "â€”": "-",
        "â€˜": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    # Remove zero-width/control chars except line breaks and tabs.
    s = "".join(ch for ch in s if ch in ("\n", "\r", "\t") or ord(ch) >= 32)
    s = re.sub(r"[ \t]+", " ", s).strip()
    if s.lower() in _NULL_LIKE:
        return None
    return s


def smart_truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[: max_chars - 3].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return (cut or text[: max_chars - 3].rstrip()) + "..."


def normalize_title_text(title: Optional[str], max_chars: int = 140) -> str:
    title = (clean_text(title) or "").strip()
    title = re.sub(r"^\s*Opportunity\s*:\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" -:;,.")
    if len(title) > max_chars:
        title = smart_truncate(title, max_chars).strip(" -:;,.")
    return title or "Opportunity"


def _normalize_sentence(text: Optional[str]) -> str:
    s = re.sub(r"\s+", " ", (clean_text(text) or "").strip()).strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def normalize_summary_text(summary: Optional[str], fallback_parts: List[Optional[str]], min_words: int = 35, max_chars: int = 600) -> str:
    sentences = [_normalize_sentence(s) for s in _SENT_SPLIT_RE.split((clean_text(summary) or "").strip()) if _normalize_sentence(s)]
    extras: List[str] = []
    for part in fallback_parts:
        for sentence in _SENT_SPLIT_RE.split((clean_text(part) or "").strip()):
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
    text = re.sub(r"\s+", " ", (clean_text(reason) or "").strip()).strip()
    return smart_truncate(text, max_chars) if text else ""


def normalize_outreach_text(outreach: Optional[str], fallback: Optional[str] = None, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", (clean_text(outreach) or clean_text(fallback) or "").strip()).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return smart_truncate(text, max_chars) if text else ""


def normalize_details_text(details: Optional[str], min_words: int = 70, max_chars: int = 1600) -> Optional[str]:
    text = re.sub(r"\s+", " ", (clean_text(details) or "").strip()).strip()
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
    text = clean_text(reason_block) or ""
    m = re.search(r"Reason:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return re.sub(r"\s+", " ", text).strip()
def derive_target_company(event_urls: list[str], sources: list[str]) -> str | None:
    def _label_from_host(host: str) -> Optional[str]:
        host = (host or "").strip().lower()
        if not host:
            return None
        host = re.sub(r"^www\d*\.", "", host)
        parts = [p for p in host.split(".") if p]
        if len(parts) >= 2:
            core = parts[-2]
            if core in {"co", "com", "org", "net", "gov", "edu", "io", "xyz"} and len(parts) >= 3:
                core = parts[-3]
        elif parts:
            core = parts[0]
        else:
            return None
        core = core.strip("-_")
        if not core:
            return None
        tokens = [tok for tok in re.split(r"[-_]+", core) if tok]
        if not tokens:
            return None
        return " ".join(tok.upper() if len(tok) <= 3 else tok.capitalize() for tok in tokens)

    # 1) Prefer a readable label from URL domain (deterministic, no hardcoding)
    for u in (event_urls or []):
        u = (u or "").strip()
        if not u:
            continue
        try:
            host = urlparse(u).netloc.strip().lower()
            if host:
                labeled = _label_from_host(host)
                if labeled:
                    return labeled
                return host
        except Exception:
            continue

    # 2) Fallback: source label
    for s in (sources or []):
        normalized = normalize_target_company(s)
        if not normalized:
            continue
        if ":" in normalized:
            left = normalize_target_company(normalized.split(":", 1)[0])
            if left:
                return left
        return normalized

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


def append_crash_log(context: str, exc: Exception) -> None:
    """Persist fatal/non-fatal crash details to a local log for postmortems."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        crash_log_path = OUTPUT_DIR / "opportunitymatcher_crash.log"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(crash_log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {context}: {type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc())
            f.write("\n" + ("=" * 80) + "\n")
        print(f"[OpportunityMatcher] Crash details logged to {crash_log_path}")
    except Exception as log_exc:
        print(f"[OpportunityMatcher] WARNING: Failed to write crash log ({log_exc})")
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
        raise FileNotFoundError(
            f"events JSONL not found: {path_str}. "
            f"Expected preprocess output such as Final_events.jsonl under Data/ or data/."
        )

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
            summary = obj.get("summary_text") or obj.get("summary") or obj.get("rss_summary_text") or ""
            body = obj.get("body_text") or obj.get("description") or obj.get("content") or ""
            source = obj.get("source") or obj.get("source_name") or ""
            url = obj.get("url") or obj.get("link") or ""
            ai_enrichment = obj.get("ai_enrichment") if isinstance(obj.get("ai_enrichment"), dict) else {}
            bd_signal = obj.get("bd_signal") if isinstance(obj.get("bd_signal"), dict) else ai_enrichment.get("bd_signal")

            events.append(Event(
                event_id=str(event_id),
                title=title,
                summary=summary,
                description=body,
                tags=[],
                published_at=dt.isoformat() if dt else (published_raw or ""),
                source=source,
                url=url,
                bd_signal=bd_signal if isinstance(bd_signal, dict) else None,
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

AUDIT_STATUS_FOLLOWUP_RE = re.compile(
    r"\b(submitted (?:a )?request|submitted update request|already submitted|when (?:this|it) will be reviewed|"
    r"review timing|confirm receipt|in[- ]flight submission)\b",
    re.IGNORECASE,
)
AUDIT_SELF_APPLICATION_RE = re.compile(
    r"\b(active application|has publicly applied|is applying for|applying for the .*operator|"
    r"operator admission|trusted operator|professional operator|self-?nominate|self nomination)\b",
    re.IGNORECASE,
)
AUDIT_OPEN_GRANT_PROPOSAL_RE = re.compile(
    r"\bopen grant proposal\b",
    re.IGNORECASE,
)
AUDIT_FORUM_SUPPORT_RE = re.compile(
    r"\b(seeking support/funding|seeking support|support/funding|forum request|request for support|request for funding)\b",
    re.IGNORECASE,
)
AUDIT_FORUM_QUESTION_RE = re.compile(
    r"\b(whether .* program exists|does .* program exist|asking whether|formal integration/onboarding program)\b",
    re.IGNORECASE,
)
AUDIT_NOMINATION_RE = re.compile(
    r"\b(listing committee|committee season|nomination window|self-?nominate|self nomination)\b",
    re.IGNORECASE,
)
AUDIT_TOOLING_RELEASE_RE = re.compile(
    r"\b(cli|command line|sdk docs|public launch|developer tooling)\b",
    re.IGNORECASE,
)
AUDIT_NARROW_GRANT_THREAD_RE = re.compile(
    r"\b(grant thread|grant application|offer delivery support|implementation support|offer collaboration/support)\b",
    re.IGNORECASE,
)
AUDIT_KEEP_OVERRIDE_RE = re.compile(
    r"\b(webhooks?|sdk|data feeds?|data streams?|stablecoin|wallet-as-a-service|early partners?|"
    r"cohort|integrat(?:e|ion)|onboard|adopt|renewal|audit|bug bounty)\b",
    re.IGNORECASE,
)

ELLIPSIS_RE = re.compile(r"\.{3,}")
WHITESPACE_RE = re.compile(r"\s+")

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
    if evidence_type in {"grant_or_rfp_or_program_open", "vendor_or_procurement_need", "token_launch_or_tge", "tooling_or_standards_implementation_open"}:
        return True
    if evidence_type == "support_added_with_bd_action" and OPEN_MOTION_RE.search(evidence_snippet or ""):
        return True
    # Partnerships are only acceptable for mega counterparties if it is an explicit named partnership/integration.
    if evidence_type == "partnership_or_integration":
        return True
    return False


def audit_guardrail_bucket(
    opportunity: "Opportunity",
    full_text: str,
    evidence_type: str,
    evidence_snippet: str,
    duplicate_target_present: bool,
) -> Optional[str]:
    text = " ".join(
        [
            clean_text(opportunity.title) or "",
            clean_text(opportunity.summary) or "",
            clean_text(full_text) or "",
            clean_text(evidence_snippet) or "",
        ]
    )
    if not text:
        return None

    explicit_open_motion = has_open_motion(
        evidence_type,
        evidence_snippet,
        title=opportunity.title,
        target_company=opportunity.target_company or "",
    )
    keep_override = bool(AUDIT_KEEP_OVERRIDE_RE.search(text))
    explicitly_open_to_many = bool(
        re.search(
            r"\b(validators? can apply|applications? open|open to external|open to developers|"
            r"open to all|registration(?: now)? open|submit a proposal|rolling submissions)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    concrete_procurement = bool(
        re.search(
            r"\b(rfp|request for proposal|vendor selection|external auditor|procurement|"
            r"renewal|services proposal|audit kickoff|bug bounty)\b",
            text,
            flags=re.IGNORECASE,
        )
    )

    if AUDIT_NOMINATION_RE.search(text):
        return "drop"

    if AUDIT_STATUS_FOLLOWUP_RE.search(text):
        return "drop"

    if AUDIT_OPEN_GRANT_PROPOSAL_RE.search(text):
        return "drop"

    if AUDIT_SELF_APPLICATION_RE.search(text) and not explicitly_open_to_many:
        return "drop"

    if AUDIT_FORUM_SUPPORT_RE.search(text) and not concrete_procurement:
        return "drop"

    if AUDIT_FORUM_QUESTION_RE.search(text) and not explicit_open_motion:
        return "drop"

    if duplicate_target_present and AUDIT_TOOLING_RELEASE_RE.search(text) and not explicit_open_motion and not keep_override:
        return "drop"

    if AUDIT_NARROW_GRANT_THREAD_RE.search(text) and not concrete_procurement and not keep_override:
        return "watch"

    return None

def enforce_verbatim_snippet(snippet: Optional[str], max_words: int = 40) -> Optional[str]:
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



def slice_body_for_gatekeeper(desc: Optional[str], head: int = 3500, tail: int = 1500) -> str:
    """Return first `head` + last `tail` characters of desc for Gatekeeper grounding.
    Total default size = 5,000 chars (3,500 head + 1,500 tail), preserving top context and bottom intake/contact lines.
    If desc is shorter than head+tail, return as-is.
    """
    s = (desc or "")
    if len(s) <= head + tail:
        return s
    return s[:head] + "\n...\n" + s[-tail:]

def _compact_gatekeeper_text(s: Optional[str], max_chars: int) -> str:
    """Trim and normalize whitespace for gatekeeper payload fields."""
    t = WHITESPACE_RE.sub(" ", (s or "")).strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip()

def slice_body_for_finder(desc: Optional[str], head: int = 4000, tail: int = 4000) -> str:
    """Return first `head` + last `tail` characters of desc (captures contact/boilerplate at end).
    If desc is shorter than head+tail, return as-is.
    """
    s = (desc or "")
    if len(s) <= head + tail:
        return s
    return s[:head] + "\n...\n" + s[-tail:]


_ACTION_SURFACE_RE = re.compile(
    r"\b(applications? open|application thread|apply now|apply|open grant|grants? program|rfp|request for proposal|"
    r"proposal category|intake|early partners?|partner program|looking for partner projects|seeking collaborators|"
    r"admission|operator status|professional operator|trusted operator|applying for|renewal|services proposal|"
    r"annual engagement|fund an audit|cover an audit|audit funding|appetite to add|"
    r"builders? can|integrators? can|partners? can now|issuers? can|banks? can|fintechs? can|"
    r"co-design|co-fund|co-run|projects? can|operators? can|validators? can|delegators? can|"
    r"deploy|deployment|list(?:ing)?|monitor(?:ing)?|extend|sdk|api|webhooks?|data streams?|dashboard|"
    r"light nodes?|passkey|smart accounts?|framework|security checks?|protection|scam checks?|"
    r"codebase|docs?|documentation|openapi|swagger|repository|rest api|demo api|live demo|"
    r"open[ -]sourc(?:e|ed)|publicly available|token standard|privacy pool|address poisoning|"
    r"address poisoning protection|scam address|screening|scam(?:-| )detection|governance data|governance api|data api|data pipeline|delegator dashboard|"
    r"integrate|integration|onboard|onboarding|settle|embed|connect|route|claim rewards?|"
    r"stake now|connect your wallet|swap(?: for)?|referral link)\b",
    flags=re.IGNORECASE,
)

_STRONG_ACTION_SURFACE_RE = re.compile(
    r"\b(connect your wallet|stake now|claim rewards?|swap(?: for)?|referral link|"
    r"live demo (?:api|endpoint)|rest api|openapi|swagger|codebase|sdk|documentation|docs|"
    r"publicly available|open[ -]sourc(?:e|ed)|github repository|repository|tagged release|"
    r"builders? can|projects? can|operators? can|integrators? can|issuers? can|banks? can|"
    r"fintechs? can|deploy|integrate|onboard|settle|embed|connect|route|extend|"
    r"token standard|privacy pool|address poisoning|address poisoning protection|scam address|screening|scam(?:-| )detection|"
    r"governance data|governance api|data api|data pipeline|delegator dashboard)\b",
    flags=re.IGNORECASE,
)

_WEAK_PARTICIPATION_RE = re.compile(
    r"\b(feedback|have a say|discussion|retrospective|ideas|community feedback|"
    r"vote|comment|survey|interview|user testing|user research)\b",
    flags=re.IGNORECASE,
)


def _action_sentence_score(sent: str) -> int:
    s = (sent or "").strip()
    if not s:
        return -999

    score = 0
    if _ACTION_SURFACE_RE.search(s):
        score += 2
    if _STRONG_ACTION_SURFACE_RE.search(s):
        score += 4

    lowered = s.lower()
    for needle in ("sdk", "api", "docs", "documentation", "codebase", "openapi", "swagger", "repository"):
        if needle in lowered:
            score += 1
    if "publicly available" in lowered or "open source" in lowered or "open-sourced" in lowered:
        score += 2
    if "live demo" in lowered or "demo api" in lowered:
        score += 2
    if "token standard" in lowered or "privacy pool" in lowered:
        score += 2
    if "address poisoning" in lowered or "address poisoning protection" in lowered or "scam address" in lowered or "screening" in lowered or "scam detection" in lowered:
        score += 2
    if "dashboard" in lowered or "delegator dashboard" in lowered or "governance data" in lowered or "governance api" in lowered or "data api" in lowered:
        score += 2
    if "connect your wallet" in lowered or "stake now" in lowered or "claim rewards" in lowered or "swap" in lowered:
        score += 2

    if _WEAK_PARTICIPATION_RE.search(s) and not _STRONG_ACTION_SURFACE_RE.search(s):
        score -= 4

    return score


def _extract_relevant_excerpt(text: Optional[str], anchor: str = "", max_chars: int = 700) -> str:
    cleaned = re.sub(r"\s+", " ", (clean_text(text) or "").strip()).strip()
    if not cleaned:
        return ""
    if anchor:
        norm_anchor = re.sub(r"\s+", " ", anchor).strip()
        idx = cleaned.lower().find(norm_anchor.lower())
        if idx >= 0:
            start = max(0, idx - 220)
            end = min(len(cleaned), idx + len(norm_anchor) + 420)
            return smart_truncate(cleaned[start:end].strip(), max_chars)
    sentences = re.split(r"(?<=[\.\!\?])\s+", cleaned)
    best_sent = ""
    best_score = -999
    for sent in sentences:
        sent = sent.strip()
        if not _ACTION_SURFACE_RE.search(sent):
            continue
        score = _action_sentence_score(sent)
        if score > best_score:
            best_score = score
            best_sent = sent
    if best_sent:
        return smart_truncate(best_sent, max_chars)
    return smart_truncate(cleaned, max_chars)


def _extract_action_surface(texts: List[Optional[str]], evidence_snippet: str = "", max_chars: int = 420) -> str:
    candidates: List[str] = []
    if evidence_snippet:
        candidates.append(evidence_snippet)
    for text in texts:
        excerpt = _extract_relevant_excerpt(text, evidence_snippet, max_chars=max_chars)
        if excerpt:
            candidates.append(excerpt)
    for item in candidates:
        if _ACTION_SURFACE_RE.search(item):
            return smart_truncate(item, max_chars)
    return smart_truncate(candidates[0], max_chars) if candidates else ""


def build_gatekeeper_action_hint(ev: "Event", max_chars: int = 500) -> str:
    """Return the strongest likely action sentence to help Gatekeeper scan long events.

    This is only decision support. It must never override the original event text.
    """
    candidates: List[str] = []
    title = clean_text(ev.title) or ""
    summary = clean_text(ev.summary) or ""
    desc = slice_body_for_gatekeeper(ev.description, head=1800, tail=1000)

    if title and _ACTION_SURFACE_RE.search(title):
        candidates.append(title)
    for text in [desc, summary]:
        excerpt = _extract_relevant_excerpt(text, "", max_chars=max_chars)
        if excerpt:
            candidates.append(excerpt)

    seen: Set[str] = set()
    ordered: List[str] = []
    for item in candidates:
        norm = normalize_key(item)
        if norm and norm not in seen:
            seen.add(norm)
            ordered.append(item)

    best_item = ""
    best_score = -999
    for item in ordered:
        if not _ACTION_SURFACE_RE.search(item):
            continue
        score = _action_sentence_score(item)
        if score > best_score:
            best_score = score
            best_item = item
    if best_item:
        return smart_truncate(best_item, max_chars)
    return smart_truncate(ordered[0], max_chars) if ordered else ""


def build_event_evidence_package(ev: "Event", evidence_snippet: str, evidence_type: str = "") -> Dict[str, Any]:
    clean_title = clean_text(ev.title) or ""
    clean_summary = smart_truncate(clean_text(ev.summary) or "", 700)
    body_excerpt = _extract_relevant_excerpt(ev.description, evidence_snippet, max_chars=900)
    action_surface = _extract_action_surface([ev.summary, ev.description], evidence_snippet, max_chars=420)
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


def build_finder_event_payload(ev: "Event", gr: "GateResult") -> Dict[str, Any]:
    """Construct a grounded but lighter-weight payload for Finder after Gatekeeper has passed the event."""
    evidence_package = build_event_evidence_package(ev, gr.evidence_snippet or "", gr.evidence_type)
    body_excerpt = evidence_package.get("strong_body_excerpt") or ""
    action_surface = evidence_package.get("explicit_action_surface") or ""
    body_focus = _extract_relevant_excerpt(ev.description, gr.evidence_snippet or "", max_chars=1500)
    body_context = smart_truncate(slice_body_for_finder(ev.description, head=1800, tail=1200), 2200)
    description_parts: List[str] = []
    for part in [
        action_surface,
        gr.supporting_snippet or "",
        body_excerpt,
        body_focus,
        body_context,
    ]:
        cleaned = clean_text(part) or ""
        if not cleaned:
            continue
        norm = normalize_key(cleaned)
        if any(norm and norm in normalize_key(existing) for existing in description_parts):
            continue
        description_parts.append(cleaned)
    description = "\n\n".join(description_parts).strip()

    return {
        "event_id": ev.event_id,
        "title": ev.title or "",
        "summary": smart_truncate(ev.summary or "", 900),
        "description": smart_truncate(description, 2200),
        "tags": ev.tags or [],
        "published_at": ev.published_at or "",
        "source": ev.source or "",
        "url": ev.url or "",
        "bd_signal": ev.bd_signal or {},
        "evidence_package": evidence_package,
        "gate": {
            "evidence_type": gr.evidence_type,
            "evidence_snippet": gr.evidence_snippet,
            "supporting_snippet": gr.supporting_snippet,
            "posting_entity": gr.posting_entity,
            "subject_entity": gr.subject_entity,
            "bd_counterparty": gr.bd_counterparty,
        },
    }


def _pattern_guard_haystack(
    title: str = "",
    summary: str = "",
    body: str = "",
    evidence_snippet: str = "",
    reason_text: str = "",
) -> str:
    return " ".join(
        [
            clean_text(title) or "",
            clean_text(summary) or "",
            clean_text(evidence_snippet) or "",
            clean_text(reason_text) or "",
            smart_truncate(clean_text(body) or "", 1800),
        ]
    )


def evaluate_pattern_guard(
    evidence_type: str,
    title: str = "",
    summary: str = "",
    body: str = "",
    evidence_snippet: str = "",
    reason_text: str = "",
) -> Dict[str, Any]:
    haystack = _pattern_guard_haystack(title, summary, body, evidence_snippet, reason_text)
    explicit_open_intake = bool(re.search(r"\b(applications? open|application thread|apply now|apply|open grant|grants? program|rfp|request for proposal|proposal category|intake)\b", haystack, flags=re.IGNORECASE))
    early_partner = bool(re.search(r"\b(early partners?|partner program|looking for partner projects|seeking collaborators)\b", haystack, flags=re.IGNORECASE))
    operator_admission = bool(re.search(r"\b(admission|operator status|professional operator|trusted operator|applying for)\b", haystack, flags=re.IGNORECASE))
    audit_funding = bool(re.search(r"\b(fund an audit|cover an audit|audit funding|appetite to add)\b", haystack, flags=re.IGNORECASE))
    explicit_external_action = bool(re.search(r"\b(builders? can|integrators? can|partners? can now|issuers? can|banks? can|fintechs? can|"
                                             r"projects? can|operators? can|validators? can|delegators? can|integrate|integration|"
                                             r"onboard|onboarding|settle|embed|connect|route|extend|deploy|deployments?|"
                                             r"collaborat(?:e|ion)|co-build(?:ing)?|distribution|listing|relaunch|reliability|"
                                             r"connect your wallet|stake now|claim rewards?|swap(?: for)?|"
                                             r"codebase|sdk|documentation|docs|openapi|swagger|rest api|demo api|live demo|"
                                             r"publicly available|open[ -]sourc(?:e|ed)|repository|tagged release|"
                                             r"token standard|privacy pool|address poisoning|scam address|screening|"
                                             r"governance data|data pipeline|delegator dashboard|light node|monitoring|operations|open to collaboration)\b", haystack, flags=re.IGNORECASE))
    builder_surface = bool(re.search(r"\b(codebase|sdk|documentation|docs|openapi|swagger|rest api|demo api|live demo|"
                                     r"publicly available|open[ -]sourc(?:e|ed)|repository|tagged release|"
                                     r"token standard|privacy pool|address poisoning|scam address|screening|"
                                     r"governance data|data pipeline|delegator dashboard|connect your wallet|"
                                     r"claim rewards?|swap(?: for)?|stake now|light node|monitoring|reliability|"
                                     r"scam(?:-| )detection)\b", haystack, flags=re.IGNORECASE))
    plausible_channel = bool(re.search(r"\b(github|forum thread|docs|documentation|repository|tagged release|live demo|demo api|"
                                      r"openapi|swagger|community|maintain(?:ed|er)|open to collaboration)\b", haystack, flags=re.IGNORECASE))

    strong = (
        (evidence_type == "grant_or_rfp_or_program_open" and explicit_open_intake)
        or (evidence_type == "partnership_or_integration" and (early_partner or operator_admission or audit_funding))
        or (evidence_type == "vendor_or_procurement_need" and audit_funding)
    )
    medium = explicit_external_action or early_partner or operator_admission
    return {
        "explicit_open_intake": explicit_open_intake,
        "early_partner": early_partner,
        "operator_admission": operator_admission,
        "audit_funding": audit_funding,
        "explicit_external_action": explicit_external_action,
        "builder_surface": builder_surface,
        "plausible_channel": plausible_channel,
        "strong": strong,
        "medium": medium,
    }


def apply_pattern_guard_to_candidate(candidate: Dict[str, Any], source_events: List["Event"]) -> Dict[str, Any]:
    evidence_type = (candidate.get("evidence_type") or "").strip()
    evidence_snippet = enforce_verbatim_snippet(candidate.get("evidence_snippet")) or ""
    body = "\n".join([ev.description or "" for ev in source_events[:2]])
    summary = "\n".join([ev.summary or "" for ev in source_events[:2]])
    title = candidate.get("title") or " | ".join([ev.title or "" for ev in source_events[:2]])
    guard = evaluate_pattern_guard(
        evidence_type=evidence_type,
        title=title,
        summary=summary,
        body=body,
        evidence_snippet=evidence_snippet,
        reason_text=candidate.get("reason") or "",
    )
    candidate["pattern_guard"] = guard
    evidence_type = (candidate.get("evidence_type") or "").strip()
    protected_medium = (
        evidence_type in {
            "support_added_with_bd_action",
            "tooling_or_standards_implementation_open",
        }
        or (evidence_type == "partnership_or_integration" and guard["builder_surface"])
    )
    if not candidate.get("is_opportunity", False) and (guard["strong"] or (protected_medium and guard["medium"])):
        candidate["is_opportunity"] = True
        sub = dict(candidate.get("sub_scores") or {})
        if guard["strong"]:
            for key in ("evidence_score", "actionability_score"):
                sub[key] = min(3, int(sub.get(key, 0)) + 1)
        candidate["sub_scores"] = sub
    if candidate.get("is_opportunity", False) and guard["medium"]:
        if protected_medium:
            sub = dict(candidate.get("sub_scores") or {})
            sub["fit_score"] = max(2, int(sub.get("fit_score", 0)))
            sub["evidence_score"] = max(2, int(sub.get("evidence_score", 0)))
            sub["actionability_score"] = max(2, int(sub.get("actionability_score", 0)))
            if guard["plausible_channel"] or guard["builder_surface"]:
                sub["feasibility_score"] = max(2, int(sub.get("feasibility_score", 0)))
            else:
                sub["feasibility_score"] = max(1, int(sub.get("feasibility_score", 0)))
            candidate["sub_scores"] = sub
    return candidate


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


class TokenTracker:
    """Track OpenAI token usage and estimated costs."""

    def __init__(self):
        self.model_usage = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_estimated_cost = 0.0
        self._lock = threading.Lock()

    def get_pricing(self, model: str):
        """Return (input_cost_per_1m, output_cost_per_1m) in USD per 1M tokens."""
        pricing = {
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
            "o3": (2.0, 8.0),
            "gpt-5.1": (1.25, 10.0),
            "gpt-5.2": (1.25, 10.0),
            "o3-mini": (1.1, 4.4),
            "gpt-5-mini": (0.25, 2.0),
        }
        return pricing.get(model)

    def _ensure_model(self, model: str):
        if model not in self.model_usage:
            self.model_usage[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0,
            }

    def add_usage(self, usage, model: str = "gpt-4o"):
        if not usage:
            return

        with self._lock:
            pricing = self.get_pricing(model)
            if not pricing:
                raise ValueError(f"No pricing found for model: {model}")

            self._ensure_model(model)

            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)

            input_cost_per_1m, output_cost_per_1m = pricing

            cost = (input_tokens / 1_000_000) * input_cost_per_1m + (output_tokens / 1_000_000) * output_cost_per_1m

            self.model_usage[model]["input_tokens"] += input_tokens
            self.model_usage[model]["output_tokens"] += output_tokens
            self.model_usage[model]["estimated_cost"] += cost

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_estimated_cost += cost

    def print_summary(self):
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
            print(f"  Estimated cost: ${data['estimated_cost']:.4f}")

        print("\n" + "=" * 72)
        print("[OpportunityMatcher] TOTAL USAGE SUMMARY")
        print("=" * 72)

        total_tokens = self.total_input_tokens + self.total_output_tokens
        print(f"  Total input tokens:   {self.total_input_tokens:,}")
        print(f"  Total output tokens:  {self.total_output_tokens:,}")
        print(f"  Grand total tokens:   {total_tokens:,}")
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

DEFAULT_FILTER_MODEL = "gpt-5.2"
FILTER_BATCH_SIZE = 10
MAX_BATCH_WORKERS = 3

DEFAULT_MODEL = "gpt-5.2"      # Finder + Batcher default
DEFAULT_GATEKEEPER_MODEL = "gpt-4o-mini"
DEFAULT_CRITIC_MODEL = "o3"
DEFAULT_ENRICHMENT_MODEL = "gpt-5.2"
DEFAULT_REFINER_MODEL = "gpt-5.2"
DEFAULT_RECOVERY_MODEL = "gpt-5.2"

DEFAULT_MAX_EVENTS_PER_BATCH = 16
DEFAULT_MAX_OUTPUT_TOKENS = 2200
DEFAULT_MIN_CONFIDENCE = 0.60
RECOVERY_MAX_CANDIDATES = 12
RECOVERY_MAX_RESTORED = 5


# Time horizon (weeks) used across ALL agent prompts; overridden by CLI flag --time-horizon-weeks
TIME_HORIZON_WEEKS = 12
TIME_HORIZON_STR = "1-12 weeks"
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

    # Narrow: explicit procurement / vendor selection signals.
    "vendor_or_procurement_need": {"proc"},

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

    # Explicit tooling / standards implementation surfaces are primarily technical BD motions.
    "tooling_or_standards_implementation_open": {"tech", "part"},
}

# Strict "gener" rule: only allow "gener" for these evidence types.
STRICT_GENER_ALLOWED_EVIDENCE: set[str] = {"support_added_with_bd_action"}

# Cardinality guardrail: cap categories per opportunity (post-validation)
MAX_CATEGORIES_PER_OPPORTUNITY = 2


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
    bd_signal: Optional[Dict[str, Any]] = None

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
            "bd_signal": self.bd_signal or {},
        }

def build_gatekeeper_events_payload(events: List[Event], compact: bool = False) -> List[Dict[str, Any]]:
    """Construct gatekeeper payload; compact mode is used for retry on parse failure."""
    if compact:
        title_max = 180
        summary_max = 450
        desc_head = 1200
        desc_tail = 500
        action_hint_max = 260
        source_max = 120
        url_max = 260
    else:
        title_max = 260
        summary_max = 900
        desc_head = 2200
        desc_tail = 900
        action_hint_max = 500
        source_max = 120
        url_max = 400

    payload: List[Dict[str, Any]] = []
    for ev in events:
        payload.append(
            {
                "event_id": ev.event_id,
                "title": _compact_gatekeeper_text(ev.title, title_max),
                "summary": _compact_gatekeeper_text(ev.summary, summary_max),
                "description": _compact_gatekeeper_text(
                    slice_body_for_gatekeeper(ev.description, head=desc_head, tail=desc_tail),
                    desc_head + desc_tail + 20,
                ),
                "action_hint_excerpt": _compact_gatekeeper_text(
                    build_gatekeeper_action_hint(ev),
                    action_hint_max,
                ),
                "bd_signal": ev.bd_signal or {},
                "source": _compact_gatekeeper_text(ev.source, source_max),
                "url": _compact_gatekeeper_text(ev.url, url_max),
            }
        )
    return payload

@dataclass
class GateResult:
    event_id: str
    hard_exclusion: bool
    hard_exclusion_reason: Optional[str]
    evidence_type: str
    evidence_snippet: Optional[str]
    supporting_snippet: Optional[str]
    posting_entity: Optional[str]
    subject_entity: Optional[str]
    bd_counterparty: Optional[str]

    def passes(self) -> bool:
        return (not self.hard_exclusion) and self.evidence_type != "none" and bool(self.evidence_snippet)


def _maybe_relax_gatekeeper_hard_exclusion(ev: Event, gr: GateResult) -> GateResult:
    if not gr.hard_exclusion:
        return gr

    haystack = _pattern_guard_haystack(ev.title, ev.summary, ev.description, "", "")
    lowered = haystack.lower()

    if re.search(r"\b(applicant-side|grant application|open grant proposal|self-nominat|delegate request|delegation|vote|snapshot|forum discussion only|governance participation only)\b", lowered):
        return gr

    trust_wallet_surface = bool(
        re.search(r"\b(trust wallet|address poisoning|address poisoning protection|scam address|screening|scam(?:-| )detection)\b", haystack, flags=re.IGNORECASE)
    )
    participation_surface = bool(
        re.search(r"\b(participation architecture|final grant report)\b", haystack, flags=re.IGNORECASE)
        and re.search(r"\b(governance data|governance api|data api|data pipeline|dashboard|api|open to collaboration|integrat(?:e|ion)|publicly available|builders? can)\b", haystack, flags=re.IGNORECASE)
    )
    atlas_surface = bool(
        re.search(r"\batlas\b", haystack, flags=re.IGNORECASE)
        and re.search(r"\b(dashboard|delegator dashboard|connect your wallet|claim rewards?|swap|stake now|api|listing)\b", haystack, flags=re.IGNORECASE)
    )
    streetcredit_surface = bool(
        re.search(r"\bstreetcredit\b", haystack, flags=re.IGNORECASE)
        and re.search(r"\bcelo\b", haystack, flags=re.IGNORECASE)
        and re.search(r"\bintegrat(?:e|ion)|partner|collaborat|feedback\b", haystack, flags=re.IGNORECASE)
    )

    if not any((trust_wallet_surface, participation_surface, atlas_surface, streetcredit_surface)):
        return gr

    evidence_snippet = _extract_action_surface([ev.summary, ev.description, ev.title], "", 240)
    if not evidence_snippet:
        return gr

    if streetcredit_surface:
        evidence_type = "partner_program_intake_or_ecosystem_collaboration_intake"
    else:
        evidence_type = "tooling_or_standards_implementation_open"

    return GateResult(
        event_id=gr.event_id,
        hard_exclusion=False,
        hard_exclusion_reason="softened_builder_surface_override",
        evidence_type=evidence_type,
        evidence_snippet=evidence_snippet,
        supporting_snippet=gr.supporting_snippet,
        posting_entity=gr.posting_entity,
        subject_entity=gr.subject_entity,
        bd_counterparty=gr.bd_counterparty,
    )


@dataclass
class Opportunity:
    opportunity_id: str
    title: str
    summary: str
    opportunity_details: Optional[str]
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

        query = f"""
            INSERT INTO {self.schema}.opportunities_filter (
                opportunity_id,
                title,
                summary,
                reason,
                who_to_contact,
                suggested_outreach_angle,
                categories,
                filter_chain,
                filter_sector,
                filter_seeking,
                time_found,
                confidence,
                tags,
                target_company,
                sources,
                event_ids,
                event_titles,
                event_url,
                bd_weeks,
                opportunity_details
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            )
            ON CONFLICT (opportunity_id) DO UPDATE SET
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                reason = EXCLUDED.reason,
                who_to_contact = EXCLUDED.who_to_contact,
                suggested_outreach_angle = EXCLUDED.suggested_outreach_angle,
                categories = EXCLUDED.categories,
                filter_chain = EXCLUDED.filter_chain,
                filter_sector = EXCLUDED.filter_sector,
                filter_seeking = EXCLUDED.filter_seeking,
                confidence = EXCLUDED.confidence,
                tags = EXCLUDED.tags,
                target_company = EXCLUDED.target_company,
                sources = EXCLUDED.sources,
                event_ids = EXCLUDED.event_ids,
                event_titles = EXCLUDED.event_titles,
                event_url = EXCLUDED.event_url,
                bd_weeks = EXCLUDED.bd_weeks,
                opportunity_details = EXCLUDED.opportunity_details
        """

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self.check_connection()
                with self.conn.cursor() as cur:
                    for o in opportunities:
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
                                o.tags,
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
            except Exception as e:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                print(f"[OpportunityMatcher] ERROR: save_opportunities failed: {e}")
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
                                    "tooling_or_standards_implementation_open",
                                    "none"
                                ],
                            },
                            "evidence_snippet": {"type": ["string", "null"]},
                            "supporting_snippet": {"type": ["string", "null"]},
                            "posting_entity": {"type": ["string", "null"]},
                            "subject_entity": {"type": ["string", "null"]},
                            "bd_counterparty": {"type": ["string", "null"]},
                        },
                        "required": [
                            "event_id",
                            "hard_exclusion",
                            "hard_exclusion_reason",
                            "evidence_type",
                            "evidence_snippet",
                            "supporting_snippet",
                            "posting_entity",
                            "subject_entity",
                            "bd_counterparty",
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
                            "summary": {"type": "string"},
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
                                    "support_added_with_bd_action",
                                    "tooling_or_standards_implementation_open"
                                ],
                            },
                            "evidence_snippet": {"type": "string"},
                            "recommended_action": {"type": ["string", "null"]},
                            "suggested_outreach_angle": {"type": "string"},
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
                            "summary",
                            "reason",
                            "evidence_type",
                            "evidence_snippet",
                            "recommended_action",
                            "suggested_outreach_angle",
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
                        "properties": {
                            "index": {"type": "integer"},
                            "opportunity_title": {"type": "string"},
                            "score": {"type": "number"},
                            "recommendation": {"type": "string", "enum": ["keep", "drop", "watchlist"]},
                            "critic_notes": {"type": "string"},
                            "reframed_opportunity": {
                                "type": "array",
                                "items": opp_items_schema
                            }
                        },
                        "required": [
                            "index",
                            "score",
                            "recommendation",
                            "critic_notes",
                            "reframed_opportunity"
                        ],
                        "additionalProperties": False
                    }
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
        "name": "near_miss_recovery_v1",
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
            tracker.add_usage(resp.usage, model)

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

    # Conservative scaling + lower cap to reduce truncation risk.
    tokens_per_event = 120
    base_tokens = 900
    safety_buffer = 500

    max_tokens = base_tokens + (len(events) * tokens_per_event) + safety_buffer
    max_tokens = min(max_tokens, 5200)

    events_payload = build_gatekeeper_events_payload(events, compact=False)

    system_prompt = """You are a compliance gatekeeper for business-opportunity detection.

Your job is binary: PASS or FAIL each event. You do not score quality. You only determine whether
a qualifying BD action surface exists and extract a verbatim snippet proving it.

If a structured bd_signal field is provided in the event payload, use it as a weak prior only: it may help you find the likely action surface faster, but it NEVER overrides the event text.
If bd_signal.surface_type is present (for example live_api_dashboard_surface, live_security_rail,
named_integration_exploration, or live_adoption_surface), use it only as a navigation hint for what kind of
surface to look for in the event text. It is advisory metadata, not evidence.

bd_signal is advisory metadata, not evidence. Do not pass an event solely because bd_signal.has_action_surface=true. If bd_signal conflicts with the event text, trust the event text and fail closed.

If action_hint_excerpt is provided, use it only as a navigation aid to locate the strongest grounded action sentence in the summary/description.
It may help you avoid anchoring on a weak headline or generic intro, but it NEVER overrides the original event text.

=== HARD EXCLUSIONS (auto-fail                                                                                                                                                                                        set hard_exclusion=true) ===

Content type exclusions:
- Price moves, token price commentary, market recaps
- Opinion, editorial, macro narrative, sentiment analysis
- Pure funding/investment/valuation news with no external action surface
  ("raises $X", "Series A/B/C", "closes fund", "M&A rumor", "valuation reaches $X")
- Institutional portfolio or ETF rebalancing commentary with no external action surface
- GitHub CI builds, release candidates, and version bumps with no BD-relevant announcement
  (titles matching patterns like: "_ci", "-rc.", "v0.", "@v", "viem@", "wagmi@", "log/v")
- NEVER use the price-move / market-recap exclusion for wallet-security, address-poisoning, token-standard,
  governance-data, dashboard, or similar tooling/product surfaces.
- Do NOT use content-type exclusion for product/security/data/tooling/dashboard/API/SDK/webhook/monitoring/
  validator/light-node announcements when the text explicitly shows that outside teams can implement, integrate,
  adopt, deploy, list, monitor, extend, or onboard now.
- Wallet-security launches that add address-poisoning protection, scam-address checks, or similar security screening
  across live chains should normally be evaluated as tooling_or_standards_implementation_open, not as market/news content.
- Final grant reports, case studies, and open-source/tooling releases are NOT auto-fail if they clearly describe
  a live external implementation, collaboration, or adoption surface.
- Final grant reports are also NOT auto-fail when they expose a live API, dashboard, governance-data surface, or
  other implementation surface that outside teams can use, extend, or integrate now.
- Exclude final reports only when they are retrospective summaries with no live API, dashboard, governance-data
  surface, docs, or implementation surface for outside teams to use now.

Secondary inference exclusions (IMPORTANT):
- Events where the BD opportunity would only exist for a THIRD PARTY, not the named company.
  Example: "Company A partners with Company B"                                                                                                                                                                                        this is a signal for A and B, not for you.
  Only pass if the event exposes an OPEN action surface that any external BD team could act on.
- Applicant-side or candidate-side actions are NOT BD leads by default: a company following up on its own submitted request,
  applying for operator/admission status, self-nominating for a committee, or submitting its own proposal for funding.
  These are the posting entity seeking approval for itself, not a host opening an intake for outside partners.
- Third-party "Open Grant Proposal" or forum application threads do NOT pass just because a host platform exists.
  The host must explicitly invite external vendors/partners beyond the proposer, or run a true external intake/RFP.
- PR announcements with no external intake (no application, no RFP, no partner program, no contact channel).
  A press release saying "we expanded" is NOT a pass unless it includes an explicit external motion.
- Single-repo maintenance chores are NOT BD opportunities by default: repo URL migrations, internal PR wiring,
  maintenance rebuilds for existing deployments, or one-off code contribution tasks with no broader external
  partner/cohort/funded scope.
- Do NOT use secondary-inference exclusion when the named company itself creates the external action surface.
  If the target explicitly opens an implementation, adoption, or collaboration path for outside teams now,
  keep evaluating even if media coverage, a report, a case study, or another company is mentioned for context.

User-research and community-noise exclusions:
- User research recruitment is NOT a BD opportunity by default: user testing, survey calls, interview participant calls, beta tester recruitment, or generic community feedback calls with no concrete build/integration path.
  Do NOT use this exclusion when the same text also exposes a live API, dashboard, data/security/intelligence rail, shipped tooling/final-report surface, or a named integration path that outside teams can act on now.
- Governance/community participation-only actions are NOT BD opportunities: comment, vote, forum discussion, AMA attendance, or proposal feedback with no commercial intake.
- Committee elections, governance self-nominations, and status-follow-up threads are NOT BD opportunities.

KEEP OVERRIDE (important):
- KEEP these events if they include an explicit external commercial intake surface, even if they contain the above noise terms.
- Valid KEEP signals include at least one of:
  1) explicit vendor/RFP/procurement language, or
  2) paid deliverable contract scope (not participant stipend), or
  3) explicit partner/integration program opening, or
  4) explicit external implementation/adoption surface in a product/security/data/standard/tooling context, or
  5) a scoped paid submission/prize/funded-selection path for a concrete external deliverable.

=== MINIMUM EVIDENCE TO PASS (must have >=1) ===

At least one of the following must be explicitly present in the event text:

- partnership_or_integration
  Explicit named partnership OR integration announced with external parties.
  Must include named parties AND an explicit integration or partnership program.
  A press release about two companies working together ONLY passes if there is an explicit
  action surface for a third party (partner intake, API access, integration program).

- grant_or_rfp_or_program_open
  Explicit: applications open / RFP/RFQ/RFI issued / call for proposals / grant round open.
  The program must be open to EXTERNAL participants.
  NOT: venture raises, fund closes, "launches fund", "raises $X from investors".

- vendor_or_procurement_need
  Explicit tender / procurement process / vendor search / RFP for vendors.
  Must reflect a buying intent, not an investment.

- token_launch_or_tge
  Explicit token launch / TGE / token generation event with a stated date,
  or an open signup / waitlist for participants.

- regulatory_or_compliance_update
  Explicit license, approval, registration, or compliance change.
  ONLY passes if there is an external action surface:
  application process, partner onboarding path, vendor intake, integration path, or named contact.
  A regulator approving something internally is NOT a pass.

- institutional_allocation_or_treasury
  Explicit corporate, treasury, or institutional allocation decision.
  ONLY passes if there is an external action surface:
  vendor selection process, platform onboarding, integration, partner channel, or RFP.
  A company disclosing it holds crypto is NOT a pass.

- support_added_with_bd_action
  New chain support, feature, or product added AND an explicit BD action surface.
  The BD action surface must be stated (e.g. "integrators are encouraged to apply",
  "partners can now access via X"). The support alone is not sufficient.
  Explicit implementation/adoption paths also count when grounded, such as:
  builders/operators/institutions/projects can deploy, integrate, monitor, extend, list, settle, or onboard now
  through a newly available framework, API, SDK, dashboard, data/security surface, or similar external product rail.
  Generic launch PR still fails.

- tooling_or_standards_implementation_open
  Explicit tooling, standards, governance-data, dashboard, monitoring, validator/delegator, or GitHub-style
  implementation surface where external builders, operators, integrators, or product teams can concretely build,
  adopt, extend, list, plug in, or support it now.
  This requires a real build/adoption path in the text, not generic "feedback welcome" wording.
  Public codebase + SDK + documentation availability, live demo APIs/docs, shipped token standards/frameworks,
  and final grant reports/case studies describing an already-live external implementation surface can qualify here.
  You do NOT need a literal "apply/contact us" phrase if the event text clearly shows what outside teams can do now.
  Use this when the surface is more concrete than a generic launch but does not fit a classic grant/RFP/procurement intake.

- partner_program_intake_or_ecosystem_collaboration_intake
  Explicit statements that a project is looking for partner projects, collaborators, or program members
  for near-term intake (for example: "looking for 4-5 partner projects", "seeking ecosystem collaborators").
  This is a valid medium-quality BD surface.
  Keep lower priority than hard integration/RFP/procurement signals, but do NOT auto-fail.
  A founder/operator thread exploring whether a named integration makes sense can also qualify here when it
  explicitly invites relevant builders, ecosystem teams, or operators to engage on that integration path now,
  including when framed as feedback on a specific named integration rather than a formal intake form.

=== EVIDENCE SNIPPET RULES ===

- evidence_snippet MUST be verbatim text copied from the event summary or description.
- Do NOT paraphrase or summarise. Copy the exact words.
- Length: 25-40 words. Enough context for the Finder to understand the action surface.
  Do not truncate so aggressively that the action surface becomes ambiguous.
- Search the full summary and description before failing. Prefer the strongest actionable sentence available,
  not a generic headline, generic summary line, or weak introductory sentence.
- If title/summary are generic but the body contains a stronger intake / implementation / collaboration / adoption sentence,
  use the body sentence as evidence_snippet and use supporting_snippet for surrounding context if helpful.
- If the event contains both a weak participation sentence ("feedback welcome", "have a say", "community discussion")
  and a stronger operational/build sentence ("connect your wallet", "claim rewards", "swap", "builders can create",
  "codebase/SDK/docs publicly available", "live demo API", "open to collaboration/integrate"), choose the stronger
  operational/build sentence as evidence_snippet.
- For dashboard/tooling posts, prefer concrete usage/integration lines like "connect your wallet", "claim rewards",
  "swap ... and then stake", or "integrate the API" over generic "feedback welcome" lines.
- If you cannot find a verbatim excerpt that proves the evidence type: set evidence_type='none' and evidence_snippet=null.
- supporting_snippet is optional. When available, return a second verbatim snippet from the same event
  that adds actor, scope, timing, or intake context without repeating the evidence_snippet. Use null if not needed.
- supporting_snippet should usually be 15-60 words and can be one or two sentences if the copied text is contiguous.

=== ACTOR / COUNTERPARTY FIELDS ===

- posting_entity: who authored, posted, or published the event/post.
- subject_entity: who the event is mainly about or whose business motion is being described.
- bd_counterparty: who an external BD team would actually approach if the event is actionable.
- These may be the same entity. If unclear, use null rather than guessing.
- Distinguish proposer vs host carefully. Example:
  if Blockful posts a proposal asking Safe DAO for funding, posting_entity=Blockful.
  If the actionable commercial counterparty is Safe DAO, set bd_counterparty=Safe DAO.

- Do not repeat or duplicate phrases inside evidence_snippet. Return one clean contiguous excerpt only.
- Consistency rule:
  if evidence_type != 'none', then hard_exclusion MUST be false and evidence_snippet MUST be non-null.
  If hard_exclusion=true, then evidence_type MUST be 'none' and evidence_snippet MUST be null.

=== OUTPUT ===
Output ONLY JSON per schema. No explanations.
"""

    def _gatekeeper_request(payload: List[Dict[str, str]], token_budget: int) -> Tuple[str, bool]:
        user_prompt = f"""EVENTS:
{json.dumps(payload, default=_json_safe, separators=(",", ":"))}

Return gatekeeper results for every event_id."""

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": openai_gatekeeper_schema()},
            temperature=0,
            max_completion_tokens=token_budget,
        )

        if tracker:
            tracker.add_usage(resp.usage, model)

        near_cap = False
        if resp.usage and hasattr(resp.usage, "completion_tokens"):
            near_cap = bool(resp.usage.completion_tokens >= token_budget * 0.95)
            if near_cap:
                print(f"  WARNING: Gatekeeper response likely truncated ({resp.usage.completion_tokens}/{token_budget} tokens)")

        return resp.choices[0].message.content, near_cap

    def _gatekeeper_per_event_fallback(reason: str) -> Dict[str, GateResult]:
        print(f"  WARNING: Gatekeeper falling back to per-event compact calls for this batch ({reason})")
        fallback_out: Dict[str, GateResult] = {}
        for ev in events:
            try:
                single_payload = build_gatekeeper_events_payload([ev], compact=True)
                raw_single, _ = _gatekeeper_request(single_payload, 1200)
                parsed_single = json.loads(raw_single)
                row = (parsed_single.get("results") or [{}])[0]
                fallback_out[ev.event_id] = GateResult(
                    event_id=row.get("event_id") or ev.event_id,
                    hard_exclusion=bool(row.get("hard_exclusion", True)),
                    hard_exclusion_reason=row.get("hard_exclusion_reason") or "gatekeeper_fallback",
                    evidence_type=row.get("evidence_type") or "none",
                    evidence_snippet=enforce_verbatim_snippet(row.get("evidence_snippet")),
                    supporting_snippet=enforce_verbatim_snippet(row.get("supporting_snippet"), max_words=60),
                    posting_entity=clean_text(row.get("posting_entity")) or None,
                    subject_entity=clean_text(row.get("subject_entity")) or None,
                    bd_counterparty=clean_text(row.get("bd_counterparty")) or None,
                )
            except Exception as e3:
                print(f"  WARNING: Gatekeeper single-event fallback failed for {ev.event_id}: {e3}")
                fallback_out[ev.event_id] = GateResult(
                    event_id=ev.event_id,
                    hard_exclusion=True,
                    hard_exclusion_reason="gatekeeper_parse_failure",
                    evidence_type="none",
                    evidence_snippet=None,
                    supporting_snippet=None,
                    posting_entity=None,
                    subject_entity=None,
                    bd_counterparty=None,
                )
        return fallback_out

    raw, first_near_cap = _gatekeeper_request(events_payload, max_tokens)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  WARNING: Gatekeeper JSON parse failed (attempt 1): {e}")
        retry_payload = build_gatekeeper_events_payload(events, compact=True)
        retry_tokens = min(max_tokens, 3600)
        raw_retry, retry_near_cap = _gatekeeper_request(retry_payload, retry_tokens)
        try:
            parsed = json.loads(raw_retry)
            if retry_near_cap:
                return _gatekeeper_per_event_fallback("retry_response_near_token_cap")
        except json.JSONDecodeError as e2:
            print(f"  WARNING: Gatekeeper JSON parse failed (attempt 2): {e2}")
            return _gatekeeper_per_event_fallback("retry_json_parse_failed")

    out: Dict[str, GateResult] = {}
    for r in parsed.get("results", []):
        gr = GateResult(
            event_id=r["event_id"],
            hard_exclusion=bool(r["hard_exclusion"]),
            hard_exclusion_reason=r.get("hard_exclusion_reason"),
            evidence_type=r.get("evidence_type") or "none",
            evidence_snippet=enforce_verbatim_snippet(r.get("evidence_snippet")),
            supporting_snippet=enforce_verbatim_snippet(r.get("supporting_snippet"), max_words=60),
            posting_entity=clean_text(r.get("posting_entity")) or None,
            subject_entity=clean_text(r.get("subject_entity")) or None,
            bd_counterparty=clean_text(r.get("bd_counterparty")) or None,
        )
        ev = next((e for e in events if e.event_id == gr.event_id), None)
        if ev is not None:
            gr = _maybe_relax_gatekeeper_hard_exclusion(ev, gr)
        out[gr.event_id] = gr
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

    events_payload = []
    for ev, gr in gated_events:
        events_payload.append(build_finder_event_payload(ev, gr))

    system_prompt = f"""You are a senior BD analyst at a Web3 business development intelligence company.
Your output is sold to Web3 BD teams as actionable lead intelligence. Quality directly affects revenue.

You receive events that have already passed the Gatekeeper compliance gate.
Your job is to identify genuine BD leads                                                                                                                                                                                        not to summarise what happened.

=== THE CORE DISTINCTION (read carefully) ===

A SIGNAL is something that happened: "Protocol X launched a new product."
A LEAD is something a BD team can act on: "Protocol X opened applications for integration partners."

You produce LEADS only. For each candidate, decide:
- is_opportunity=true when it is a real BD lead that should go into the normal opportunity path
- is_opportunity=false when it is signal/noise/non-lead that should be rejected from the normal path

The test: "Could a BD team send an outreach email TODAY based on this, and plausibly receive a reply
from a partnerships inbox, ecosystem/devrel owner, maintainer, or clearly relevant implementation owner within {TIME_HORIZON_STR}?"
If the honest answer is clearly yes or plausibly yes with grounded evidence                                                                                                                                                                                        set is_opportunity=true.
If the honest answer is clearly no                                                                                                                                                                                        set is_opportunity=false.

=== NON-NEGOTIABLE RULES ===

1) Only gated events. You only receive events that passed the Gatekeeper. Do not re-evaluate gate compliance.

1A) If a bd_signal object is provided with the event, use it as supporting context for actionability and outreach framing only. Treat bd_signal as a summary hint, not as proof. gate.evidence_snippet remains the only grounding source for evidence claims and recommended_action.
    If bd_signal.surface_type is present, you may use it as a weak classification hint when deciding whether the
    grounded evidence looks like a live API/dashboard surface, live security rail, named integration exploration,
    or live adoption surface. It must never override gate.evidence_snippet.
1B) If an evidence_package object is provided, use it actively.
   - clean_title and clean_summary help you avoid noisy wording
   - strong_body_excerpt highlights the strongest relevant body region
   - explicit_action_surface highlights the most actionable text slice
   - source_context helps you interpret source type and context
   Treat evidence_package as decision support, not as permission to invent facts.
1C) If gate.supporting_snippet is provided, use it only as extra grounded context for actor, scope, timing,
   or intake interpretation. It may clarify a real action surface, but it must not turn a generic launch into a lead.
1D) If gate.posting_entity / gate.subject_entity / gate.bd_counterparty are provided, use them to avoid proposer-vs-host confusion.
   posting_entity is who published the post. subject_entity is who the event is about. bd_counterparty is who a BD team
   would actually approach. Prefer bd_counterparty when choosing target_company if the evidence supports that reading.

2) evidence_snippet must be verbatim. Use the provided gate.evidence_snippet exactly. Do not paraphrase or extend it.

3) evidence_type must match gate.evidence_type. Do not invent new evidence types.

4) recommended_action must be grounded in evidence_snippet.
   The exact words in evidence_snippet must justify the action. If they do not, do not classify as is_opportunity=true.

5) No secondary inference.
   If the opportunity only exists because of a deal between OTHER parties (Company A partnered with Company B,
   therefore pitch Company B), classify as is_opportunity=false unless the event explicitly opens an external intake
   that any third party can act on.

5C) Narrow allowed exception: product-enabled integration/onboarding surfaces.
   Classify as is_opportunity=true when the target itself explicitly creates a new external adoption or integration
   path for outside teams now, and the text clearly shows who can act and how.
   This includes explicit SDK/API/webhook/dashboard/data/security/tooling surfaces, public codebase/docs/demo releases,
   token-standard/framework rollouts, early-partner cohorts, and final grant reports/case studies with a live
   external implementation path.
   The action surface must be visible in gate.evidence_snippet itself, not only in bd_signal, evidence_package,
   supporting_snippet, or optimistic outreach framing.
   Do NOT use this exception for generic launch PR, roadmap mentions, completed deployments, or vague ecosystem language.

5A) Do NOT treat applicant-side status threads as leads.
   If the posting entity is following up on its own submitted request, applying for admission/operator status,
   self-nominating for a committee, or asking a host to approve its own application, classify as is_opportunity=false
   unless the host simultaneously opens a separate external vendor/partner intake for third parties.

5B) Do NOT treat third-party grant applications as host intake.
   If a proposer submits its own grant/proposal/application to a host program, that is the proposer seeking funding
   for itself. It is NOT a BD lead for external readers unless the host explicitly invites outside vendors, delivery
   partners, or collaborators beyond the proposer in the grounded evidence.

6) No hard-exclusion content. If you suspect an event is price/market/opinion content, classify as is_opportunity=false.

7) Cross-run deduplication.
   If two events describe the same action surface at the same target company, produce ONE opportunity only.
   Do not produce separate opportunities for the same underlying BD motion because titles differ slightly
   or because they come from different sources.

7A) bd_signal may support prioritization, but it never overrules grounding.
   bd_signal.has_action_surface=true may help confirm that the event is likely a lead candidate, but you must still avoid is_opportunity=true if the provided gate.evidence_snippet does not justify a concrete BD action within the time horizon.

8) User-research and community-noise guard.
   Treat user-testing, surveys, interview recruitment, beta-testing, and generic community feedback calls as is_opportunity=false by default unless the text also exposes a concrete build, integration, partner execution, or live API/dashboard/data/security/tooling path that outside teams can act on now.
   Also treat single-repo maintenance chores, repo URL migrations, and one-off implementation chores as is_opportunity=false by default unless the text clearly opens a broader external partner/cohort/funded rollout path.

9) KEEP override for probable good opportunities.
   If the event explicitly contains commercial intake, you MUST keep it as is_opportunity=true even if it includes user-research/governance wording.
   Valid override signals: vendor/RFP/procurement language, paid deliverable contract scope (not participant stipend), explicit partner/integration program opening, or a scoped paid submission/prize/funded-selection path.

10) Governance participation-only guard.
   If the only action is comment/vote/discussion/forum participation with no commercial intake, classify as is_opportunity=false.

11) Partner/collaboration intake is valid medium quality when the text explicitly says it is looking for partner
    projects, collaborators, program members, or external teams to co-design, co-fund, co-run, deploy, or support
    something concrete in the near term.
    This can include a founder/operator thread exploring a named integration when the post explicitly asks relevant
    ecosystem teams or builders to engage on that integration path now. Do not downgrade it to generic "feedback"
    merely because the author asks whether the ecosystem finds the integration valuable; generic product/usability
    feedback with no named integration path is the case that should still be filtered.
    If the post names a specific chain plus a concrete use case, do not treat "would love feedback" wording by itself
    as enough reason to collapse the opportunity into generic community discussion.
12) Protected medium-confidence band.
   Do NOT mark a gate-passed implementation/adoption surface as weak noise only because contact is indirect.
   If the grounded text clearly shows a near-term external build, integration, rollout, or operational support path,
   these cases should usually land in a medium band rather than below-threshold weak band.
   This includes final reports/case studies with a live external surface and product/security/dashboard/data posts
   that pair weak feedback wording with a concrete actionable integration or support path.
   Still reject applicant-side asks, governance-only participation, generic launches, and closed bilateral deals.
13) Scoped paid submission programs can qualify.
   If the text explicitly offers payment, prize money, or funded selection for a concrete external deliverable and
   provides a submission path now, treat it as a medium opportunity rather than auto-discarding it as mere participation.

=== REQUIRED OUTPUT FIELDS ===

For each output item, you MUST populate:
- is_opportunity

- title: specific and descriptive. Include the target company name and the action type.
  Good: "Apply to Aave DAO's open RFP for risk management tooling"
  Bad: "Opportunity with Aave"

- target_company: the company or DAO the BD team would approach. Not the company that published the news.

- summary: 2-3 sentences. What is the opportunity, what is the evidence surface, and what would the outreach motion be.

- suggested_outreach_angle: this is the most important field for BD teams.
  Write 2-3 sentences describing SPECIFICALLY how to approach this target.
  Include: who to contact (role or name if known), what to lead with, and what problem or need you are addressing.
  Do NOT write "reach out to their partnerships team." Write the actual angle.
  Example: "Contact the Aave governance forum thread author directly. Lead with your existing DeFi protocol
  risk audit experience. Position your tooling as a turnkey solution for the specific crvUSD oracle
  risk scenario described in the proposal."

- recommended_action: one sentence. The single most important next step. Must be grounded in evidence_snippet.

- who_to_contact: role, name, or channel. If explicitly named in the event, use that. If not, infer the most
  likely contact point (e.g. "ecosystem team", "governance forum", "partnerships@...").

- evidence_snippet: verbatim from gate. Do not modify.

- Do not duplicate phrases in any related fields (summary/reason/recommended_action). Keep wording concise and non-repetitive.

- source_event_ids: exact event_id values from the input.

- sub_scores (0-3 each):
  fit_score: How well does this opportunity fit a generic Web3 BD team's capabilities?
    3 = any capable Web3 BD team could credibly pursue this
    2 = requires some specialisation
    1 = highly niche or technical requirement
    0 = unclear fit

  evidence_score: How explicit and unambiguous is the evidence?
    3 = explicit named RFP / open program / integration call-to-action with named parties
    2 = strong implied intake (e.g. forum post asking for vendor recommendations)
    1 = weak signal (e.g. company announced expansion, intake implied but not stated)
    0 = no meaningful evidence

  actionability_score: Can a BD team take a concrete next step within {TIME_HORIZON_STR}?
    3 = clear entry point exists now (apply here, post here, contact this person)
    2 = entry point likely exists but requires one step to find
    1 = timing unclear or action requires significant pre-work
    0 = not actionable within the time horizon

  feasibility_score: Is it realistic that a BD team could get a reply?
    3 = open program, anyone can apply / respond
    2 = warm intro or forum post possible
    1 = cold outreach to a large org with no named contact
    0 = not reachable within the time horizon

  Protected-medium note:
    For explicit implementation/adoption surfaces that already passed Gatekeeper (SDK/API/webhook/dashboard/data/security/tooling,
    partner co-design/co-fund/co-run, rollout/adoption support), do not underrate evidence/actionability just because contact is indirect.
    These should generally score at least 2 on evidence and actionability when the path is explicit in the grounded text.

=== QUALITY STANDARD ===

Target quality band: allow STRONG and MEDIUM opportunities; reject weak/noise opportunities.
- STRONG: explicit intake + specific action path + plausible reply path.
- MEDIUM: explicit intake + actionable next step, even if contact path is weak or not named.

- Confidence calibration: for product-enabled institutional onboarding surfaces with explicit action surface but indirect contact path,
  target medium confidence (roughly 0.60-0.80), not top confidence.

Zero opportunities is acceptable and preferable to weak opportunities.
Do not manufacture leads to fill a batch. If the events in a batch are signals not leads, return zero.
One strong, specific, actionable lead is worth more than five generic ones.
Retention classes A/B/C are valid medium-priority opportunities.
Prefer top-tier explicit partner/integration/vendor signals first, but keep 1-2 clearly grounded medium-priority retention-class opportunities per run.

Classification guidance:
- is_opportunity=true: explicit intake / proposal / application / admissions / procurement / actionable integration surface with a plausible next step now, or a grounded medium-strength BD motion
- tooling_or_standards_implementation_open is appropriate for explicit build/adoption paths around tooling, standards,
  governance-data APIs, dashboards, validator/delegator infrastructure, monitoring, or wallet-security rails.
- It is also appropriate for builder-facing codebase/SDK/docs/demo surfaces or token-standard/framework rollouts when
  the text clearly states that outside teams can build, integrate, or collaborate now.
- is_opportunity=false: no real external BD motion, or only signal/discussion/noise
"""

    user_prompt = f"""GATED EVENTS (each includes gate evidence_type + evidence_snippet):
{json.dumps(events_payload, default=_json_safe, separators=(",", ":"))}

AVAILABLE CATEGORIES (use only category_text_id values):
{json.dumps(category_catalog, default=_json_safe, separators=(",", ":"))}

TASK:
For each distinct business opportunity you can justify, output an object with:
- is_opportunity
- reason (1-3 sentences)
- evidence_snippet (verbatim)
- recommended_action (only if TRUE)
- target_company, who_to_contact (if you can infer), categories/tags
- source_event_ids (must be exact event_id values above)
- sub_scores (0-3 each)

Quality bar:
- It is acceptable to return zero opportunities.
- Prefer precision over volume.
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
        tracker.add_usage(resp.usage, model)

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
        "posting_entity",
        "subject_entity",
        "bd_counterparty",
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

    system_prompt = f"""You are a compliance critic for Web3 BD opportunities.

Your role is enforcement, not generation. You do not improve opportunities                                                                                                                                                                                        you accept, reframe, or discard them.
Be strict. A discarded weak opportunity is better than a published false lead.

IMPORTANT: For each item in opportunity_feedback, include the integer field 'index' corresponding to
the position of the input opportunity in the provided list. This field is required.

If source_bd_signals are provided, treat them as diagnostic metadata only. They may help detect internal inconsistency, but they never override evidence_snippet, which remains the primary grounding source.
If supporting_snippet is provided, treat it as secondary grounded context for actor/scope/timing only. It may clarify a real action surface, but it must not rescue a generic launch that lacks an explicit external path.
If posting_entity / subject_entity / bd_counterparty are provided, use them only to resolve who the real commercial counterparty is.
They help prevent proposer-vs-host confusion, but they must not justify claims unsupported by evidence_snippet.

QUALITY CALIBRATION (important):
- Allow STRONG and MEDIUM opportunities to pass; discard weak/noise opportunities.

- For product-enabled institutional onboarding surfaces, prefer medium confidence (roughly 0.60-0.80) unless explicit intake/contact is stated.
- Do not require a named person if the intake surface is explicit and reachable via a clear channel.
- Partner/collaboration intake with explicit near-term intake is valid medium quality.
- For gate-passed implementation/adoption surfaces, do not discard solely because contact is indirect.
  If evidence and actionability are explicit, preserve them as medium-quality unless another discard rule clearly applies.
- A relevant reply path can be a partnerships/BD inbox, ecosystem or devrel owner, maintainer, or clearly relevant implementation owner.
- A scoped paid submission, prize, or funded-selection program can qualify when the deliverable, payment, and submission path are explicit.
- Do not discard a final report, dashboard/API/tooling post, or wallet-security/data-intelligence surface only because the text also asks for feedback; if the actionable implementation/support path is explicit, preserve it as medium quality.
- Do not keep single-repo maintenance chores, repo URL migrations, or one-off implementation tasks unless the evidence shows a broader external partner program, funded scope, or multi-team rollout path.

=== DISCARD CONDITIONS (any one is sufficient) ===

FALSE-NEGATIVE RETENTION OVERRIDE:
Before DISCARD, check whether the source creates a NEW external action surface for implementation, integration,
partner intake, rollout/adoption support, or operator/tooling usage. If yes, prefer REFRAME over DISCARD unless
the action is mostly inferred, commercially irrelevant, or not grounded in exact source wording.
This can include a founder/operator thread exploring a named integration when the post explicitly invites relevant
ecosystem teams or builders to engage on that integration path now. Do not flatten this into generic community
feedback when the named integration target and near-term use case are explicit.

1) HARD EXCLUSION CONTENT
   Price moves, market recaps, opinion, macro narrative, sentiment analysis.
   If the core of the opportunity is about price, market conditions, or opinion                                                                                                                                                                                                    DISCARD.

2) EVIDENCE NOT GENUINELY PRESENT
   is_opportunity=true but the stated evidence_type is not actually supported by the evidence_snippet.
   If you cannot confirm the evidence type from the snippet alone                                                                                                                                                                                                    DISCARD.

3) SECONDARY INFERENCE
   Test: "Does this opportunity require the target company to have done something to create the action surface,
   or does it exist because of a deal/event between OTHER parties?"
   If the answer is "other parties" and there is no explicit external intake                                                                                                                                                                                                    DISCARD.
   If the answer is "the target itself created a new external integration/onboarding path"                                                                                                                                                                                                    keep evaluating;
   this can be valid medium-quality BD surface when the path is explicit and near-term.

   If the target itself created a new external integration, onboarding, or implementation path, keep evaluating.

4) SIGNALS NOT LEADS
   A signal is something that happened. A lead is something a BD team can act on TODAY.
   If the recommended_action or suggested_outreach_angle uses speculative language
   ("may need", "might be open to", "could explore", "is likely seeking", "exploring whether")                                                                                                                                                                                                    DISCARD
   unless the evidence_snippet contains an explicit statement that overrides the speculation.

5) TIME HORIZON VIOLATION
   The opportunity must be actionable within {TIME_HORIZON_STR}.
   If the best realistic next step is likely beyond {TIME_HORIZON_WEEKS} weeks, or timing is speculative                                                                                                                                                                                                    DISCARD.
   This includes:
   - Pitching large institutions or regulators with no explicit open program, RFP, or intake.
   - "Strategic" or "thought leadership" positioning with no external action surface.
   - Opportunities where the action surface is real but the timing is stated as future ("launching next quarter").

6) GROUNDING VIOLATION (NON-NEGOTIABLE)
   The recommended_action MUST be justified by exact words in evidence_snippet.
   If you cannot identify the specific phrase in evidence_snippet that justifies the outreach                                                                                                                                                                                                    DISCARD.
   This rule overrides phrasing quality, sub-scores, and novelty.

7) GENERIC OUTREACH ANGLE
   If suggested_outreach_angle contains only generic instructions ("reach out to their partnerships team",
   "engage their BD team", "contact them about potential collaboration") with no specific angle, person,
   channel, or positioning                                                                                                                                                                                        flag as REFRAME if a better angle can be derived from evidence_snippet,
   otherwise DISCARD.

8) GOVERNANCE PARTICIPATION                                                                                                                                                                                        NOT PROCUREMENT
   If the only action surface is community participation with no commercial transaction attached:
   leaving a comment, voting on a proposal, submitting ecosystem feedback, or attending an event                                                                                                                                                                                                    DISCARD.
   Do NOT use this rule when the same grounded text exposes a concrete implementation, integration, dashboard/API,
   security/intelligence, or live-tooling path for outside teams.

   This applies even if the source text uses formal language ("applications open", "we invite submissions",
   "call for contributors"). The test is not how the language sounds                                                                                                                                                                                        it is whether money changes hands.
   Ask: is someone being paid to deliver something, or just asked to say something?

   Examples that MUST be discarded:
   - A governance forum post requesting community input on a proposal, even on an important topic.
   - A DAO inviting feedback on a token design or parameter change.
   - An ecosystem calling for contributors to co-design a community initiative.
   - An event or workshop open to community members with no vendor procurement attached.

   Examples that PASS this check:
   - A governance post funding a specific deliverable from an external party, with a named budget.
   - A DAO proposal with an on-chain payment mechanism and acceptance criteria for a vendor.
   - A Project Catalyst-style open grant application with a stated funding amount and deliverable scope.
   - An ecosystem RFP for a paid external service provider, posted via a governance channel.

   

9) USER-RESEARCH RECRUITMENT NOISE
   If the opportunity is based only on user testing, survey participation, interviews, beta testing, or generic
   product/usability feedback calls with no named integration path, DISCARD.
   Participant stipends do not count as commercial intake.
   If the post instead asks for feedback on a specific named integration, live API/dashboard/tooling surface, or wallet-security/data-intelligence rail, keep evaluating rather than auto-discarding.
   Wallet-security rails with live multi-chain deployment and a concrete protection surface should not be discarded
   only because the partner or support path is implicit rather than written as an explicit intake form; a clearly live
   external protection, integration, or support surface is enough to keep evaluating.

KEEP OVERRIDE (rules precedence):
- KEEP overrides REMOVE when explicit commercial intake is present.
- If evidence_snippet clearly includes vendor/RFP/procurement language, paid deliverable contract scope, explicit partner/integration program opening, or a scoped paid submission/prize/funded-selection path, do NOT discard under Rule 8 or Rule 9.

=== REFRAME RULES ===

Only reframe if ALL of the following are true:
- The core opportunity is genuine and the evidence supports it.
- The correction needed is to phrasing, framing, or sub-scores only.
- You do NOT need to add any new facts, companies, claims, or context not present in evidence_snippet.

If a reframe would require adding new information                                                                                                                                                                                                    DISCARD instead.

When reframing, provide a fully corrected opportunity object including corrected suggested_outreach_angle.
If explicit commercial intake is present but wording/contact path is weak, prefer REFRAME over DISCARD.

=== SUB-SCORE CORRECTIONS ===

Correct sub-scores when inflated or deflated. Common inflation patterns to watch for:
- fit_score=3 for highly niche technical requirements (should be 1-2).
- evidence_score=3 for implied intakes rather than explicit ones (should be 1-2).
- actionability_score=3 when no clear entry point is stated (should be 1-2).
- All scores=3 (maximum inflation) with generic or speculative outreach angles.

=== OUTPUT PER ITEM ===
- index (integer, required)
- status: keep | reframe | discard
- feedback: 1-2 sentences, specific. State which rule triggered the decision.
- corrected_sub_scores: if scores are inflated or deflated (even for kept items).
- reframed_opportunity: full corrected object if status=reframe.
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
            tracker.add_usage(resp.usage, m)

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
                    "posting_entity": gr.posting_entity,
                    "subject_entity": gr.subject_entity,
                    "bd_counterparty": gr.bd_counterparty,
                },
            }
        )

    system_prompt = f"""You are the REFINER agent. You are the final quality gate before opportunities are published to BD teams.

Your decisions are permanent. Errors you miss become false leads in the product.
Prefer DISCARD over doubt. One strong lead is worth more than three weak ones.

QUALITY CALIBRATION:
- Keep STRONG and MEDIUM opportunities.
- Medium is acceptable when explicit commercial intake exists and next-step action is plausible within horizon, even if contact path is weak.
- Discard only weak/noise opportunities.
- Protected class allowed: partner_program_intake / ecosystem_collaboration_intake with explicit near-term intake should be kept as medium quality.
  Keep lower priority than hard integration/RFP/procurement opportunities, but do not discard by default.
- Relevant reply paths for implementation surfaces can include ecosystem/devrel owners, maintainers, or other clearly relevant implementation owners.
- A scoped paid submission, prize, or funded-selection program can qualify when the deliverable, payment, and submission path are explicit.
- Do not auto-discard final reports/case studies, dashboard/API/tooling posts, or wallet-security/data-intelligence surfaces when the same evidence exposes a concrete implementation, integration, or support path for outside teams now.
- Do not discard wallet-security rails with live multi-chain deployment or dashboard/staking surfaces with live supported
  chains only because the call to action is implicit; if outside teams can realistically adopt, integrate, support, or
  build on the surface now, keep them as medium quality.
- A live dashboard/staking surface with supported chains and immediate user action such as "dashboard", "stake now",
  connect wallet, or visible positions is a valid adoption/integration surface even without explicit "apply now" wording.
- Do not keep repo migrations, maintenance rebuilds, or one-off code contribution chores unless the evidence clearly shows a broader external partner/cohort/funded rollout path.

=== NON-NEGOTIABLE CONSTRAINTS ===

1) No new opportunities.
   You may only DISCARD or REFRAME the opportunities provided.
   Do not introduce new companies, new partnerships, new products, or new claims.

2) No new facts.
   Every claim in the final output must be supported by the provided evidence_snippet.
   If a claim in the opportunity cannot be traced to specific words in evidence_snippet, remove it.
   When reframing, you may only use information already present in the opportunity or its evidence_snippet.
   If bd_signal is present in the gated event payload, use it only as supporting metadata when choosing between DISCARD and REFRAME. It must not add new facts, and it must never override evidence_snippet.
   If bd_signal.surface_type is present, treat it only as a weak indication of the likely surface class; use it to
   avoid misreading a live API/dashboard, security rail, named integration exploration, or live adoption surface as
   generic feedback or generic launch when evidence_snippet itself supports the stronger reading.
   If gate.posting_entity / gate.subject_entity / gate.bd_counterparty are present in the gated event payload, use them only
   to resolve who the real counterparty is. They help prevent proposer-vs-host confusion, but must not justify claims unsupported by evidence.

3) evidence_snippet must be verbatim.
   Do not paraphrase, extend, or modify evidence_snippet. Use it exactly as provided.

4) recommended_action must be grounded.
   recommended_action must be explicitly justified by exact words in evidence_snippet.
   If the justification is not there, remove recommended_action or DISCARD the opportunity.

5) suggested_outreach_angle must be specific.
   If suggested_outreach_angle is generic ("reach out to their partnerships team", "engage their BD team"),
   reframe it using information available in the opportunity and evidence_snippet.
   If no specific angle can be derived from the available evidence, DISCARD.

6) Time horizon: {TIME_HORIZON_STR}.
   Any opportunity not actionable within this window                                                                                                                                                                                                    DISCARD.

7) Prefer DISCARD over weak REFRAME.

=== ADDITIONAL DISCARD CONDITIONS ===

Apply these even if the Critic marked an item as 'keep':

SECONDARY INFERENCE:
If the opportunity only exists because of a deal or event between OTHER parties, and there is no explicit
external intake surface that any third party can act on                                                                                                                                                                                                    DISCARD.
Ask: "Is there an open door for a BD team to walk through, explicitly stated in the evidence?"
If the answer requires inference or optimism                                                                                                                                                                                                    DISCARD.
Do NOT use this discard when the target itself opened a named integration, live tooling/API/dashboard surface,
or shipped final-report implementation surface that outside teams can use or support now.



USER-RESEARCH RECRUITMENT NOISE:
If the opportunity is only user testing, surveys, interviews, beta testing, or generic community feedback participation
with no concrete build, integration, or partner execution path, DISCARD.
Participant stipends are not sufficient.
If it is only a repo migration, maintenance rebuild, or one-off code contribution task for an existing deployment, DISCARD unless the evidence clearly opens a broader external partner/cohort/funded rollout path.

GOVERNANCE PARTICIPATION-ONLY:
If action is limited to comment/vote/discussion/forum participation with no commercial intake, DISCARD.

KEEP OVERRIDE (must take precedence):
If explicit commercial intake exists, KEEP is allowed and should override the above removals.
Valid overrides: vendor/RFP/procurement language, paid deliverable contract scope, explicit partner/integration program opening, or a scoped paid submission/prize/funded-selection path.
When override conditions are met but outreach wording/contact specificity is weak, prefer REFRAME instead of DISCARD.

CROSS-HORIZON DUPLICATES:
If two opportunities in this batch describe the same action surface at the same target company,
even if their titles differ slightly or they were generated in different BD horizon passes                                                                                                                                                                                                    keep the higher-confidence
version and DISCARD the other.
Same action surface = a BD person would send the same outreach email for both.

=== OUTPUT ===
Output only JSON per schema.
Zero opportunities is an acceptable and sometimes correct output.
"""

    user_prompt = f"""GATED EVENTS:
{json.dumps(events_payload, default=_json_safe, separators=(",", ":"))}

ORIGINAL OPPORTUNITIES:
{json.dumps(original_opps, separators=(",", ":"))}

CRITIC FEEDBACK:
{json.dumps(critic_feedback, separators=(",", ":"))}

AVAILABLE CATEGORIES (use only category_text_id values):
{json.dumps(category_catalog, separators=(",", ":"))}

TASK:
Return a refined list of compliant opportunities (0 is acceptable)."""

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
        tracker.add_usage(resp.usage, model)

    raw = resp.choices[0].message.content
    parsed = json.loads(raw)
    return parsed.get("opportunities", [])


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


def validate_enriched_keep_output(original: Opportunity, candidate: Dict[str, Any]) -> bool:
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
    return True


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

    system_prompt = """You are a deterministic opportunity enrichment agent for final KEEP opportunities.

Your job is to improve clarity, conciseness, and BD usefulness while preserving all factual claims.

You must:
- Rewrite only title, summary, reason, suggested_outreach_angle, and opportunity_details.
- Preserve the same target company, evidence basis, categories, and BD motion.
- Keep all claims grounded in the provided opportunity payload and source events.
- Rebuild the key fields around the actual opportunity, not around general event narration.
- Make the opportunity itself unmistakable:
  - title should name the target company and the BD action/opportunity
  - summary should explain the opportunity surface, why it is actionable, and what the BD motion is
  - reason should justify why this is a valid opportunity now
  - suggested_outreach_angle should directly support winning this opportunity, not give generic networking advice
- Favor opportunity-centric wording over article-centric wording. Do not merely summarize the news/event.
- Write summary as 3-4 sentences, at least 35 words, and no more than 600 characters.
- Make suggested_outreach_angle a single crisp BD motion.
- Write opportunity_details as an internal-only field: 6-15 sentences, at least 70 words, and no more than 1600 total characters including spaces and punctuation.
- Always return opportunity_details for KEEP opportunities. If source material is thin, write the best grounded concise version that still fits the limits.

You must NOT:
- Introduce new entities, claims, timelines, or speculation.
- Change evidence basis or invent a different action surface.
- Rewrite categories, target_company, or evidence_snippet.
- Drift into generic polishing that weakens the opportunity framing.

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
        tracker.add_usage(resp.usage, model)

    raw = (resp.choices[0].message.content or "").strip()
    parsed = json.loads(raw)
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
        "opportunity_details": normalize_details_text(parsed.get("opportunity_details")),
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


def should_review_near_miss(row: Dict[str, Any], min_confidence: float) -> bool:
    stage = (row.get("stage") or "").strip().lower()
    if stage not in {"finder", "critic"}:
        return False

    reason = (row.get("reason") or "").strip().lower()
    haystack = near_miss_review_haystack(row)

    if reason == "below_min_confidence":
        return True

    if (
        (row.get("evidence_type") or "").strip() == "vendor_or_procurement_need"
        and _RECOVERY_HINT_RE.search(haystack)
    ):
        return True

    if (
        (row.get("evidence_type") or "").strip() == "grant_or_rfp_or_program_open"
        and (
            re.search(r"\b(open grant|grant proposal|application thread|applications? open|rfp)\b", haystack, flags=re.IGNORECASE)
            or re.search(r"\b(proposer|github\.com/|forum thread|reply in-thread)\b", haystack, flags=re.IGNORECASE)
        )
    ):
        return True

    if (
        (row.get("evidence_type") or "").strip() == "partnership_or_integration"
        and re.search(r"\b(admission|operator status|professional operator|trusted operator|audit|cover an audit|appetite to add)\b", haystack, flags=re.IGNORECASE)
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
    if (row.get("reason") or "").strip().lower() == "below_min_confidence":
        score += 2.0
        if confidence >= min_confidence - 0.05:
            score += 0.75
    if evidence_type == "grant_or_rfp_or_program_open":
        if re.search(r"\b(open grant|grant proposal|application thread|applications? open|rfp)\b", haystack, flags=re.IGNORECASE):
            score += 2.25
        if re.search(r"\b(proposer|github\.com/|forum thread|reply in-thread)\b", haystack, flags=re.IGNORECASE):
            score += 0.75
    if evidence_type == "partnership_or_integration":
        if re.search(r"\b(admission|operator status|professional operator|trusted operator)\b", haystack, flags=re.IGNORECASE):
            score += 2.0
        if re.search(r"\b(audit|cover an audit|appetite to add|add .* into the .* app)\b", haystack, flags=re.IGNORECASE):
            score += 2.0
    if re.search(r"\b(application|applications|apply|application thread|rfp|open grant)\b", haystack, flags=re.IGNORECASE):
        score += 1.5
    if re.search(r"\b(admission|operator status|professional operator)\b", haystack, flags=re.IGNORECASE):
        score += 1.2
    if re.search(r"\b(renew|renewal|annual engagement|services proposal|support/funding)\b", haystack, flags=re.IGNORECASE):
        score += 1.0
    if re.search(r"\b(integrat(?:e|ion)|integrator|partner program|early partner)\b", haystack, flags=re.IGNORECASE):
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

    return [item[3] for item in sorted(candidates)[:2]]


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
            -recovery_priority(r, min_confidence),
            str(r.get("title") or ""),
        )
    )
    return candidates[:cap]


def _polish_recovery_body_context(text: Optional[str], max_chars: int = 1600) -> str:
    cleaned = re.sub(r"\s+", " ", (clean_text(text) or "").strip()).strip()
    return smart_truncate(cleaned, max_chars) if cleaned else ""


def call_recovery_agent(
    client: OpenAI,
    model: str,
    candidates: List[Dict[str, Any]],
    category_catalog: List[Dict[str, str]],
    max_output_tokens: int,
    accepted_examples: Optional[List[Dict[str, Any]]] = None,
    tracker: Optional[TokenTracker] = None,
    force_one: bool = False,
) -> List[Dict[str, Any]]:
    system_prompt = f"""You are a deterministic Near-Miss Recovery Agent.

Your job is to review a SMALL set of borderline near-miss candidates and recover only the strongest real BD opportunities.

Recovery bar:
- Recover only when the evidence supports a real external action surface within {TIME_HORIZON_STR}.
- If unsure, do not recover.
- Prefer precision over creativity, but rescue obvious false negatives.

Patterns that may be recovered when explicitly grounded:
- open application / intake threads
- operator or provider admission / onboarding paths
- paid services proposal / renewal / annual engagement surfaces
- concrete integration partner / integrator build surfaces

Comparative rule:
- If a borderline candidate is at least as strong as the accepted_examples provided for this run, prefer recovery.
- Use accepted_examples as calibration for what already counts as a valid final opportunity.
- Do not demand stricter evidence from borderline candidates than from accepted_examples.

Important rescue patterns for this run:
- DAO/forum posts explicitly asking for audit funding or app-integration appetite
- node/operator admission applications with a real onboarding/admission path
- open grant proposals where the source event context clearly shows a live proposal/intake surface even if the short evidence_snippet is weak
- product-enabled integration launches where external teams can adopt now (for example: webhooks, SDKs, data feeds/streams,
  stablecoin support, wallet onboarding rails, or early-partner cohorts)

Do NOT key off specific project names or titles. Apply these rules by pattern only.

Rules:
- Use the provided evidence_snippet verbatim.
- Keep evidence_type grounded to the provided candidate.
- Preserve source_event_ids exactly as provided for the candidate.
- Do not invent new facts, entities, deadlines, or categories.
- Recovery is a promotion step only.
- If recover=true, return a full opportunity object with:
  - is_opportunity=true
- If recover=false, return opportunity=null.
- If recover=false, set opportunity=null.
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

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": openai_recovery_schema_v1()},
        temperature=0,
        max_completion_tokens=max_output_tokens,
    )

    if tracker:
        tracker.add_usage(resp.usage, model)

    raw = (resp.choices[0].message.content or "").strip()
    parsed = json.loads(raw) if raw else {}
    return parsed.get("decisions", []) or []



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

    system_prompt = """You are a strict filter classification agent for business development targeting.
Return JSON ONLY. No explanations.

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
- filter_sector = what domain the TARGET company operates in (their core business identity). Stable                                                                                                                                                                                        does not change opportunity to opportunity.
- filter_seeking = which partner/vendor capability is most relevant for BD outreach based on the specific evidence in this opportunity. Situational.

STRICT RULES:

1) Allowed IDs only.
   Never return an ID not listed above. If uncertain, use the appropriate Other ID (chain-oth, sect-oth, seek-oth).

2) chain-multi usage.
   Use chain-multi ONLY if the opportunity explicitly involves cross-chain, bridge, or interoperability requirements,
   OR explicitly states compatibility with 2+ named ecosystems as a requirement.
   If the company is broadly multi-chain but this opportunity is ecosystem-specific, tag the primary ecosystem only.
   If the primary ecosystem is unclear and it is not a cross-chain motion, use chain-oth.

3) Sector vs Seeking separation.
   Sector = what the target company IS (stable identity).
   Seeking = what this specific opportunity requires for BD outreach (situational).
   These must answer different questions.
   If you are using the same reasoning for both, one of them is wrong.
   Tag sect-ai ONLY if the target company's primary business is AI. Incidental AI mentions do not qualify.

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
        tracker.add_usage(resp.usage, model)

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

        # 3) If already a sector, keep canonical casing; otherwise DROP (fail-closed)
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


def dedupe_opportunities(opps: List[Opportunity]) -> List[Opportunity]:
    """Deduplicate opportunities.

    Client feedback: duplicates erode pilot credibility, but false merges can hide real opportunities.
    We therefore dedupe conservatively and later (post-structuring):
    1) Only exact event-set duplicates collapse automatically.
    2) Otherwise require high structured overlap and >80% fuzzy similarity across the core fields.
    3) If unsure, keep both.
    """
    if not opps:
        return []

    # 1) exact event-set dedupe only
    kept: List[Opportunity] = []
    seen_event_keys: set = set()

    for o in sorted(opps, key=lambda x: x.confidence, reverse=True):
        o_key = "|".join(sorted([eid for eid in (o.event_ids or "").split("|") if eid]))
        if o_key and o_key in seen_event_keys:
            continue
        kept.append(o)
        if o_key:
            seen_event_keys.add(o_key)

    # 2) high-certainty structured/fuzzy dedupe
    best_title: List[Opportunity] = []
    for o in kept:
        matched = False
        for existing in best_title:
            primary_cat_a = normalize_key((o.categories or [""])[0])
            primary_cat_b = normalize_key((existing.categories or [""])[0])
            ev_type_a = extract_evidence_type_from_reason(o.reason)
            ev_type_b = extract_evidence_type_from_reason(existing.reason)
            sims = _opportunity_similarity_signals(o, existing)

            same_structured_motion = (
                normalize_key(o.target_company or "") == normalize_key(existing.target_company or "")
                and primary_cat_a == primary_cat_b
                and ev_type_a == ev_type_b
            )
            strong_fuzzy = (
                sims["target_company"] >= 0.95
                and sims["outreach"] >= 0.80
                and (
                    sims["title"] >= 0.80
                    or sims["summary"] >= 0.80
                    or sims["reason"] >= 0.80
                )
            )
            if same_structured_motion and strong_fuzzy:
                matched = True
                break
        if not matched:
            best_title.append(o)

    # 3) target_company + category + evidence_type + action (late-stage structured dedup, still conservative)
    best_structured: Dict[str, Opportunity] = {}
    for o in best_title:
        primary_cat = normalize_key((o.categories or [""])[0])
        ev_type = extract_evidence_type_from_reason(o.reason)
        action = normalize_key(o.suggested_outreach_angle or "")
        if len(action) > 120:
            action = action[:120]
        key = f"{normalize_key(o.target_company or '')}||{primary_cat}||{ev_type}||{action}"
        incumbent = best_structured.get(key)
        if incumbent is None:
            best_structured[key] = o
            continue
        sims = _opportunity_similarity_signals(o, incumbent)
        if sims["target_company"] >= 0.95 and sims["outreach"] >= 0.80 and max(sims["title"], sims["summary"], sims["reason"]) >= 0.80:
            if o.confidence > incumbent.confidence:
                best_structured[key] = o
        else:
            alt_key = f"{key}||{o.opportunity_id}"
            best_structured[alt_key] = o

    return list(best_structured.values())


# ---------------------------
# Finalizer (post-processing)                                                                                                                                                                                        keep 5-agent logic intact
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


def _fuzzy_similarity(a: Optional[str], b: Optional[str]) -> float:
    aa = normalize_key(a or "")
    bb = normalize_key(b or "")
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def _opportunity_similarity_signals(a: Opportunity, b: Opportunity) -> Dict[str, float]:
    return {
        "title": _fuzzy_similarity(a.title, b.title),
        "summary": _fuzzy_similarity(a.summary, b.summary),
        "reason": _fuzzy_similarity(extract_reason_text_from_reason(a.reason), extract_reason_text_from_reason(b.reason)),
        "outreach": _fuzzy_similarity(a.suggested_outreach_angle, b.suggested_outreach_angle),
        "target_company": _fuzzy_similarity(a.target_company, b.target_company),
    }


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
    very high similarity between the core opportunity fields. If unsure, keep both.
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

                cand_ev_type = extract_evidence_type_from_reason(cand.reason)
                action_sim = _jaccard(base_action_toks, _tokenize_simple(cand.suggested_outreach_angle))
                title_sim = _jaccard(base_title_toks, _tokenize_simple(cand.title))
                fuzzy = _opportunity_similarity_signals(base, cand)

                should_merge = False
                same_motion = (
                    normalize_key(base.target_company or "") == normalize_key(cand.target_company or "")
                    and base_ev_type
                    and cand_ev_type
                    and base_ev_type == cand_ev_type
                )
                strong_fuzzy = (
                    fuzzy["target_company"] >= 0.95
                    and fuzzy["outreach"] >= 0.80
                    and max(fuzzy["title"], fuzzy["summary"], fuzzy["reason"]) >= 0.80
                )
                if same_motion and strong_fuzzy and (action_sim >= 0.60 or title_sim >= 0.75):
                    should_merge = True

                if not should_merge:
                    continue

                #& Merge cand into merged (keep merged's title/action; union sources/event ids/titles)
                merged = Opportunity(
                    opportunity_id=merged.opportunity_id,
                    title=merged.title,
                    summary=merged.summary,
                    opportunity_details=merged.opportunity_details or cand.opportunity_details,
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
                    bd_weeks=merged.bd_weeks,
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
    - Apply a deterministic KEEP/WATCHLIST/DROP policy without modifying any agent logic.

    Returns: (keep, watchlist, dropped)
    """
    opps = _consolidate_similar_opportunities(opps)

    keep: List[Opportunity] = []
    watch: List[Opportunity] = []
    drop: List[Opportunity] = []
    target_counts = Counter(
        normalize_key(o.target_company or o.title or "")
        for o in opps
        if normalize_key(o.target_company or o.title or "")
    )

    for o in opps:
        primary_cat = (o.categories or [""])[0]
        cat_norm = normalize_key(primary_cat)
        ev_type = extract_evidence_type_from_reason(o.reason)
        ev_snip = extract_evidence_snippet_from_reason(o.reason)
        full_text = " ".join([
            clean_text(o.title) or "",
            clean_text(o.summary) or "",
            extract_reason_text_from_reason(o.reason),
            clean_text(o.suggested_outreach_angle) or "",
            clean_text(ev_snip) or "",
        ])
        target_key = normalize_key(o.target_company or o.title or "")

        audit_bucket = audit_guardrail_bucket(
            opportunity=o,
            full_text=full_text,
            evidence_type=ev_type,
            evidence_snippet=ev_snip,
            duplicate_target_present=target_counts.get(target_key, 0) > 1,
        )
        if audit_bucket == "drop":
            drop.append(o)
            continue
        if audit_bucket == "watch":
            watch.append(o)
            continue

        # Drop policy-heavy items that lack a concrete, near-term BD hook.
        if ("regulat" in cat_norm or "policy" in cat_norm or "compliance" in cat_norm or ev_type == "regulatory_or_compliance_update"):
            mega = is_mega_counterparty(o.target_company or "") or is_mega_counterparty(o.title)
            if mega and not has_open_motion(ev_type, ev_snip, title=o.title, target_company=o.target_company or ""):
                drop.append(o)
                continue

        # Funding-only items without clear external action are watchlist at best.
        if ("fund" in cat_norm or "investment" in cat_norm) and not has_open_motion(ev_type, ev_snip, title=o.title, target_company=o.target_company or ""):
            watch.append(o)
            continue

        # Vertical-dependent items: keep as watchlist (can be promoted based on ICP).
        if any(k in cat_norm for k in ["ticket", "entertain", "gaming", "nft"]):
            watch.append(o)
            continue

        # Indirect support / enablement pattern: a real opportunity exists, but the motion is one layer removed from
        # the strongest direct-intake cases. These should usually remain visible as WATCHLIST rather than be dropped.
        if ev_type == "grant_or_rfp_or_program_open":
            indirect_execution = bool(re.search(
                r"\b(technical sponsor|implementation partner|delivery partner|execution support|collaborator|"
                r"co-author|coauthor|program delivery|curriculum execution|workshops|office hours)\b",
                full_text,
                flags=re.IGNORECASE,
            ))
            indirect_enablement = bool(re.search(
                r"\b(help (?:the )?proposer|support (?:the )?proposer|support execution|execution plan|"
                r"milestone plan|delivery scaffolding|onboarding/ops support|enablement|pilot integration|"
                r"beta adopter|beta onboarding|technical evaluation|authorization|authorized|whitelist|"
                r"access path|introduction path|ecosystem access|registry access|write access)\b",
                full_text,
                flags=re.IGNORECASE,
            ))
            direct_intake = bool(re.search(
                r"\b(submit|apply|application|applications open|register|registration|rfp|request for proposal|"
                r"proposal to|reply in-thread|reply in thread|fillout|form)\b",
                full_text,
                flags=re.IGNORECASE,
            ))
            direct_procurement_or_admission = bool(re.search(
                r"\b(audit kickoff|external auditor|vendor selection|operator admission|professional operator|"
                r"trusted operator|admissions pipeline|shortlisted|procurement)\b",
                full_text,
                flags=re.IGNORECASE,
            ))
            if (indirect_execution or indirect_enablement) and not direct_intake and not direct_procurement_or_admission:
                watch.append(o)
                continue

        keep.append(o)

    return keep, watch, drop


def write_opportunities_csv(path: str, opps: List[Opportunity], status: str) -> None:
    """Write an opportunities CSV for operator review.

    This is a lightweight reporting artifact; it does not affect DB writes.
    """
    fieldnames = [
        "status",
        "confidence",
        "title",
        "summary",
        "opportunity_details",
        "target_company",
        "primary_category",
        "suggested_outreach_angle",
        "evidence_type",
        "evidence_snippet",
        "sources",
        "event_url",
        "bd_weeks",
        "event_ids",
        "event_titles",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for o in sorted(opps, key=lambda x: x.confidence, reverse=True):
            primary_cat = (o.categories or [""])[0]
            ev_type = extract_evidence_type_from_reason(o.reason)
            ev_snip = extract_evidence_snippet_from_reason(o.reason)
            w.writerow({
                "status": status,
                "confidence": f"{o.confidence:.3f}",
                "title": o.title,
                "summary": o.summary or "",
                "opportunity_details": o.opportunity_details or "",
                "target_company": o.target_company or "",
                "primary_category": primary_cat,
                "suggested_outreach_angle": o.suggested_outreach_angle or "",
                "evidence_type": ev_type,
                "evidence_snippet": ev_snip,
                "sources": " | ".join(o.sources or []),
                "event_url": o.event_url or "",
                "bd_weeks": o.bd_weeks,
                "event_ids": o.event_ids or "",
                "event_titles": o.event_titles or "",
            })


def write_opportunities_json(path: str, opps: List[Opportunity], status: str) -> None:
    """JSON backup artifact for cases where DB persistence fails after local scoring succeeds."""
    payload = []
    for o in sorted(opps, key=lambda x: x.confidence, reverse=True):
        payload.append({
            "status": status,
            "opportunity_id": o.opportunity_id,
            "confidence": o.confidence,
            "title": o.title,
            "summary": o.summary or "",
            "opportunity_details": o.opportunity_details or "",
            "target_company": o.target_company or "",
            "categories": o.categories or [],
            "suggested_outreach_angle": o.suggested_outreach_angle or "",
            "who_to_contact": o.who_to_contact or "",
            "sources": o.sources or [],
            "event_url": o.event_url or "",
            "bd_weeks": o.bd_weeks,
            "event_ids": o.event_ids or [],
            "event_titles": o.event_titles or [],
            "time_found": o.time_found,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_safe)


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
            w.writerow({k: row.get(k, "") for k in fieldnames})
def build_reason_block(reason: str, evidence_type: str, evidence_snippet: str) -> str:
    return (
        f"Evidence type: {evidence_type}\n"
        f"Evidence snippet: {evidence_snippet}\n"
        f"Reason: {reason}"
    )


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

    reason_block = build_reason_block(clean_text(ro.get("reason", "")) or "", evidence_type, clean_text(evidence_snippet) or "")
    recommended_action = clean_text(ro.get("recommended_action") or "") or ""
    suggested_outreach_angle = clean_text(ro.get("suggested_outreach_angle") or "") or ""
    tc = normalize_target_company(ro.get("target_company"))
    if not tc:
        tc = derive_target_company(event_urls, sources)
    tc = clean_text(tc)

    opport = Opportunity(
        opportunity_id=str(uuid.uuid4()),
        title=(clean_text(ro.get("title", "")) or "").strip(),
        summary=((clean_text(ro.get("summary", "")) or "").strip()[:800] or (clean_text(ro.get("reason", "")) or "").strip()[:800] or "Opportunity"),
        opportunity_details=None,
        reason=reason_block,
        who_to_contact=clean_text(ro.get("who_to_contact")),
        suggested_outreach_angle=(suggested_outreach_angle or recommended_action),
        categories=validated_categories,
        filter_chain="chain-oth",
        filter_sector="sect-oth",
        filter_seeking="seek-oth",
        time_found=_now_iso(),
        confidence=float(ro.get("confidence", 0)),
        tags=[clean_text(t) for t in (ro.get("tags", []) or []) if clean_text(t)],
        target_company=tc,
        sources=[clean_text(s) for s in sources if clean_text(s)],
        event_ids="|".join(event_ids_list),
        event_titles="|".join(event_titles),
        event_url="|".join(event_urls),
        bd_weeks=int(TIME_HORIZON_WEEKS),
    )
    return opport, None


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
    all_opps: List[Opportunity] = []
    all_gate_results: List[Dict[str, Any]] = []  # Track all gatekeeper outputs
    rejection_audit: List[Dict[str, Any]] = []   # Track downstream rejects (coverage/recall diagnostics)
    near_miss_watchlist: List[Dict[str, Any]] = []  # Manual-review near misses; does not affect final outputs
    event_lookup = {ev.event_id: ev for ev in events}

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

    # Batch-level concurrency:
    # - Each worker runs the full 5-agent pipeline sequentially for its batch.
    # - New batches are picked up by the next available worker.

    def _run_one_batch(batch_num: int, batch_events: List[Event]) -> Dict[str, Any]:
        local_opps: List[Opportunity] = []
        local_gate_results: List[Dict[str, Any]] = []
        local_rejection_audit: List[Dict[str, Any]] = []
        local_near_miss_watchlist: List[Dict[str, Any]] = []
        local_client = client.get_client()
        thread_name = pretty_thread_name()

        print(
            f"[OpportunityMatcher] Batch {batch_num}/{len(intelligent_batches)} assigned to {thread_name} "
            f"({len(batch_events)} events)"
        )

        try:
            gate_map = call_gatekeeper(local_client, gatekeeper_model, batch_events, tracker)
        except Exception as e:
            print(f"  WARNING: Gatekeeper failed: {e}")
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
                "posting_entity": gate_result.posting_entity,
                "subject_entity": gate_result.subject_entity,
                "bd_counterparty": gate_result.bd_counterparty,
                "passes_gate": gate_result.passes(),
            })

        gated: List[Tuple[Event, GateResult]] = []
        excluded = 0
        for ev in batch_events:
            gr = gate_map.get(ev.event_id)
            if not gr:
                excluded += 1
                local_rejection_audit.append({
                    "stage": "gatekeeper",
                    "event_id": ev.event_id,
                    "event_title": ev.title or "",
                    "decision": "excluded",
                    "reason": "missing_gate_result",
                })
                continue
            if gr.passes():
                gated.append((ev, gr))
            else:
                excluded += 1
                local_rejection_audit.append({
                    "stage": "gatekeeper",
                    "event_id": ev.event_id,
                    "event_title": ev.title or "",
                    "decision": "excluded",
                    "reason": gr.hard_exclusion_reason if gr.hard_exclusion else ("no_qualifying_evidence" if gr.evidence_type == "none" else "no_verbatim_evidence"),
                    "hard_exclusion": gr.hard_exclusion,
                    "evidence_type": gr.evidence_type,
                })

        print(f"  [Gatekeeper- Batch{batch_num}] Passed: {len(gated)} | Excluded: {excluded}")
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

        for o in raw_opps:
            if not isinstance(o, dict):
                continue
            supporting_snippet = clean_text(o.get("supporting_snippet")) or ""
            if not supporting_snippet:
                for eid in normalize_source_event_ids(o.get("source_event_ids")):
                    gate_result = gate_map.get(str(eid))
                    if gate_result and gate_result.supporting_snippet:
                        supporting_snippet = gate_result.supporting_snippet
                        break
            if supporting_snippet:
                o["supporting_snippet"] = supporting_snippet

        filtered: List[Dict[str, Any]] = []
        filtered_reasons = {"not_opportunity": 0, "low_confidence": 0}
        batch_title_map = {ev.event_id: (ev.title or "").strip() for ev in batch_events}
        batch_event_map = {ev.event_id: ev for ev in batch_events}

        def _near_miss_context(o: Dict[str, Any]) -> Dict[str, Any]:
            event_ids = [str(eid) for eid in (o.get("source_event_ids", []) or []) if str(eid)]
            source_events = [batch_event_map[eid] for eid in event_ids if eid in batch_event_map]
            summary = ""
            body_preview = ""
            bd_signal = {}
            supporting_snippet = clean_text(o.get("supporting_snippet")) or ""
            derived_target_company = clean_text(o.get("target_company")) or ""
            if source_events:
                ev = source_events[0]
                summary = clean_text(ev.summary) or ""
                desc = clean_text(ev.description) or ""
                body_preview = desc[:3000]
                bd_signal = ev.bd_signal or {}
                if not supporting_snippet:
                    gate_result = gate_map.get(ev.event_id)
                    supporting_snippet = clean_text(getattr(gate_result, "supporting_snippet", None)) or ""
                if not derived_target_company:
                    derived_target_company = derive_target_company(
                        [e.url for e in source_events if e.url],
                        [e.source for e in source_events if e.source],
                    ) or ""
            return {
                "summary": summary,
                "body_preview": body_preview,
                "target_company": derived_target_company,
                "categories": o.get("categories", []) or [],
                "reason_text": clean_text(o.get("reason")) or "",
                "suggested_outreach_angle": clean_text(o.get("suggested_outreach_angle")) or "",
                "supporting_snippet": supporting_snippet,
                "bd_signal": bd_signal,
            }

        def _debug_title(o: Dict[str, Any]) -> str:
            title = (o.get("title") or "").strip()
            if title:
                return title
            alt_title = (o.get("opportunity_title") or "").strip()
            if alt_title:
                return alt_title
            for eid in o.get("source_event_ids", []) or []:
                mapped = batch_title_map.get(str(eid), "").strip()
                if mapped:
                    return mapped
            target = (o.get("target_company") or "").strip()
            return target or "Untitled"

        for o in raw_opps:
            source_events_for_candidate = [
                batch_event_map[eid]
                for eid in [str(eid) for eid in (o.get("source_event_ids", []) or [])]
                if eid in batch_event_map
            ]
            o = apply_pattern_guard_to_candidate(o, source_events_for_candidate)
            if not o.get("is_opportunity", False):
                filtered_reasons["not_opportunity"] += 1
                dbg_title = _debug_title(o)
                print(f"    DEBUG: Filtered '{dbg_title[:80]}' - is_opportunity=False")
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
            o["confidence"] = conf
            filtered.append(o)

        print(f"  [Finder- Batch{batch_num}] Produced: {len(raw_opps)} | Kept after gate+confidence: {len(filtered)}")
        if len(raw_opps) > len(filtered):
            print(
                f"    Filtered: {filtered_reasons['not_opportunity']} (is_opportunity=False), "
                f"{filtered_reasons['low_confidence']} (confidence<{min_confidence})"
            )

        if not filtered:
            return {"batch_num": batch_num, "thread_name": thread_name, "opps": [], "gate": local_gate_results, "rejects": local_rejection_audit, "near_miss": local_near_miss_watchlist}

        for o in filtered:
            if not isinstance(o, dict):
                continue
            posting_entity = clean_text(o.get("posting_entity")) or ""
            subject_entity = clean_text(o.get("subject_entity")) or ""
            bd_counterparty = clean_text(o.get("bd_counterparty")) or ""
            if posting_entity and subject_entity and bd_counterparty:
                continue
            for eid in normalize_source_event_ids(o.get("source_event_ids")):
                gate_result = gate_map.get(str(eid))
                if not gate_result:
                    continue
                posting_entity = posting_entity or clean_text(gate_result.posting_entity) or ""
                subject_entity = subject_entity or clean_text(gate_result.subject_entity) or ""
                bd_counterparty = bd_counterparty or clean_text(gate_result.bd_counterparty) or ""
                if posting_entity and subject_entity and bd_counterparty:
                    break
            if posting_entity:
                o["posting_entity"] = posting_entity
            if subject_entity:
                o["subject_entity"] = subject_entity
            if bd_counterparty:
                o["bd_counterparty"] = bd_counterparty

        print(f"  [Critic- Batch{batch_num}][{thread_name}]", end=" ")
        feedback = call_critic(local_client, critic_model, filtered, {ev.event_id: ev for ev in batch_events}, tracker)
        rating = float(feedback.get("overall_rating", 0))
        print(f"rated {rating}/10")

        final_items: List[Dict[str, Any]] = []
        fb_items = feedback.get("opportunity_feedback", []) or []
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
            status = fb.get("status")
            if status == "discard":
                local_rejection_audit.append({
                    "stage": "critic",
                    "decision": "discard",
                    "title": filtered[idx].get("title", ""),
                    "target_company": filtered[idx].get("target_company"),
                    "source_event_ids": filtered[idx].get("source_event_ids", []),
                    "feedback": fb.get("feedback", ""),
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
                    "feedback": fb.get("feedback", ""),
                    "sub_scores": filtered[idx].get("sub_scores", {}) or {},
                    **_near_miss_context(filtered[idx]),
                })
                continue
            if status == "keep":
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

        if rating < 8.5 or len(final_items) == 0:
            print(f"  [Refiner][Batch {batch_num}][{thread_name}]", end=" ")
            try:
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
                final_items = refined_keep
                print(f"-> {len(final_items)}")
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

        time_found = datetime.now(timezone.utc).date().isoformat()
        for ro in final_items:
            event_ids_list = ro.get("source_event_ids", [])
            if not event_ids_list:
                continue

            sources: List[str] = []
            event_titles: List[str] = []
            event_urls: List[str] = []
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

            sources = [clean_text(s) for s in sources if clean_text(s)]
            evidence_type = ro.get("evidence_type")
            evidence_snippet = enforce_verbatim_snippet(ro.get("evidence_snippet")) or ""
            reason_block = build_reason_block(clean_text(ro.get("reason", "")) or "", evidence_type, clean_text(evidence_snippet) or "")
            recommended_action = clean_text(ro.get("recommended_action") or "") or ""
            suggested_outreach_angle = clean_text(ro.get("suggested_outreach_angle") or "") or ""

            validated_categories = validate_categories(
                ro.get("categories", []),
                evidence_type,
                evidence_snippet,
                categories_map,
                category_definitions,
            )
            if not validated_categories:
                local_rejection_audit.append({
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
                title=(clean_text(ro.get("title", "")) or "").strip(),
                summary=((clean_text(ro.get("summary", "")) or "").strip()[:800] or (clean_text(ro.get("reason", "")) or "").strip()[:800] or "Opportunity"),
                opportunity_details=None,
                reason=reason_block,
                who_to_contact=clean_text(ro.get("who_to_contact")),
                suggested_outreach_angle=(suggested_outreach_angle or recommended_action),
                categories=validated_categories,
                filter_chain="chain-oth",
                filter_sector="sect-oth",
                filter_seeking="seek-oth",
                time_found=time_found,
                confidence=float(ro.get("confidence", 0)),
                tags=[clean_text(t) for t in (ro.get("tags", []) or []) if clean_text(t)],
                target_company=tc,
                sources=sources,
                event_ids="|".join(event_ids_list),
                event_titles="|".join(event_titles),
                event_url="|".join(event_urls),
                bd_weeks=int(TIME_HORIZON_WEEKS),
            )
            local_opps.append(opport)

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

    recovery_candidates = select_recovery_candidates(
        near_miss_watchlist,
        event_lookup,
        min_confidence=min_confidence,
        cap=RECOVERY_MAX_CANDIDATES,
    )
    if recovery_candidates:
        accepted_examples = build_recovery_reference_examples(all_opps, max_items=6)
        recovery_payload: List[Dict[str, Any]] = []
        recovery_payload_rows: List[Dict[str, Any]] = []
        for row in recovery_candidates:
            source_events = select_recovery_source_events(row, event_lookup)
            if not source_events:
                continue
            idx = len(recovery_payload_rows)
            recovery_payload_rows.append(row)
            recovery_payload.append(
                {
                    "index": idx,
                    "near_miss": {
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

        if recovery_payload:
            print(f"[OpportunityMatcher] Running Recovery Agent on {len(recovery_payload)} near-miss candidates...")
            recovered_opps: List[Opportunity] = []
            recovered_row_ids: Set[int] = set()

            def _apply_recovery_decisions(decisions: List[Dict[str, Any]]) -> None:
                nonlocal recovered_opps, recovered_row_ids
                for decision in decisions:
                    if not isinstance(decision, dict) or not decision.get("recover"):
                        continue
                    idx = int(decision.get("index", -1))
                    if idx < 0 or idx >= len(recovery_payload_rows):
                        continue
                    row = recovery_payload_rows[idx]
                    ro = decision.get("opportunity")
                    if not isinstance(ro, dict) or not ro.get("is_opportunity", False):
                        continue
                    ro["source_event_ids"] = normalize_source_event_ids(row.get("source_event_ids"))
                    ro["evidence_type"] = row.get("evidence_type")
                    ro["evidence_snippet"] = enforce_verbatim_snippet(row.get("evidence_snippet")) or ""
                    conf = confidence_from_subscores(ro.get("sub_scores", {}) or {})
                    if conf < min_confidence:
                        continue
                    ro["confidence"] = conf
                    built, _build_error = build_opportunity_from_raw_candidate(
                        ro,
                        event_lookup,
                        categories_map,
                        category_definitions,
                    )
                    if not built:
                        continue
                    if any(existing.event_ids == built.event_ids and existing.title == built.title for existing in recovered_opps):
                        continue
                    recovered_opps.append(built)
                    recovered_row_ids.add(id(row))
                    rejection_audit.append({
                        "stage": "recovery_agent",
                        "decision": "recovered",
                        "title": built.title,
                        "target_company": built.target_company,
                        "source_event_ids": normalize_source_event_ids(row.get("source_event_ids")),
                        "feedback": decision.get("rationale", ""),
                    })
                    if len(recovered_opps) >= RECOVERY_MAX_RESTORED:
                        break

            try:
                recovery_decisions = call_recovery_agent(
                    client=client.get_client(),
                    model=DEFAULT_RECOVERY_MODEL,
                    candidates=recovery_payload,
                    category_catalog=category_catalog,
                    accepted_examples=accepted_examples,
                    max_output_tokens=max_output_tokens,
                    tracker=tracker,
                )
                _apply_recovery_decisions(recovery_decisions)
                if not recovered_opps and recovery_payload:
                    fallback_payload = recovery_payload[: min(3, len(recovery_payload))]
                    fallback_rows = recovery_payload_rows[: len(fallback_payload)]
                    fallback_decisions = call_recovery_agent(
                        client=client.get_client(),
                        model=DEFAULT_RECOVERY_MODEL,
                        candidates=fallback_payload,
                        category_catalog=category_catalog,
                        accepted_examples=accepted_examples,
                        max_output_tokens=max_output_tokens,
                        tracker=tracker,
                        force_one=True,
                    )
                    original_rows = recovery_payload_rows
                    recovery_payload_rows = fallback_rows
                    try:
                        _apply_recovery_decisions(fallback_decisions)
                    finally:
                        recovery_payload_rows = original_rows
            except Exception as e:
                print(f"[OpportunityMatcher] WARNING: Recovery Agent failed: {e}")
                recovered_opps = []
                recovered_row_ids = set()

            if recovered_opps:
                all_opps.extend(recovered_opps)
                near_miss_watchlist = [r for r in near_miss_watchlist if id(r) not in recovered_row_ids]
            print(f"[OpportunityMatcher] Recovery Agent restored {len(recovered_opps)} opportunities (cap={RECOVERY_MAX_RESTORED})")

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
    # Finalizer enforcement: DROP items are never saved as opportunities for MVP.
    final_opps_for_db = keep_opps
    print(
        f"[OpportunityMatcher] Finalizer: KEEP={len(keep_opps)} | WATCHLIST={len(watch_opps)} | DROP={len(drop_opps)}"
    )

# ---------------------------
# FILTER AGENT (Batch Mode)
# ---------------------------

##^ commented on 24-Feb-> chains_map, sectors_map, seeking_map = load_filter_master(str(FILTER_MASTER_JSON_PATH))

    if keep_opps:
        event_lookup = {ev.event_id: ev for ev in events}
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
                if validate_enriched_keep_output(o, enriched):
                    evidence_type = extract_evidence_type_from_reason(o.reason) or ""
                    evidence_snippet = extract_evidence_snippet_from_reason(o.reason) or ""
                    o.title = enriched["title"]
                    o.summary = enriched["summary"]
                    o.reason = build_reason_block(enriched["reason"], evidence_type, evidence_snippet)
                    o.suggested_outreach_angle = enriched["suggested_outreach_angle"]
                    o.opportunity_details = enriched.get("opportunity_details")
            except Exception as e:
                print(f"[OpportunityMatcher] WARNING: KEEP enrichment failed for {o.opportunity_id}: {e}")

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
    try:
        write_opportunities_csv(str(OUTPUT_DIR / f"opportunities_KEEP_{ts}.csv"), keep_opps, "KEEP")
        write_opportunities_csv(str(OUTPUT_DIR / f"opportunities_WATCHLIST_{ts}.csv"), watch_opps, "WATCHLIST")
        write_opportunities_csv(str(OUTPUT_DIR / f"opportunities_DROP_{ts}.csv"), drop_opps, "DROP")
    except Exception as e:
        print(f"[OpportunityMatcher] WARNING: Failed to write opportunity CSV artifacts: {e}")
        append_crash_log("write_opportunities_csv", e)

    # Save Gatekeeper results to JSON for auditing
    if all_gate_results:
        gatekeeper_output_file = str(OUTPUT_DIR / "gatekeeper_results.json")
        try:
            with open(gatekeeper_output_file, "w") as f:
                json.dump(all_gate_results, f, indent=2, default=_json_safe)
            print(f"[OpportunityMatcher] Saved {len(all_gate_results)} gatekeeper results to {gatekeeper_output_file}")
        except Exception as e:
            print(f"[OpportunityMatcher] WARNING: Failed to write gatekeeper audit: {e}")
            append_crash_log("write_gatekeeper_audit", e)

    # Save rejection audit for recall/coverage diagnostics (does not affect DB writes)
    if rejection_audit:
        rejection_output_file = str(OUTPUT_DIR / "rejection_audit.json")
        try:
            with open(rejection_output_file, "w") as f:
                json.dump(rejection_audit, f, indent=2, default=_json_safe)
            print(f"[OpportunityMatcher] Saved {len(rejection_audit)} rejection audit rows to {rejection_output_file}")
        except Exception as e:
            print(f"[OpportunityMatcher] WARNING: Failed to write rejection audit: {e}")
            append_crash_log("write_rejection_audit", e)

    # Save near-miss watchlist for manual review (does not affect final outputs)
    if near_miss_watchlist:
        ts_nm = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        near_miss_csv = str(OUTPUT_DIR / f"near_miss_watchlist_{ts_nm}.csv")
        try:
            write_near_miss_watchlist_csv(near_miss_csv, near_miss_watchlist)
            print(f"[OpportunityMatcher] Saved near-miss CSV to {near_miss_csv}")
        except Exception as e:
            print(f"[OpportunityMatcher] WARNING: Failed to write near-miss watchlist artifacts: {e}")
            append_crash_log("write_near_miss_watchlist", e)

    if keep_opps:
        try:
            saved = db.save_opportunities(final_opps_for_db)
            print(f"[OpportunityMatcher] Saved {saved} opportunities")
            return saved
        except Exception as e:
            print(f"[OpportunityMatcher] ERROR: Failed to save opportunities: {e}")
            append_crash_log("save_opportunities", e)
            try:
                ts_db = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_json = str(OUTPUT_DIR / f"opportunities_KEEP_db_save_failed_{ts_db}.json")
                write_opportunities_json(backup_json, keep_opps, "KEEP")
                print(
                    "[OpportunityMatcher] WARNING: DB save failed, but KEEP opportunities were preserved locally at "
                    f"{backup_json}"
                )
            except Exception as backup_error:
                print(f"[OpportunityMatcher] WARNING: Failed to write DB-failure backup artifact: {backup_error}")
                append_crash_log("write_db_failure_backup", backup_error)
            return 0
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
        choices=[2, 4, 8, 10, 12, 15, 20, 30],
        help="Time horizon (in weeks) enforced across ALL agent prompts (1-N weeks). Allowed: 2, 4, 8, 10, 12, 15, 20, 30.",
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
    TIME_HORIZON_STR = f"1-{horizon_weeks} weeks"

    # Candidate mode tuning
    if args.candidate_mode == "expanded":
        # Keep confidence strict; only increase completion budget
        if args.max_output_tokens < 2800:
            args.max_output_tokens = 2800

    print(f"[OpportunityMatcher] Script started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("[OpportunityMatcher] Mode: EVENT-CENTRIC (client-gate compliant)")
    print(
        f"[OpportunityMatcher] Models: Finder={args.finder_model}, Gatekeeper={args.gatekeeper_model}, Critic={args.critic_model}, Enrichment={args.enrichment_model}, Refiner={args.refiner_model}, Filter={getattr(args, 'filter_model', DEFAULT_FILTER_MODEL)}"
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
        append_crash_log("openai_client_init", e)
        return 1

    try:
        config_path = SCRIPT_DIR / "Database_Connection.csv"
        db_config = _read_database_config(str(config_path))
    except Exception as e:
        print(f"[OpportunityMatcher] ERROR reading database config: {e}")
        append_crash_log("read_database_config", e)
        return 1

    db_schema = db_config["DB_SCHEMA"]
    try:
        db = DatabaseManager(
            db_config["DB_HOST"],
            int(db_config["DB_PORT"]),
            db_config["DB_USER"],
            db_config["DB_PASSWORD"],
            db_config["DB_NAME"],
            schema=db_schema,
        )
    except Exception as e:
        print(f"[OpportunityMatcher] ERROR: Failed to connect to database: {e}")
        append_crash_log("database_init", e)
        return 1

    print(
        "[OpportunityMatcher] Connected to database: "
        f"host={db_config['DB_HOST']} port={db_config['DB_PORT']} "
        f"dbname={db_config['DB_NAME']} schema={db_schema} user={db_config['DB_USER']}"
    )

    tracker = TokenTracker()
    try:
        # Helpful diagnostics; especially useful when the script unexpectedly loads 0 events/categories.
        db.debug_db_snapshot()

        # Load events from JSONL (Model2_v12 behavior)
        events_path = str(FINAL_EVENTS_JSONL_PATH) if not str(args.input).strip() else str(args.input).strip()
        try:
            events = load_events_from_jsonl(events_path, days_back=args.days_back)
        except Exception as e:
            print(f"[OpportunityMatcher] ERROR: Failed to load events from JSONL: {e}")
            append_crash_log("load_events_from_jsonl", e)
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
    except Exception as e:
        print(f"[OpportunityMatcher] FATAL: Unhandled runtime error: {e}")
        append_crash_log("main_runtime", e)
        return 1
    finally:
        try:
            db.close()
        except Exception as e:
            print(f"[OpportunityMatcher] WARNING: Failed to close DB connection cleanly: {e}")

    elapsed = time.time() - script_start_time
    print(f"\n[OpportunityMatcher] COMPLETE - Total opportunities saved: {total}")
    print(f"[OpportunityMatcher] Execution time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")

    tracker.print_summary()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())















