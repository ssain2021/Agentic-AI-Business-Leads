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
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openai import OpenAI



import hashlib
from datetime import timedelta
from pathlib import Path


_NULL_LIKE = {"", "null", "none", "n/a", "na", "unknown", "-"}


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
            summary = obj.get("summary_text") or obj.get("summary") or obj.get("rss_summary_text") or ""
            body = obj.get("body_text") or obj.get("description") or obj.get("content") or ""
            source = obj.get("source") or obj.get("source_name") or ""
            url = obj.get("url") or obj.get("link") or ""

            events.append(Event(
                event_id=str(event_id),
                title=title,
                summary=summary,
                description=body,
                tags=[],
                published_at=dt.isoformat() if dt else (published_raw or ""),
                source=source,
                url=url,
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

def enforce_verbatim_snippet(snippet: Optional[str], max_words: int = 20) -> Optional[str]:
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
    s = (desc or "")
    if len(s) <= head + tail:
        return s
    return s[:head] + "\n...\n" + s[-tail:]


def slice_body_for_finder(desc: Optional[str], head: int = 4000, tail: int = 4000) -> str:
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


class TokenTracker:
    """Track OpenAI token usage and estimated costs."""

    def __init__(self):
        self.model_usage = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_estimated_cost = 0.0

    def get_pricing(self, model: str):
        """Return (input_cost_per_1m, output_cost_per_1m) in USD per 1M tokens."""
        pricing = {
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
            "o3": (2.0, 8.0),
            "gpt-5.1": (1.25, 10.0),
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


# ---------------------------
# Configuration
# ---------------------------

DEFAULT_FILTER_MODEL = "gpt-5.1"
FILTER_BATCH_SIZE = 10

DEFAULT_MODEL = "gpt-5.1"      # Finder + Batcher default
DEFAULT_GATEKEEPER_MODEL = "gpt-4o-mini"
DEFAULT_CRITIC_MODEL = "o3"
DEFAULT_REFINER_MODEL = "gpt-5.1"

DEFAULT_MAX_EVENTS_PER_BATCH = 20
DEFAULT_MAX_OUTPUT_TOKENS = 2200
DEFAULT_MIN_CONFIDENCE = 0.60


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
        }


@dataclass
class GateResult:
    event_id: str
    hard_exclusion: bool
    hard_exclusion_reason: Optional[str]
    evidence_type: str
    evidence_snippet: Optional[str]

    def passes(self) -> bool:
        return (not self.hard_exclusion) and self.evidence_type != "none" and bool(self.evidence_snippet)


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
                bd_weeks
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
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
                bd_weeks = EXCLUDED.bd_weeks
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
                        },
                        "required": [
                            "event_id",
                            "hard_exclusion",
                            "hard_exclusion_reason",
                            "evidence_type",
                            "evidence_snippet",
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
    opp_items_schema = openai_opportunity_schema_v2().get("items", {})

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

    # Adjust token limit based on batch size (minimum 1800, scale up for larger batches)
# Conservative scaling to prevent truncation
    tokens_per_event = 180      # was 100
    base_tokens = 1200          # was 800
    safety_buffer = 800         # extra headroom for JSON structure

    max_tokens = base_tokens + (len(events) * tokens_per_event) + safety_buffer

    # Hard cap to prevent runaway cost
    max_tokens = min(max_tokens, 6000)

    events_payload = [
        {
            "event_id": ev.event_id,
            "title": ev.title or "",
            "summary": ev.summary or "",
            "description": slice_body_for_gatekeeper(ev.description),
            "source": ev.source or "",
            "url": ev.url or "",
        }
        for ev in events
    ]

    system_prompt = """You are a compliance gatekeeper for business-opportunity detection.

Apply the client gate criteria.

HARD EXCLUSIONS (auto-fail):
- price moves, market recaps, opinion/editorial, macro narrative
- pure funding/investment/valuation news WITHOUT an external action surface
- institutional portfolio/ETF rebalancing commentary WITHOUT an external action surface

MINIMUM EVIDENCE TO PASS (must have >=1 and include a verbatim snippet):
- partnership_or_integration: explicit named partnership OR explicit integration (named parties, integration announced, partnership program)
- grant_or_rfp_or_program_open: explicit applications open / RFP/RFQ/RFI / call for proposals / grant program with external participants
- vendor_or_procurement_need: explicit tender/procurement/vendor search/RFP for vendors, or clear buying intent (not investment)
- token_launch_or_tge: explicit token launch / TGE / token generation event / launch date or signup/waitlist for the launch
- regulatory_or_compliance_update: explicit license/approval/registration/compliance change that creates an actionable BD motion
  (ONLY if there is an external action surface: application, partner onboarding, vendor intake, integration path, or contact channel)
- institutional_allocation_or_treasury: explicit corporate/treasury/institutional allocation decision that creates an actionable BD motion
  (ONLY if there is an external action surface: vendor selection, platform onboarding, integration, partner channel, RFP)
- support_added_with_bd_action: new support/feature added AND an explicit BD action surface (partners/integrators/vendors encouraged)

IMPORTANT DISAMBIGUATION (reduce 'grant' inflation):
- Venture funding rounds, fund closes, M&A rumors, â€œraises $Xâ€, â€œSeries A/Bâ€, â€œlaunches fundâ€ are NOT grants/RFP.
  These should be hard_exclusion=true unless the text includes an explicit external program/RFP/intake.

For each event:
- Set hard_exclusion=true if the event is primarily exclusion content.
- Choose evidence_type from the allowed enums; choose 'none' if no qualifying evidence.
- Provide evidence_snippet: a SHORT verbatim excerpt (1-2 sentences) from summary/description proving the evidence.
  If you cannot find a verbatim excerpt, set evidence_type='none' and evidence_snippet=null.

Output ONLY JSON per schema."""

    user_prompt = f"""EVENTS:
{json.dumps(events_payload, default=_json_safe, separators=(",", ":"))}

Return gatekeeper results for every event_id."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": openai_gatekeeper_schema()},
        temperature=0,
        max_completion_tokens=max_tokens,
    )

    if tracker:
        tracker.add_usage(resp.usage, model)

    raw = resp.choices[0].message.content
    
    # Check if response was truncated
    if resp.usage and hasattr(resp.usage, 'completion_tokens'):
        if resp.usage.completion_tokens >= max_tokens * 0.95:
            print(f"  WARNING: Gatekeeper response likely truncated ({resp.usage.completion_tokens}/{max_tokens} tokens)")
    
    parsed = json.loads(raw)

    out: Dict[str, GateResult] = {}
    for r in parsed.get("results", []):
        gr = GateResult(
            event_id=r["event_id"],
            hard_exclusion=bool(r["hard_exclusion"]),
            hard_exclusion_reason=r.get("hard_exclusion_reason"),
            evidence_type=r.get("evidence_type") or "none",
            evidence_snippet=enforce_verbatim_snippet(r.get("evidence_snippet")),
        )
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
        events_payload.append(
            {
                **ev.to_dict(),
                "gate": {
                    "evidence_type": gr.evidence_type,
                    "evidence_snippet": gr.evidence_snippet,
                },
            }
        )

    system_prompt = f"""You are a business development analyst generating *client-gate compliant* opportunities.

NON-NEGOTIABLE RULES:
1) You are ONLY given events that already passed Gatekeeper.
2) Your output MUST follow the required schema fields.
3) evidence_snippet MUST be verbatim (use the provided gate.evidence_snippet; do not invent).
   - Keep evidence_snippet SHORT (â‰ˆ10â€“20 words) and directly supportive of the claim.

3.5) evidence_type MUST match gate.evidence_type for the supporting event(s); do not invent new evidence types.
4) recommended_action is ONLY allowed when is_opportunity=true and must be grounded in evidence_snippet.
5) NO hard-exclusion content. If you suspect an event is actually exclusion content, set is_opportunity=false.
6) Categories MUST align with the provided definitions; do not guess.

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
        "reason",
        "evidence_type",
        "evidence_snippet",
        "recommended_action",
        "target_company",
        "who_to_contact",
        "categories",
        "confidence",
        "source_event_ids",
        "sub_scores",
    ]
    short = [{k: o.get(k) for k in keys} for o in opportunities]

    system_prompt = f"""You are a compliance critic for business opportunities.

Your job is to enforce the client rules:
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
- If the item looks like hard exclusion content (price moves / macro / opinion / market recap) â†’ DISCARD.
- If is_opportunity=true but the minimal evidence type is not genuinely present â†’ DISCARD.

For each opportunity:
- status: keep | reframe | discard
- feedback: short, specific
- corrected_sub_scores: if sub-scores are inflated/deflated
- reframed_opportunity: if status==reframe, provide a fully corrected version that follows schema.

Be strict: prefer DISCARD over weak REFRAME.

REALISM / REACHABILITY HARD RULE (PILOT-CRITICAL):
- If you cannot plausibly imagine this opportunity receiving a reply from a
  partnerships / business development inbox within ~{TIME_HORIZON_WEEKS} weeks â†’ DISCARD.

TIME HORIZON (NON-NEGOTIABLE):
- The opportunity must be actionable within {TIME_HORIZON_STR}.
- If the best realistic next step is likely >{TIME_HORIZON_WEEKS} weeks away, or timing is speculative/unclear â†’ DISCARD.

This includes (must discard):
- Pitching mega-institutions or regulators without an explicit partner intake,
  program application, RFP/procurement, or named integration already announced.
- "Strategic" or "thought leadership" angles with no external action surface.
- PR announcements that do not expose a concrete integration, vendor, or partner motion.
- SECONDARY INFERENCE opportunities: items inferred from a deal, expansion, funding,
  or partnership between other parties that do NOT themselves expose a new external
  action surface for the target company (e.g., â€œCompany A expands after AMD dealâ€).

FINAL GROUNDING REQUIREMENT (NON-NEGOTIABLE):
- The recommended_action MUST be explicitly justified by a specific phrase or sentence
  in the evidence_snippet. If you cannot point to the exact words in the snippet that
  justify the outreach â†’ DISCARD.

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
                },
            }
        )

    system_prompt = f"""You are the REFINER agent. Produce a final, client-gate compliant opportunity set.

NON-NEGOTIABLE CONSTRAINTS (prevents inference regression):
1) You may ONLY operate on the ORIGINAL OPPORTUNITIES provided.
   - You can DISCARD or REFRAME them, but DO NOT introduce new opportunities.
2) You must NOT add new companies, new partnerships, new products, or new claims.
   If a claim is not supported by the provided evidence_snippet, remove it.
3) evidence_snippet MUST be verbatim and MUST be taken exactly from gate evidence.
4) recommended_action is ONLY allowed when is_opportunity=true AND must be explicitly justified by the exact words in evidence_snippet.
5) TIME HORIZON: actionable within {TIME_HORIZON_STR} only.
6) Prefer DISCARD over weak REFRAME.

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
        tracker.add_usage(resp.usage, model)

    raw = resp.choices[0].message.content
    parsed = json.loads(raw)
    return parsed.get("opportunities", [])



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
        ev_type = extract_evidence_type_from_reason(o.reason)
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
                    title=merged.title,
                    summary=merged.summary,
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

    for o in opps:
        primary_cat = (o.categories or [""])[0]
        cat_norm = normalize_key(primary_cat)
        ev_type = extract_evidence_type_from_reason(o.reason)
        ev_snip = extract_evidence_snippet_from_reason(o.reason)

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

def build_reason_block(reason: str, evidence_type: str, evidence_snippet: str) -> str:
    return (
        f"Evidence type: {evidence_type}\n"
        f"Evidence snippet: {evidence_snippet}\n"
        f"Reason: {reason}"
    )


# ---------------------------
# Main Processing
# ---------------------------

def process_events_batch(
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

    for batch_num, batch_events in enumerate(intelligent_batches, 1):
        print(f"[OpportunityMatcher] Batch {batch_num}/{len(intelligent_batches)} ({len(batch_events)} events)...")

        # 1) Gatekeeper
        try:
            gate_map = call_gatekeeper(client, gatekeeper_model, batch_events, tracker)
        except Exception as e:
            print(f"  WARNING: Gatekeeper failed: {e}")
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
                "passes_gate": gate_result.passes(),
            })

        gated: List[Tuple[Event, GateResult]] = []
        excluded = 0
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
            if gr.passes():
                gated.append((ev, gr))
            else:
                excluded += 1
                rejection_audit.append({
                    "stage": "gatekeeper",
                    "event_id": ev.event_id,
                    "event_title": ev.title or "",
                    "decision": "excluded",
                    "reason": gr.hard_exclusion_reason if gr.hard_exclusion else ("no_qualifying_evidence" if gr.evidence_type == "none" else "no_verbatim_evidence"),
                    "hard_exclusion": gr.hard_exclusion,
                    "evidence_type": gr.evidence_type,
                })

        print(f"  [Gatekeeper] Passed: {len(gated)} | Excluded: {excluded}")
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
        rating = float(feedback.get("overall_rating", 0))
        print(f"rated {rating}/10")

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
            reason_block = build_reason_block(ro.get("reason", ""), evidence_type, evidence_snippet)

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
        print("[OpportunityMatcher] Running Filter Agent...")

        event_lookup = {ev.event_id: ev for ev in events}

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
    parser.add_argument("--refiner-model", default=DEFAULT_REFINER_MODEL, help="Model for Refiner")
    parser.add_argument("--days-back", type=int, default=30, help="Days of events to analyze")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Optional JSONL input path. Default: Final_events.jsonl in the Data folder.",
    )
    parser.add_argument("--max-events-per-batch", type=int, default=20, help="Max events per batch")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Max output tokens")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE, help="Minimum confidence threshold")
    parser.add_argument(
        "--time-horizon-weeks",
        type=int,
        default=12,
        choices=[2, 4, 10, 12, 20, 25, 30],
        help="Time horizon (in weeks) enforced across ALL agent prompts (1â€“N weeks). Allowed: 2, 4, 10, 12, 20, 25, 30.",
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
        f"[OpportunityMatcher] Models: Finder={args.finder_model}, Gatekeeper={args.gatekeeper_model}, Critic={args.critic_model}, Refiner={args.refiner_model}, Filter={getattr(args, 'filter_model', DEFAULT_FILTER_MODEL)}"
    )

    if load_dotenv:
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[OpportunityMatcher] ERROR: OPENAI_API_KEY not set")
        return 1
    print(f"[OpportunityMatcher] OPENAI_API_KEY tail: ...{api_key[-5:]}")

    client = OpenAI()

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

