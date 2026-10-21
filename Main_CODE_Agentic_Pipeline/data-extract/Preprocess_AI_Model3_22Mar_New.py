#!/usr/bin/env python3
"""
Preprocess_AI.py

Purpose
-------
Single end-to-end preparation stage that merges:
  1) Preprocess.py      (clean/normalize RSS events, build AI-ready text fields, contact extraction)
  2) AI_pipeline.py     (single OpenAI call per event for intelligent summary + enrichment)

Context / current usage
-----------------------
- Opportunity qualification & categorization are handled by a separate multi-agent program ("Opportunity Matcher").
- This script therefore does NOT attempt to decide opportunity vs non-opportunity, and does NOT categorize.
- It enriches *all* events with:
    - ai_summary (3-5 sentence summary)
    - ai_enrichment (event_type, entities, key_points, why_it_matters, contact_leads, etc.)

Defaults (no CLI required)
-------------------------
Input : ./Data/events.json
Output: ./Data/Final_events.jsonl

Key behaviors (kept consistent with current pipeline)
----------------------------------------------------
- RSS Collector collects everything; this stage filters:
    - Drop events outside a date range (default: last 7 days) based on published_at
    - Apply a conservative "noise" filter (default ON; lenient)
- Clean HTML while preserving paragraph structure for downstream models.
- Keep a long cleaned body_text for downstream processing (default up to 15000 chars).
- Run best-effort contact extraction (from RSS fields, and optionally fetch pages).
- Run a single OpenAI call per event to produce summary + enrichment.

Dependencies
------------
pip install beautifulsoup4 lxml python-dateutil bleach requests openai

Env:
  OPENAI_API_KEY must be set unless --dry-run

Notes
-----
- This file is intentionally self-contained (single script).
- CLI switches exist for debugging, but defaults are set so typical usage is:
    python Preprocess_AI.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html as _html
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

try:
    import bleach
except Exception:  # pragma: no cover
    bleach = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # type: ignore


# =============================================================================
# Defaults / paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data"

DEFAULT_INPUT = DATA_DIR / "events.json"
DEFAULT_CLEAN_OUTPUT = DATA_DIR / "events_clean.jsonl"
# Legacy intermediate output (no longer written; kept for backward-compat CLI args)
DEFAULT_OUTPUT = DATA_DIR / "events_ai_enriched.jsonl"
DEFAULT_FILTERED_OUTPUT = DATA_DIR / "events_filtered.jsonl"
DEFAULT_FINAL_OUTPUT = DATA_DIR / "Final_events.jsonl"
DEFAULT_FILTERED_DEBUG_OUTPUT = DATA_DIR / "events_filtered_out_debug.jsonl"

# Default window for date-range filtering if no explicit range is provided.
# (Per latest requirement: last 7 days)
DEFAULT_MAX_AGE_DAYS = 7

# Keep detailed texts for downstream (up to 15000 chars for cleaned body)
DEFAULT_MAX_BODY_CHARS = 15000

# AI request text cap (controls OpenAI cost). Uses title/summary/body + contact hints.
DEFAULT_AI_MAX_EVENT_CHARS = 6000

#DEFAULT_MODEL = "gpt-4.1-mini"    # OLD Model
DEFAULT_MODEL = "gpt-5.1"
DEFAULT_MAX_OUTPUT_TOKENS_AI = 800

# AI summary max chars (configurable; default 500)
DEFAULT_AI_SUMMARY_MAX_CHARS = 500
DEFAULT_AI_BATCH_SIZE = 40
DEFAULT_AI_MAX_WORKERS = 3


# =============================================================================
# Token tracking (from AI_pipeline.py)
# =============================================================================

class TokenTracker:
    """Track OpenAI token usage and estimated costs."""

    def __init__(self):
        self.model_usage: Dict[str, Dict[str, Any]] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_estimated_cost = 0.0
        self._lock = threading.Lock()

    def get_pricing(self, model: str):
        """Return (input_cost_per_1m, output_cost_per_1m) in USD per 1M tokens."""
        pricing = {
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
            "gpt-4.1": (2.5, 10.0),
            "gpt-4.1-mini": (0.15, 0.6),
            "o3": (2.0, 8.0),
            "o3-mini": (1.1, 4.4),
            "gpt-5.1": (1.25, 10.0),
            "gpt-5.2": (1.25, 12.5),
            "gpt-5-mini": (0.25, 2.0),
        }
        return pricing.get(model)

    def _ensure_model(self, model: str):
        if model not in self.model_usage:
            self.model_usage[model] = {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0}

    def add_usage(self, usage, model: str):
        if not usage:
            return

        with self._lock:
            pricing = self.get_pricing(model)
            # If unknown pricing, track tokens but don't estimate cost.
            if not pricing:
                self._ensure_model(model)
                input_tokens = getattr(usage, "prompt_tokens", 0)
                output_tokens = getattr(usage, "completion_tokens", 0)
                self.model_usage[model]["input_tokens"] += input_tokens
                self.model_usage[model]["output_tokens"] += output_tokens
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                return

            self._ensure_model(model)

            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            in_cost, out_cost = pricing
            cost = (input_tokens / 1_000_000) * in_cost + (output_tokens / 1_000_000) * out_cost

            self.model_usage[model]["input_tokens"] += input_tokens
            self.model_usage[model]["output_tokens"] += output_tokens
            self.model_usage[model]["estimated_cost"] += cost

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_estimated_cost += cost

    def print_summary(self):
        lines: List[str] = []
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"{_now_iso()} [Preprocess_AI] MODEL-WISE USAGE SUMMARY")
        lines.append("=" * 72)
        for model, data in self.model_usage.items():
            total_tokens = data["input_tokens"] + data["output_tokens"]
            lines.append("")
            lines.append(f"Model: {model}")
            lines.append("-" * 72)
            lines.append(f"  Input tokens:   {data['input_tokens']:,}")
            lines.append(f"  Output tokens:  {data['output_tokens']:,}")
            lines.append(f"  Total tokens:   {total_tokens:,}")
            if "estimated_cost" in data:
                lines.append(f"  Estimated cost: ${data['estimated_cost']:.4f}")
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"{_now_iso()} [Preprocess_AI] TOTAL USAGE SUMMARY")
        lines.append("=" * 72)
        total_tokens = self.total_input_tokens + self.total_output_tokens
        lines.append(f"  Total input tokens:   {self.total_input_tokens:,}")
        lines.append(f"  Total output tokens:  {self.total_output_tokens:,}")
        lines.append(f"  Grand total tokens:   {total_tokens:,}")
        lines.append(f"  Total estimated cost: ${self.total_estimated_cost:.4f}")
        lines.append("=" * 72)
        print("\n".join(lines))


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


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _now_hms_utc() -> str:
    return time.strftime("T%H:%M:%SZ", time.gmtime())


def log_info(message: str) -> None:
    print(f"{_now_iso()} {message}")


def _is_insufficient_quota(err: Exception) -> bool:
    msg = str(err)
    return (
        "insufficient_quota" in msg
        or "You exceeded your current quota" in msg
        or "check your plan and billing details" in msg
    )


def safe_truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def smart_truncate(text: str, max_chars: int) -> str:
    """Truncate on a word boundary (best-effort) to <= max_chars."""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer last whitespace/punctuation boundary in the window
    m = re.search(r"[\s\,;:\-]\S*$", cut)
    if m and m.start() > max(0, max_chars - 40):
        cut = cut[: m.start()]
    cut = cut.rstrip()
    if not cut:
        return text[: max_chars - 1] + "..."
    return cut + "..."


_SENT_SPLIT_RE = re.compile(r"(?<=[\.!\?])\s+")


def finalize_ai_summary(raw_summary: str, enrichment: Dict[str, Any], max_chars: int) -> str:
    """Ensure a 3-5 sentence summary with minimum detail and length cap."""
    raw = (raw_summary or "").strip()
    if not raw:
        return "No AI Response"

    raw = _WS_RE.sub(" ", raw).strip()
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(raw) if s.strip()]
    if not sentences:
        return "No AI Response"

    def _normalize_sentence(text: str) -> str:
        text = _WS_RE.sub(" ", str(text or "")).strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _word_count(parts: List[str]) -> int:
        return sum(len(re.findall(r"\b\w+\b", p)) for p in parts)

    sentences = [_normalize_sentence(s) for s in sentences if _normalize_sentence(s)]

    extras: List[str] = []
    for source in (
        enrichment.get("why_it_matters") or [],
        enrichment.get("key_points") or [],
    ):
        if isinstance(source, list):
            for item in source:
                normalized = _normalize_sentence(str(item))
                if normalized:
                    extras.append(normalized)

    recommended_action = _normalize_sentence(str(enrichment.get("recommended_action") or ""))
    if recommended_action:
        extras.append(recommended_action)

    for extra in extras:
        if len(sentences) >= 5:
            break
        if extra not in sentences:
            sentences.append(extra)

    while len(sentences) < 3 and extras:
        extra = extras.pop(0)
        if extra not in sentences:
            sentences.append(extra)

    # Keep at most 5 sentences
    sentences = sentences[:5]

    # Re-join
    out = " ".join(sentences).strip()
    out = _WS_RE.sub(" ", out).strip()

    # Enforce max_chars
    if max_chars and len(out) > max_chars:
        out = smart_truncate(out, max_chars)

    trimmed_sentences = [s.strip() for s in _SENT_SPLIT_RE.split(out) if s.strip()]
    if len(trimmed_sentences) > 5:
        trimmed_sentences = trimmed_sentences[:5]
        out = " ".join(trimmed_sentences).strip()
        out = _WS_RE.sub(" ", out).strip()

    if _word_count(trimmed_sentences or [out]) < 35:
        for extra in extras:
            candidate_sentences = trimmed_sentences + [extra]
            candidate = " ".join(candidate_sentences).strip()
            candidate = _WS_RE.sub(" ", candidate).strip()
            if max_chars and len(candidate) > max_chars:
                continue
            trimmed_sentences = candidate_sentences[:5]
            out = candidate
            if _word_count(trimmed_sentences) >= 35:
                break

    return out


# =============================================================================
# Cleaner helpers (from Preprocess.py)
# =============================================================================

_BOILERPLATE_PHRASES = [
    "subscribe",
    "sign up",
    "newsletter",
    "cookie",
    "privacy policy",
    "terms of service",
    "read more",
    "click here",
    "advertisement",
    "sponsored",
    "all rights reserved",
]

_WS_RE = re.compile(r"\s+")

_BD_SURFACE_TYPES = {
    "live_api_dashboard_surface",
    "live_security_rail",
    "named_integration_exploration",
    "live_adoption_surface",
    "none",
}

_LIVE_API_DASHBOARD_RE = re.compile(
    r"\b(live demo|demo api|rest api|openapi|swagger|governance data|governance data pipeline|"
    r"data pipeline|deterministic triage|delegate fatigue index|api docs|/docs\b|tagged release|"
    r"open[ -]source|repository|runnable by third parties|open to collaboration)\b",
    re.IGNORECASE,
)
_LIVE_SECURITY_RAIL_RE = re.compile(
    r"\b(address poisoning|scam address|lookalike addresses|scam detection|screening feature|"
    r"wallet security|destination address check|noncustodial wallet|automatic address poisoning protection)\b",
    re.IGNORECASE,
)
_NAMED_INTEGRATION_RE = re.compile(
    r"\b(exploring whether .* integration makes sense|integration makes sense|looking for .* ecosystem feedback|"
    r"would love feedback from anyone building|named integration|last mile borrower access)\b",
    re.IGNORECASE,
)
_LIVE_ADOPTION_RE = re.compile(
    r"\b(dashboard|stake now|connect your wallet|claim rewards?|staking positions appear|"
    r"supported chains?|more chains on the way|delegator dashboard|app\.[a-z0-9.-]+)\b",
    re.IGNORECASE,
)
_IMMEDIATE_RE = re.compile(r"\b(deadline|apply by|closes? on|closing soon|due by|last day|before [A-Z][a-z]+ \d{1,2})\b", re.IGNORECASE)
_GENERIC_FEEDBACK_RE = re.compile(r"\b(provide feedback|would love feedback|looking forward to your feedback|community feedback)\b", re.IGNORECASE)


def _normalize_sentence_text(text: str) -> str:
    text = _WS_RE.sub(" ", str(text or "")).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _infer_bd_surface_type(event_text: str, recommended_action: str = "") -> str:
    haystack = _WS_RE.sub(" ", " ".join([event_text or "", recommended_action or ""])).strip()
    if not haystack:
        return "none"
    if _LIVE_SECURITY_RAIL_RE.search(haystack):
        return "live_security_rail"
    if _NAMED_INTEGRATION_RE.search(haystack) and re.search(r"\b(celo|arbitrum|base|solana|ethereum|polygon|optimism)\b", haystack, re.IGNORECASE):
        return "named_integration_exploration"
    if _LIVE_ADOPTION_RE.search(haystack):
        return "live_adoption_surface"
    if _LIVE_API_DASHBOARD_RE.search(haystack):
        return "live_api_dashboard_surface"
    return "none"


def _extract_best_bd_action_detail(event_text: str, fallback: Optional[str] = None, max_chars: int = 260) -> Optional[str]:
    text = _WS_RE.sub(" ", str(event_text or "")).strip()
    if not text:
        return smart_truncate(str(fallback).strip(), max_chars) if fallback else None

    candidates = [seg.strip() for seg in re.split(r"(?<=[\.\!\?])\s+|\n+", text) if seg and seg.strip()]
    best_sent = ""
    best_score = -999

    for sent in candidates:
        score = 0
        lowered = sent.lower()
        if _LIVE_SECURITY_RAIL_RE.search(sent):
            score += 8
        if _NAMED_INTEGRATION_RE.search(sent):
            score += 7
        if _LIVE_ADOPTION_RE.search(sent):
            score += 6
        if _LIVE_API_DASHBOARD_RE.search(sent):
            score += 6
        if re.search(r"\b(integrat(?:e|ion)|api|dashboard|stake now|connect your wallet|open to collaboration|live demo|docs)\b", sent, re.IGNORECASE):
            score += 3
        if _GENERIC_FEEDBACK_RE.search(sent):
            score -= 3
        if "financial advice" in lowered or "do your own research" in lowered:
            score -= 5
        if len(sent) < 24:
            score -= 1
        if score > best_score:
            best_score = score
            best_sent = sent

    chosen = best_sent if best_score > 0 else (fallback or "")
    chosen = _normalize_sentence_text(chosen)
    return smart_truncate(chosen, max_chars) if chosen else None


def _recommended_action_for_surface(surface_type: str, current_action: str) -> str:
    current_action = (current_action or "").strip()
    if current_action and not current_action.lower().startswith("monitor - no immediate action surface identified"):
        return current_action
    if surface_type == "live_api_dashboard_surface":
        return "Reach out to the maintainer/project owner to evaluate integrating the live API or dashboard surface and discuss implementation support."
    if surface_type == "live_security_rail":
        return "Reach out to the wallet product, security, or partnerships team to propose complementary threat-intel, scam-detection, or user-protection integrations."
    if surface_type == "named_integration_exploration":
        return "Reply in-thread with a concrete integration proposal for the named chain/use case and suggest a short follow-up call."
    if surface_type == "live_adoption_surface":
        return "Reach out to propose chain or token support, wallet integrations, or adoption help for the live dashboard surface."
    return current_action or "Monitor - no immediate action surface identified."


def reconcile_preprocess_enrichment(event_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return data

    recommended_action = str(data.get("recommended_action", "") or "").strip()
    if recommended_action:
        recommended_action = smart_truncate(recommended_action, 240)
    bd_signal = data.get("bd_signal") if isinstance(data.get("bd_signal"), dict) else {}

    surface_type = str(bd_signal.get("surface_type", "") or "").strip()
    if surface_type not in _BD_SURFACE_TYPES:
        surface_type = _infer_bd_surface_type(event_text, recommended_action)

    action_type = str(bd_signal.get("action_type", "") or "").strip()
    if surface_type == "named_integration_exploration" and action_type in {"", "none"}:
        action_type = "partner_intake"
    elif surface_type in {"live_api_dashboard_surface", "live_security_rail", "live_adoption_surface"} and action_type in {"", "none"}:
        action_type = "integration_announced"
    elif action_type not in {
        "partner_intake",
        "rfp_or_grant",
        "integration_announced",
        "vendor_search",
        "token_launch",
        "regulatory_opening",
        "none",
    }:
        action_type = "none"

    action_detail = _extract_best_bd_action_detail(event_text, bd_signal.get("action_detail"))
    if action_detail:
        action_detail = smart_truncate(str(action_detail).strip(), 260)
    has_action_surface = bool(bd_signal.get("has_action_surface", False))

    if surface_type != "none":
        has_action_surface = True
        recommended_action = _recommended_action_for_surface(surface_type, recommended_action)
        urgency = str(bd_signal.get("urgency", "") or "").strip()
        if urgency not in {"immediate", "near_term", "speculative", "none"} or urgency == "none":
            urgency = "immediate" if _IMMEDIATE_RE.search(event_text or "") else "near_term"
    else:
        urgency = str(bd_signal.get("urgency", "") or "").strip()
        if urgency not in {"immediate", "near_term", "speculative", "none"}:
            urgency = "none"

    data["recommended_action"] = (smart_truncate(recommended_action, 240) if recommended_action else "") or "Monitor - no immediate action surface identified."
    data["bd_signal"] = {
        "has_action_surface": has_action_surface if surface_type != "none" else bool(bd_signal.get("has_action_surface", False)),
        "action_type": action_type if surface_type != "none" else (action_type or "none"),
        "action_detail": action_detail,
        "urgency": urgency if surface_type != "none" else (urgency or "none"),
        "surface_type": surface_type,
    }
    return data


# Conservative, lenient noise filter: only drops obvious TA/prediction items.
NOISE_TITLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bprice\s+prediction\b", re.IGNORECASE),
    re.compile(r"\btechnical\s+analysis\b", re.IGNORECASE),
    re.compile(r"\bchart\s+analysis\b", re.IGNORECASE),
    re.compile(r"\btrading\s+signals\b", re.IGNORECASE),
    re.compile(r"\b(?:buy|sell)\s+signal\b", re.IGNORECASE),
]

GITHUB_RELEASE_NOISE_PATTERNS: List[re.Pattern] = [
    re.compile(r"_ci\b", re.IGNORECASE),
    re.compile(r"-rc\.", re.IGNORECASE),
    # Starts with semver-like tag (e.g., v1.4.2, v0.9.1, v2.3.1-hotfix)
    re.compile(r"^\s*v\d+\.\d+(?:\.\d+)?(?:[-+][\w\.-]+)?", re.IGNORECASE),
    # @v + version and package@version tags (e.g., wagmi@v2.14.3, viem@2.1.0)
    re.compile(r"@v\d+(?:\.\d+){1,3}\b", re.IGNORECASE),
    re.compile(r"\b[a-z0-9_.-]+@v?\d+(?:\.\d+){1,3}\b", re.IGNORECASE),
    # Explicit maintenance wording
    re.compile(r"\bpatch\s+release\b", re.IGNORECASE),
    re.compile(r"\bhotfix\b", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:viem@|wagmi@|log/v)", re.IGNORECASE),
]

# Keep-override rules for GitHub release titles (take precedence over drop patterns).
GITHUB_RELEASE_KEEP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bmainnet\b", re.IGNORECASE),
    re.compile(r"\bintegration\b", re.IGNORECASE),
    re.compile(r"\bpartnership\b", re.IGNORECASE),
    re.compile(r"\blaunch\b", re.IGNORECASE),
]

# "Named protocol upgrade" safeguard (e.g., "Ethereum Pectra upgrade").
NAMED_PROTOCOL_UPGRADE_RE = re.compile(
    r"\b("
    r"ethereum|bitcoin|solana|arbitrum|optimism|polygon|starknet|zksync|avalanche|"
    r"sui|aptos|cosmos|base|bnb|tron|near|monad|osmosis|wormhole|chainlink|"
    r"uniswap|aave|maker|lido|dydx|compound|pectra|dencun|cancun|shanghai"
    r")\b.*\bupgrade\b|\bupgrade\b.*\b("
    r"ethereum|bitcoin|solana|arbitrum|optimism|polygon|starknet|zksync|avalanche|"
    r"sui|aptos|cosmos|base|bnb|tron|near|monad|osmosis|wormhole|chainlink|"
    r"uniswap|aave|maker|lido|dydx|compound|pectra|dencun|cancun|shanghai"
    r")\b",
    re.IGNORECASE,
)

LOW_SIGNAL_DOMAINS = {
    "bitcoinist.com",
    "newsbtc.com",
    "cryptopotato.com",
    "cryptonews.com",
}


def _noise_score(text: str) -> int:
    if not text:
        return 0
    return sum(1 for pat in NOISE_TITLE_PATTERNS if pat.search(text))


def get_domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def is_obvious_noise_item(title_text: str, summary_text: str, link: str) -> Tuple[bool, Optional[str]]:
    t = (title_text or "").strip()
    s = (summary_text or "").strip()
    dom = get_domain(link or "")

    score_t = _noise_score(t)
    score_s = _noise_score(s)

    if score_t >= 1:
        return True, "noise:title_market_ta_or_roundup"

    if dom in LOW_SIGNAL_DOMAINS and (score_t + score_s) >= 1:
        return True, "noise:low_signal_domain_market_ta"

    # Upstream GitHub release/CI noise trimming (assessment recommendation).
    # Keep true product announcements in; drop obvious CI/RC/version-bump items.
    if "github.com" in dom:
        # Keep-override takes precedence if title clearly signals action-relevant release context.
        keep_override = any(p.search(t) for p in GITHUB_RELEASE_KEEP_PATTERNS) or bool(NAMED_PROTOCOL_UPGRADE_RE.search(t))
        if (not keep_override) and any(p.search(t) for p in GITHUB_RELEASE_NOISE_PATTERNS):
            return True, "noise:github_ci_rc_version_bump"
    return False, None


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_date_arg(value: Optional[str]) -> Optional[_dt.date]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if not _DATE_ONLY_RE.match(s):
        raise ValueError("Invalid date format. Use YYYY-MM-DD (no time).")
    dt = dateparser.parse(s)
    if dt is None:
        raise ValueError(f"Invalid date: {value}")
    return dt.date()


def is_outside_date_range(
    published_iso: str,
    from_dt: Optional[_dt.date],
    to_dt: Optional[_dt.date],
) -> Tuple[bool, Optional[str]]:
    try:
        dt = dateparser.parse(published_iso)
        if dt is None:
            return False, None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        dt = dt.astimezone(_dt.timezone.utc)
        published_date = dt.date()
        if from_dt and published_date < from_dt:
            return True, f"date_before_from:{from_dt.isoformat()}"
        if to_dt and published_date > to_dt:
            return True, f"date_after_to:{to_dt.isoformat()}"
    except Exception:
        return False, None
    return False, None


def html_to_text(html: str) -> str:
    """Convert HTML to mostly-plain text while preserving paragraph structure."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "header", "aside", "figure", "img"]):
        tag.decompose()

    for link in soup.find_all("a"):
        href = (link.get("href", "") or "").strip()
        text = link.get_text(strip=True)
        if href and text:
            link.replace_with(f"{text} ({href})")
        elif text:
            link.replace_with(text)

    for br in soup.find_all("br"):
        br.replace_with("\n")

    blocks = soup.find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"])
    paras: List[str] = []
    if blocks:
        for b in blocks:
            t = b.get_text(separator=" ", strip=True)
            if t:
                paras.append(t)
    else:
        raw = soup.get_text(separator="\n")
        paras = [p.strip() for p in re.split(r"\n{2,}", raw) if p and p.strip()]

    cleaned_paras: List[str] = []
    for p in paras:
        p2 = _WS_RE.sub(" ", p).strip()
        if p2:
            cleaned_paras.append(p2)

    text = "\n\n".join(cleaned_paras).strip()
    return text


def normalize_text(text: str, max_chars: int = DEFAULT_MAX_BODY_CHARS) -> str:
    if not text:
        return ""
    raw = str(text)
    if "\n" in raw:
        paras = [p.strip() for p in re.split(r"\n{2,}", raw) if p and p.strip()]
        paras = [_WS_RE.sub(" ", p).strip() for p in paras if p and p.strip()]
        t = "\n\n".join(paras).strip()
    else:
        t = _WS_RE.sub(" ", raw).strip()

    low = t.lower()
    for phrase in _BOILERPLATE_PHRASES:
        idx = low.find(phrase)
        if idx != -1 and idx > 300:
            t = t[:idx].strip()
            low = t.lower()

    if len(t) > max_chars:
        t = t[:max_chars].rstrip()
        if len(t) > 50 and not t.endswith((" ", "\n")):
            t = t.rsplit(" ", 1)[0].rstrip()
    return t


def best_effort_parse_datetime(value: Any) -> Tuple[Optional[str], Optional[str]]:
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        try:
            dt = _dt.datetime.utcfromtimestamp(float(value)).replace(tzinfo=_dt.timezone.utc)
            return dt.isoformat(), "epoch"
        except Exception:
            return None, "unparsed"

    s = str(value).strip()
    if not s:
        return None, None

    try:
        dt = dateparser.parse(s)
        if dt is None:
            return None, "unparsed"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.astimezone(_dt.timezone.utc).isoformat(), "rss"
    except Exception:
        return None, "unparsed"


def stable_event_id(source: str, guid: str, link: str, published_iso: str, title: str) -> str:
    source = (source or "").strip().lower()
    guid = (guid or "").strip()
    link = (link or "").strip()
    published_iso = (published_iso or "").strip()
    title = (title or "").strip()
    basis = "|".join([source, guid or link or "", published_iso or "", title[:200]])
    return hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()[:24]


def extract_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    title_raw = item.get("title_raw") or item.get("title") or ""
    description_raw = (
        item.get("description_raw")
        or item.get("summary_raw")
        or item.get("description")
        or item.get("summary")
        or ""
    )
    content_raw = item.get("content_raw") or item.get("content") or item.get("content:encoded") or ""

    published_any = (
        item.get("published_at")
        or item.get("published")
        or item.get("pubDate")
        or item.get("date")
        or item.get("updated")
    )

    source = item.get("source") or item.get("source_name") or item.get("feed") or item.get("publisher") or ""
    link = item.get("link") or item.get("url") or item.get("source_link") or ""
    guid = item.get("guid") or item.get("id") or item.get("item_id") or ""

    return {
        "title_raw": title_raw,
        "description_raw": description_raw,
        "content_raw": content_raw,
        "published_any": published_any,
        "source": source,
        "link": link,
        "guid": guid,
    }


def build_categorization_text(title_text: str, summary_text: str, body_text: str, max_chars: int) -> str:
    parts = []
    if title_text:
        parts.append(title_text)
    if summary_text:
        parts.append(summary_text)
    composed = "\n".join(parts).strip()
    if len(composed) < max_chars // 2 and body_text:
        composed = (composed + "\n" + body_text).strip()
    if len(composed) > max_chars:
        composed = composed[:max_chars].rstrip()
    return composed


def iter_items_from_file(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        elif isinstance(data, dict):
            items = data.get("items") or data.get("events") or data.get("data")
            if isinstance(items, list):
                for obj in items:
                    if isinstance(obj, dict):
                        yield obj
            else:
                yield data
        else:
            raise ValueError("Unsupported JSON structure in input.")
    else:
        raise ValueError("Input must be .json or .jsonl")


# summary_html sanitization: allow minimal semantic tags
ALLOWED_TAGS = ["p", "br", "b", "strong", "i", "em"]
ALLOWED_ATTRS = {}
ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def _bleach_link_callback(attrs, new=False):
    href = attrs.get((None, "href"), "")
    if href and not (href.startswith("http://") or href.startswith("https://") or href.startswith("mailto:")):
        attrs.pop((None, "href"), None)
    attrs[(None, "rel")] = "nofollow noopener noreferrer"
    attrs[(None, "target")] = "_blank"
    return attrs


def sanitize_summary_html(raw_html: str, max_chars: int = 2000) -> str:
    if not raw_html:
        return ""

    if bleach is None:
        text = normalize_text(html_to_text(raw_html), max_chars=max_chars)
        return f"<p>{_html.escape(text)}</p>" if text else ""

    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    cleaned_html = str(soup)

    safe = bleach.clean(
        cleaned_html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        protocols=ALLOWED_PROTOCOLS,
        strip=True,
    )

    safe = bleach.linkify(
        safe,
        callbacks=[_bleach_link_callback],
        skip_tags=["code"],
    )

    safe = safe.strip()
    if len(safe) > max_chars:
        safe = safe[:max_chars].rstrip()

    if safe and not safe.lstrip().startswith("<"):
        safe = f"<p>{safe}</p>"

    return safe


def clean_one_item(
    item: Dict[str, Any],
    now_iso: str,
    max_body_chars: int,
    max_categorization_chars: int,
    max_summary_html_chars: int,
) -> Dict[str, Any]:
    fields = extract_fields(item)

    title_raw = str(fields["title_raw"] or "")
    description_raw = str(fields["description_raw"] or "")
    content_raw = str(fields["content_raw"] or "")
    source = str(fields["source"] or "")
    link = str(fields["link"] or "")
    guid = str(fields["guid"] or "")

    published_iso, published_src = best_effort_parse_datetime(fields["published_any"])
    if not published_iso:
        published_iso = now_iso
        published_src = "fallback_now"

    title_text = normalize_text(html_to_text(title_raw), max_chars=400)
    summary_text = normalize_text(html_to_text(description_raw), max_chars=1200)

    body_text = normalize_text(html_to_text(content_raw), max_chars=max_body_chars)
    if len(body_text) > max_body_chars:
        body_text = body_text[:max_body_chars]

    if not body_text and summary_text and len(summary_text) > 600:
        body_text = summary_text

    categorization_text = build_categorization_text(
        title_text=title_text,
        summary_text=summary_text,
        body_text=body_text,
        max_chars=max_categorization_chars,
    )

    summary_html_source = "description_raw" if description_raw else ("content_raw" if content_raw else "none")
    summary_html = sanitize_summary_html(description_raw or content_raw, max_chars=max_summary_html_chars)

    event_id = item.get("event_id") or stable_event_id(
        source=source,
        guid=guid,
        link=link,
        published_iso=published_iso,
        title=title_text or title_raw,
    )

    cleaned = dict(item)
    cleaned.update(
        {
            "event_id": event_id,
            "source": source,
            "link": link,
            "guid": guid,
            "published_at": published_iso,
            "published_at_source": published_src,
            "title_raw": title_raw,
            "description_raw": description_raw,
            "content_raw": content_raw,
            "title_text": title_text,
            "summary_text": summary_text,   # RSS/description-derived; preserved later as rss_summary_text
            "body_text": body_text,         # cleaned long body
            "summary_html": summary_html,
            "summary_html_source": summary_html_source,
            "categorization_text": categorization_text,
        }
    )
    return cleaned


# =============================================================================
# Contact enrichment (from Preprocess.py)
# =============================================================================

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
SOCIAL_PATTERNS = {
    "x": re.compile(r"https?://(?:www\.)?(?:x\.com|twitter\.com)/[A-Za-z0-9_]{1,30}"),
    "linkedin": re.compile(r"https?://(?:www\.)?linkedin\.com/(?:company|in)/[A-Za-z0-9\-_%/]+"),
    "discord": re.compile(r"https?://(?:www\.)?discord(?:app)?\.com/(?:invite|channels)/[A-Za-z0-9\-_/]+"),
    "telegram": re.compile(r"https?://(?:t\.me|telegram\.me)/[A-Za-z0-9_]+"),
    "github": re.compile(r"https?://(?:www\.)?github\.com/[A-Za-z0-9_.\-]+"),
}

CONTACT_PATH_HINTS = ["/contact", "/contact-us", "/support", "/help", "/press", "/media", "/about", "/team"]
MAX_CONTACT_URLS = 5

CONTACT_URL_ALLOW_RE = re.compile(r"^/(contact|contact-us|contacts?|about|team|press|media)(/)?$", re.IGNORECASE)
CONTACT_URL_DENY_RE = re.compile(r"/press-releases?/[^/]+", re.IGNORECASE)

BAD_EMAIL_PREFIXES = (
    "noreply",
    "no-reply",
    "do-not-reply",
    "donotreply",
    "notifications@",
    "mailer-daemon",
)
BAD_EMAIL_DOMAINS = {"example.com", "email.com", "domain.com"}


def normalize_url(u: str) -> str:
    try:
        p = urlsplit(u)
        return urlunsplit((p.scheme, p.netloc, p.path, "", ""))
    except Exception:
        return u


def same_domain(candidate_url: str, base_domain: str) -> bool:
    d = get_domain(candidate_url)
    if not d or not base_domain:
        return False
    return d == base_domain or d.endswith("." + base_domain)


def is_contact_url_candidate(u: str, base_domain: str) -> bool:
    if not u or not base_domain or not same_domain(u, base_domain):
        return False
    try:
        path = (urlparse(u).path or "").strip()
        if not path:
            return False
        p = path.lower()
        if CONTACT_URL_DENY_RE.search(p):
            return False
        if CONTACT_URL_ALLOW_RE.match(p):
            return True
        if any(h in p for h in CONTACT_PATH_HINTS):
            segments = [s for s in p.split("/") if s]
            if len(segments) <= 2:
                return True
        return False
    except Exception:
        return False


def is_bad_email(email: str) -> bool:
    e = email.strip().lower()
    if not e or "@" not in e:
        return True
    local, _, dom = e.partition("@")
    if dom in BAD_EMAIL_DOMAINS:
        return True
    for p in BAD_EMAIL_PREFIXES:
        if e.startswith(p) or local.startswith(p):
            return True
    return False


def extract_mailto_emails(soup: "BeautifulSoup") -> List[str]:
    emails: List[str] = []
    try:
        for a in soup.find_all("a"):
            href = a.get("href") or ""
            if href.lower().startswith("mailto:"):
                addr = href.split(":", 1)[1].strip()
                addr = addr.split("?", 1)[0].strip()
                if addr:
                    emails.append(addr)
    except Exception:
        pass
    return emails


def extract_jsonld_author_and_emails(html: str) -> Tuple[Optional[str], Optional[str], Set[str]]:
    found_emails: Set[str] = set()
    found_name: Optional[str] = None
    found_url: Optional[str] = None

    def walk(obj):
        nonlocal found_name, found_url
        if isinstance(obj, dict):
            email = obj.get("email")
            if isinstance(email, str) and EMAIL_RE.search(email) and not is_bad_email(email):
                found_emails.add(email.strip())

            otype = obj.get("@type") or obj.get("type")
            t = otype.lower() if isinstance(otype, str) else ""

            if "author" in obj:
                walk(obj["author"])

            if t in ("person", "organization"):
                name = obj.get("name")
                url = obj.get("url")
                if found_name is None and isinstance(name, str) and name.strip():
                    found_name = name.strip()
                if found_url is None and isinstance(url, str) and url.strip():
                    found_url = url.strip()

            if "@graph" in obj:
                walk(obj["@graph"])

            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    try:
        soup = BeautifulSoup(html, "lxml")
        for s in soup.find_all("script"):
            t = (s.get("type") or "").lower()
            if "ld+json" not in t:
                continue
            raw = (s.string or s.text or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                walk(data)
            except Exception:
                continue
    except Exception:
        pass

    return found_name, found_url, found_emails


@dataclass
class FetchConfig:
    fetch_article: bool
    fetch_contact_pages: bool
    timeout: int
    max_html_chars: int
    sleep_s: float


def html_to_text_and_links(html: str, base_url: str) -> Tuple[str, List[str]]:
    if not html:
        return "", []
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    links: List[str] = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        links.append(urljoin(base_url, href))

    text = soup.get_text(separator=" ")
    text = " ".join(text.split()).strip()
    return text, links


def fetch_html(session: requests.Session, url: str, timeout: int, max_chars: int) -> str:
    try:
        resp = session.get(url, timeout=timeout, headers={"User-Agent": "CalyxonAI-PreprocessAI/1.0"})
        if resp.status_code >= 400:
            return ""
        html = resp.text or ""
        if len(html) > max_chars:
            html = html[:max_chars]
        return html
    except Exception:
        return ""


def extract_author_fields(event: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    for k in ("author", "author_name", "creator", "dc_creator"):
        v = event.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip(), None

    raw = event.get("raw_entry") or {}
    if isinstance(raw, dict):
        for k in ("author", "creator", "dc:creator"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip(), None

    authors = event.get("authors")
    if isinstance(authors, list) and authors:
        a0 = authors[0]
        if isinstance(a0, dict):
            name = (a0.get("name") or "").strip() or None
            url = (a0.get("href") or a0.get("url") or "").strip() or None
            return name, url

    return None, None


def pick_contact_page_candidates(links: List[str], base_url: str, base_domain: str) -> List[str]:
    candidates: List[str] = []
    for u in links:
        if not same_domain(u, base_domain):
            continue
        path = (urlparse(u).path or "").lower()
        if any(h in path for h in CONTACT_PATH_HINTS):
            candidates.append(u)

    if not candidates:
        for hint in CONTACT_PATH_HINTS:
            candidates.append(urljoin(base_url, hint))

    seen: Set[str] = set()
    seen_urls: Set[str] = set()
    duplicate_url_count: int = 0
    out: List[str] = []
    for u in candidates:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out[:3]


def confidence_label(has_email: bool, has_contact_url: bool, has_author: bool, fetched: bool) -> str:
    score = 0
    if has_author:
        score += 1
    if has_contact_url:
        score += 2
    if has_email:
        score += 3
    if fetched:
        score += 1
    if score >= 6:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def enrich_contact_one(event: Dict[str, Any], session: requests.Session, cfg: FetchConfig) -> Dict[str, Any]:
    link = str(event.get("link") or "")
    base_domain = get_domain(link)

    author_name, author_url = extract_author_fields(event)

    emails: Set[str] = set()
    contact_urls: Set[str] = set()
    social: Dict[str, Set[str]] = {k: set() for k in SOCIAL_PATTERNS.keys()}
    sources_used: Set[str] = set()

    def process_blob(label: str, blob: str, base_url: str):
        nonlocal author_name, author_url
        if not blob:
            return
        sources_used.add(label)

        for e in EMAIL_RE.findall(blob):
            if is_bad_email(e):
                continue
            dom = e.split("@")[-1].lower()
            strong_context = any(x in label for x in ("contact", "about", "press", "article"))
            if strong_context:
                emails.add(e)
            elif base_domain and (dom == base_domain or dom.endswith("." + base_domain)):
                emails.add(e)

        for sk, pat in SOCIAL_PATTERNS.items():
            for u in pat.findall(blob):
                social[sk].add(u)

        if "<" in blob and ("href" in blob or "mailto:" in blob or "ld+json" in blob):
            try:
                soup = BeautifulSoup(blob, "lxml")

                for me in extract_mailto_emails(soup):
                    if not is_bad_email(me):
                        emails.add(me)

                if "ld+json" in blob:
                    j_name, j_url, j_emails = extract_jsonld_author_and_emails(blob)
                    if j_name and not author_name:
                        author_name = j_name
                    if j_url and not author_url:
                        author_url = j_url
                    for je in j_emails:
                        if not is_bad_email(je):
                            emails.add(je)

                for a in soup.find_all("a"):
                    href = a.get("href")
                    if not href:
                        continue
                    u = urljoin(base_url, href)
                    u_norm = normalize_url(u)
                    if base_domain and is_contact_url_candidate(u_norm, base_domain):
                        if len(contact_urls) < MAX_CONTACT_URLS:
                            contact_urls.add(u_norm)
            except Exception:
                pass

    for k in ("summary_html", "description_raw", "content_raw", "categorization_text"):
        v = event.get(k)
        if isinstance(v, str) and v.strip():
            process_blob(k, v, link)

    fetched_any = False
    article_links: List[str] = []

    if cfg.fetch_article and link:
        html = fetch_html(session, link, timeout=cfg.timeout, max_chars=cfg.max_html_chars)
        if html:
            fetched_any = True
            text, links = html_to_text_and_links(html, link)
            article_links = links
            process_blob("article_html", html, link)
            process_blob("article_text", text, link)
        if cfg.sleep_s:
            time.sleep(cfg.sleep_s)

    if cfg.fetch_contact_pages and base_domain:
        candidates = pick_contact_page_candidates(article_links, link, base_domain)
        for cu in candidates:
            if not same_domain(cu, base_domain):
                continue
            html = fetch_html(session, cu, timeout=cfg.timeout, max_chars=cfg.max_html_chars)
            if not html:
                continue
            fetched_any = True
            text, _ = html_to_text_and_links(html, cu)
            process_blob("contact_page_html", html, cu)
            process_blob("contact_page_text", text, cu)
            contact_urls.add(cu)
            if cfg.sleep_s:
                time.sleep(cfg.sleep_s)

    social_out = {k: sorted(v) for k, v in social.items() if v}
    contact_urls_out = sorted([u for u in contact_urls if base_domain and is_contact_url_candidate(u, base_domain)])[:MAX_CONTACT_URLS]

    conf = confidence_label(
        has_email=bool(emails),
        has_contact_url=bool(contact_urls_out),
        has_author=bool(author_name),
        fetched=fetched_any,
    )

    event["contact"] = {
        "author_name": author_name,
        "author_url": author_url,
        "emails": sorted(emails),
        "contact_urls": contact_urls_out,
        "social": social_out,
        "confidence": conf,
        "sources": sorted(sources_used),
        "domain": base_domain or None,
    }
    return event


# =============================================================================
# OpenAI Summary + Enrichment (single call per event)
# =============================================================================

def openai_summary_and_enrichment_schema() -> Dict[str, Any]:
    """Combined schema for summary + enrichment in one LLM call (strict mode)."""
    return {
        "name": "summary_and_enrichment",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "event_type": {
                    "type": "string",
                    "enum": [
                        "regulatory",
                        "security_incident",
                        "market_update",
                        "funding",
                        "partnership",
                        "integration",
                        "product_launch",
                        "protocol_upgrade",
                        "listing",
                        "governance",
                        "grant_or_program",
                        "research_report",
                        "other",
                    ],
                },
                "entities": {"type": "array", "items": {"type": "string"}},
                "tokens": {"type": "array", "items": {"type": "string"}},
                "chains": {"type": "array", "items": {"type": "string"}},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "why_it_matters": {"type": "array", "items": {"type": "string"}},
                "recommended_action": {"type": "string"},
                "bd_signal": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "has_action_surface": {"type": "boolean"},
                        "action_type": {
                            "type": "string",
                            "enum": [
                                "partner_intake",
                                "rfp_or_grant",
                                "integration_announced",
                                "vendor_search",
                                "token_launch",
                                "regulatory_opening",
                                "none",
                            ],
                        },
                        "action_detail": {"type": ["string", "null"]},
                        "urgency": {
                            "type": "string",
                            "enum": ["immediate", "near_term", "speculative", "none"],
                        },
                        "surface_type": {
                            "type": "string",
                            "enum": [
                                "live_api_dashboard_surface",
                                "live_security_rail",
                                "named_integration_exploration",
                                "live_adoption_surface",
                                "none",
                            ],
                        },
                    },
                    "required": ["has_action_surface", "action_type", "action_detail", "urgency", "surface_type"],
                },
                "contact_leads": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": ["string", "null"]},
                            "role": {"type": ["string", "null"]},
                            "email": {"type": ["string", "null"]},
                            "url": {"type": ["string", "null"]},
                        },
                        "required": ["name", "role", "email", "url"],
                    },
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": [
                "summary",
                "event_type",
                "entities",
                "tokens",
                "chains",
                "key_points",
                "why_it_matters",
                "recommended_action",
                "bd_signal",
                "contact_leads",
                "confidence",
            ],
        },
        "strict": True,
    }


def build_event_text_for_ai(ev: Dict[str, Any], max_chars: int) -> str:
    """Build event text blob for OpenAI summary+enrichment.

    Uses categorization_text + body + contact hints if present.
    """
    title = (ev.get("title_text") or ev.get("title_raw") or ev.get("title") or "").strip()
    rss_summ = (ev.get("rss_summary_text") or ev.get("summary_text") or "").strip()
    body = (ev.get("body_text") or "").strip()
    cat_text = (ev.get("categorization_text") or "").strip()

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if ev.get("source"):
        parts.append(f"Source: {str(ev.get('source')).strip()}")
    if ev.get("link"):
        parts.append(f"URL: {str(ev.get('link')).strip()}")

    if rss_summ:
        parts.append(f"RSS Summary: {rss_summ}")
    if body:
        parts.append(f"Body: {body}")
    elif cat_text:
        parts.append(f"Text: {cat_text}")

    contact = ev.get("contact") or {}
    hints: List[str] = []
    if isinstance(contact, dict):
        emails = contact.get("emails") or []
        if emails:
            hints.append("Known emails: " + ", ".join([str(e) for e in emails[:10]]))
        author = contact.get("author_name")
        if author:
            hints.append(f"Known author: {author}")
        urls = contact.get("contact_urls") or []
        if urls:
            hints.append("Known contact urls: " + ", ".join([str(u) for u in urls[:5]]))
    if hints:
        parts.append("Contact hints:\n" + "\n".join(hints))

    return safe_truncate("\n".join([p for p in parts if p]).strip(), max_chars)


def call_openai_summary_and_enrichment(
    client: "OpenAI",
    model: str,
    event_text: str,
    summary_max_chars: int,
    max_output_tokens: int,
    max_retries: int,
    sleep_s: float,
    tracker: Optional[TokenTracker] = None,
) -> Dict[str, Any]:
    system = (
        f"You are a Web3 BD intelligence analyst. "
        f"Your job is NOT to summarise news. Your job is to enrich each event so that a downstream AI model "
        f"can determine whether a Web3 company has a concrete, actionable business development opportunity. "
        f"For every event: "
        f"1) Write a 3-5 sentence summary (minimum 18 words, <= {summary_max_chars} characters) that explains what happened, "
        f"identifies any external action surface (open program, RFP, partner intake, integration announced, vendor search), "
        f"and flags who a BD team could approach. If no action surface exists, say so plainly. "
        f"2) Extract structured enrichment data. "
        f"Do NOT invent facts. If a field is not present in the source text, return empty lists or nulls. "
        f"Keep key_points to 1-3 bullets, why_it_matters to 1-2 bullets (BD lens only), recommended_action to one concrete sentence. "
        f"3) Populate bd_signal with your assessment of whether this event contains a qualifying BD action surface. "
        f"Keep bd_signal consistent with recommended_action: if recommended_action is not a Monitor action, bd_signal should usually show a real action surface too."
    )
    user = (
        f"Event text:\n{event_text}\n\n"
        "Rules:\n"
        f"- summary: 3-5 sentences, minimum 18 words (<= {summary_max_chars} chars).\n"
        "  Sentence 1: what happened (factual).\n"
        "  Sentence 2: whether an external action surface exists and what it is (open program / RFP / partner intake / integration / vendor search). If none, state \"No external action surface identified.\"\n"
        "  Sentence 3: who a BD team could approach, if determinable from the text.\n"
        "  Sentence 4: why it matters for BD or the most concrete next step, if supported by the text.\n"
        "  Sentence 5 (optional): additional context if clearly supported by the text.\n"
        "\n"
        "- event_type: must be one of the enum values below.\n"
        "  NOTE: \"funding\" = venture raise only. \"grant_or_program\" = open external program with applications.\n"
        "  Do not conflate these. A Series A is funding. An open grant round is grant_or_program.\n"
        "  Enum values: regulatory | security_incident | market_update | funding | partnership | integration |\n"
        "               product_launch | protocol_upgrade | listing | governance | grant_or_program | research_report | other\n"
        "\n"
        "- entities: named organizations, projects, or protocols only. No generic words (e.g. \"blockchain\", \"team\").\n"
        "\n"
        "- tokens: token symbols only (e.g. BTC, ETH, SOL). Omit if not present.\n"
        "\n"
        "- chains: blockchain network names only (e.g. Ethereum, Solana, Base). Omit if not present.\n"
        "\n"
        "- key_points: 1-3 short bullets. Focus on facts relevant to a BD team. Skip price, sentiment, and opinion.\n"
        "\n"
        "- why_it_matters: 1-2 short bullets. BD lens only.\n"
        "  Ask: \"Why would a Web3 BD team care about this?\" not \"Why is this significant in general?\"\n"
        "\n"
        "- recommended_action: one concrete sentence grounded in the event text.\n"
        "  Must describe a specific action a BD team could take (e.g. \"Apply to the open RFP at X\" or \"Reach out to Y's ecosystem team re the announced integration program\").\n"
        "  If no concrete action is available, write: \"Monitor - no immediate action surface identified.\"\n"
        "  If the source exposes a live API/dashboard/docs surface, a live wallet-security rail, a named integration exploration, or a live adoption surface,\n"
        "  recommended_action should reflect that directly rather than falling back to generic monitoring.\n"
        "\n"
        "- bd_signal: structured object assessing BD relevance.\n"
        "  Fields:\n"
        "    has_action_surface: true | false\n"
        "    action_type: \"partner_intake\" | \"rfp_or_grant\" | \"integration_announced\" | \"vendor_search\" | \"token_launch\" | \"regulatory_opening\" | \"none\"\n"
        "    action_detail: the strongest action-bearing sentence from the source text, preferably a live API/dashboard/docs line,\n"
        "      a live security/protection line, a named integration exploration line, or a live adoption/usage line. Do not prefer a weak\n"
        "      feedback sentence if a stronger action-bearing sentence exists elsewhere in the same text. Null if none.\n"
        "    urgency: \"immediate\" | \"near_term\" | \"speculative\" | \"none\"\n"
        "      - immediate: deadline stated or program closes soon\n"
        "      - near_term: open now, no stated deadline\n"
        "      - speculative: implied but not confirmed\n"
        "      - none: no action surface\n"
        "    surface_type: \"live_api_dashboard_surface\" | \"live_security_rail\" | \"named_integration_exploration\" | \"live_adoption_surface\" | \"none\"\n"
        "      - live_api_dashboard_surface: live API/demo/docs/dashboard/release/open-source implementation surface\n"
        "      - live_security_rail: live wallet-security or scam-protection rail other teams could support or integrate around\n"
        "      - named_integration_exploration: founder/operator explicitly exploring a named chain/ecosystem integration\n"
        "      - live_adoption_surface: live product/dashboard/staking surface with immediate usage/adoption actions\n"
        "\n"
        "- contact_leads: only include if explicitly named in the source text (name, role, email, or URL). Return [] if none.\n"
        "\n"
        "- confidence: 0.0-1.0 reflecting your certainty in the enrichment, not the BD opportunity quality.\n"
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": openai_summary_and_enrichment_schema()},
                max_completion_tokens=max_output_tokens,
                temperature=0,
            )
            if tracker:
                tracker.add_usage(resp.usage, model)

            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)

            def _uniq_clean(xs: List[str], max_n: int) -> List[str]:
                out: List[str] = []
                seen: Set[str] = set()
                for x in xs:
                    x = str(x).strip()
                    if not x:
                        continue
                    k = x.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(x)
                    if len(out) >= max_n:
                        break
                return out

            data["summary"] = str(data.get("summary", "") or "").strip()
            data["entities"] = _uniq_clean(data.get("entities", []) or [], 12)
            data["tokens"] = _uniq_clean(data.get("tokens", []) or [], 12)
            data["chains"] = _uniq_clean(data.get("chains", []) or [], 12)
            data["key_points"] = _uniq_clean(data.get("key_points", []) or [], 3)
            data["why_it_matters"] = _uniq_clean(data.get("why_it_matters", []) or [], 2)
            data["recommended_action"] = str(data.get("recommended_action", "") or "").strip()

            allowed_event_types = {
                "regulatory",
                "security_incident",
                "market_update",
                "funding",
                "partnership",
                "integration",
                "product_launch",
                "protocol_upgrade",
                "listing",
                "governance",
                "grant_or_program",
                "research_report",
                "other",
            }
            et = str(data.get("event_type", "") or "").strip().lower()
            data["event_type"] = et if et in allowed_event_types else "other"

            bd_signal = data.get("bd_signal") or {}
            if not isinstance(bd_signal, dict):
                bd_signal = {}
            action_type = str(bd_signal.get("action_type", "") or "").strip()
            if action_type not in {
                "partner_intake",
                "rfp_or_grant",
                "integration_announced",
                "vendor_search",
                "token_launch",
                "regulatory_opening",
                "none",
            }:
                action_type = "none"
            urgency = str(bd_signal.get("urgency", "") or "").strip()
            if urgency not in {"immediate", "near_term", "speculative", "none"}:
                urgency = "none"
            surface_type = str(bd_signal.get("surface_type", "") or "").strip()
            if surface_type not in _BD_SURFACE_TYPES:
                surface_type = "none"
            action_detail = bd_signal.get("action_detail")
            if action_detail is not None:
                action_detail = str(action_detail).strip() or None
            data["bd_signal"] = {
                "has_action_surface": bool(bd_signal.get("has_action_surface", False)),
                "action_type": action_type,
                "action_detail": action_detail,
                "urgency": urgency,
                "surface_type": surface_type,
            }

            try:
                conf = float(data.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            data["confidence"] = max(0.0, min(1.0, conf))

            leads = data.get("contact_leads") or []
            clean_leads: List[Dict[str, Any]] = []
            for ld in leads[:5]:
                if not isinstance(ld, dict):
                    continue
                clean_leads.append(
                    {
                        "name": ld.get("name") if ld.get("name") is not None else None,
                        "role": ld.get("role") if ld.get("role") is not None else None,
                        "email": ld.get("email") if ld.get("email") is not None else None,
                        "url": ld.get("url") if ld.get("url") is not None else None,
                    }
                )
            data["contact_leads"] = clean_leads
            return reconcile_preprocess_enrichment(event_text, data)

        except Exception as e:
            if _is_insufficient_quota(e):
                raise RuntimeError(
                    "OpenAI API quota exhausted (insufficient_quota). "
                    "Update billing/add credits or use a different API key."
                ) from e
            last_err = e
            time.sleep(sleep_s * attempt)
            continue

    raise RuntimeError(f"OpenAI summary+enrichment call failed after {max_retries} attempts: {last_err}")


# =============================================================================
# IO helpers
# =============================================================================

def write_jsonl_line(f, obj: Dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def event_to_final_events_row(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Build one Final_events row from an AI-enriched event."""
    row: Dict[str, Any] = dict(ev)

    contact = ev.get("contact") if isinstance(ev.get("contact"), dict) else {}
    ai_cat = ev.get("ai_category") if isinstance(ev.get("ai_category"), dict) else {}
    ai_enrichment = ev.get("ai_enrichment") if isinstance(ev.get("ai_enrichment"), dict) else {}

    primary_id = ev.get("ai_primary_category_id") or ai_cat.get("primary_category_id")
    secondary_ids = ev.get("ai_secondary_category_ids") or ai_cat.get("secondary_category_ids") or []

    row["event_id"] = ev.get("event_id")
    row["opportunity_flag"] = ev.get("opportunity_flag")
    row["source"] = ev.get("source") or ev.get("source_name")
    row["title"] = ev.get("title_text") or ev.get("title_raw") or ev.get("title")
    row["url"] = ev.get("link") or ev.get("url")

    row["published_at"] = ev.get("published_at")
    row["ingested_at"] = ev.get("ingested_at") or ev.get("created_at")

    row["author_name"] = ev.get("author_name") or contact.get("author_name")
    row["author_url"] = ev.get("author_url") or contact.get("author_url")

    row["summary_text"] = ev.get("summary_text") or ev.get("page_excerpt_raw") or ev.get("description_raw")
    row["summary_html"] = ev.get("summary_html") or ev.get("description_raw")
    row["body_text"] = ev.get("body_text") or ev.get("body_text_raw") or ev.get("content_raw")

    row["ai_primary_category_id"] = int(primary_id) if primary_id is not None else None
    row["ai_secondary_category_ids"] = [int(x) for x in secondary_ids] if isinstance(secondary_ids, list) else []
    row["ai_category_confidence"] = _safe_float(ev.get("ai_category_confidence") or ai_cat.get("confidence"), 0.0)
    row["ai_category_reasoning"] = ai_cat.get("reasoning")
    row["ai_category"] = ai_cat if isinstance(ai_cat, dict) else {}
    row["ai_enrichment"] = ai_enrichment

    top_bd_signal = ev.get("bd_signal") if isinstance(ev.get("bd_signal"), dict) else None
    nested_bd_signal = ai_enrichment.get("bd_signal") if isinstance(ai_enrichment.get("bd_signal"), dict) else None
    row["bd_signal"] = top_bd_signal or nested_bd_signal or {
        "has_action_surface": False,
        "action_type": "none",
        "action_detail": None,
        "urgency": "none",
        "surface_type": "none",
    }
    return row


def write_final_events_from_rows(rows: Iterable[Dict[str, Any]], final_out_path: Path) -> int:
    """Create Final_events JSONL by remapping AI-enriched rows in memory."""
    final_out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with final_out_path.open("w", encoding="utf-8") as fout:
        for ev in rows:
            row = event_to_final_events_row(ev)
            write_jsonl_line(fout, row)
            written += 1
    return written


def load_existing_event_ids(out_path: Path) -> Set[str]:
    """Optional caching: skip event_ids already present in output."""
    if not out_path.exists():
        return set()
    ids: Set[str] = set()
    try:
        with out_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("event_id"):
                    ids.add(str(obj["event_id"]))
    except Exception:
        return set()
    return ids


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="CalyxonAI: preprocess + contact extraction + AI summary/enrichment.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help=f"Input events (.json or .jsonl). Default: {DEFAULT_INPUT}")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Legacy AI-enriched JSONL (not written; cache only). Default: {DEFAULT_OUTPUT}")
    ap.add_argument("--filtered-output", default=str(DEFAULT_FILTERED_OUTPUT), help=f"Filtered events JSONL (post date-range/noise). Default: {DEFAULT_FILTERED_OUTPUT}")
    ap.add_argument("--final-output", default=str(DEFAULT_FINAL_OUTPUT), help=f"Final events JSONL for core model input. Default: {DEFAULT_FINAL_OUTPUT}")
    ap.add_argument(
        "--filtered-debug-output",
        default=str(DEFAULT_FILTERED_DEBUG_OUTPUT),
        help=f"Filtered-out debug JSONL path (used only with --keep-filtered). Default: {DEFAULT_FILTERED_DEBUG_OUTPUT}",
    )
    ap.add_argument("--clean-output", default="", help=f"Optional early cleaned JSONL path (pre-filter). Example: {DEFAULT_CLEAN_OUTPUT}")

    # Filtering knobs (defaults chosen for non-CLI usage)
    ap.add_argument(
        "--from-date",
        type=str,
        default="",
        help="Start date (inclusive) for filtering. Format: YYYY-MM-DD.",
    )
    ap.add_argument(
        "--to-date",
        type=str,
        default="",
        help="End date (inclusive) for filtering. Format: YYYY-MM-DD.",
    )
    ap.add_argument("--disable-noise-filter", action="store_true", help="Disable conservative market/TA noise filtering.")
    ap.add_argument("--keep-filtered", action="store_true", help="Write filtered-out items too, marked preprocess_filtered_out=true (debug).")

    # Cleaner knobs
    ap.add_argument("--max-body-chars", type=int, default=DEFAULT_MAX_BODY_CHARS, help=f"Max chars for cleaned body_text (default {DEFAULT_MAX_BODY_CHARS}).")
    ap.add_argument("--max-categorization-chars", type=int, default=1600, help="Max chars for categorization_text (default 1600).")
    ap.add_argument("--max-summary-html-chars", type=int, default=2000, help="Max chars for summary_html (default 2000).")
    ap.add_argument("--nodedupe", action="store_true", help="Disable de-duplication (dedupe is ON by default).")

    # Contact enrichment knobs (defaults OFF to avoid extra traffic)
    ap.add_argument("--fetch-article", action="store_true", help="Fetch article HTML for better contact coverage (slower).")
    ap.add_argument("--fetch-contact-pages", action="store_true", help="Also fetch likely contact pages (same domain) (slower).")
    ap.add_argument("--timeout", type=int, default=15, help="HTTP timeout (seconds).")
    ap.add_argument("--max-html-chars", type=int, default=2_000_000, help="Max HTML chars kept per fetch.")
    ap.add_argument("--sleep-fetch", type=float, default=0.0, help="Sleep between fetches (seconds).")

    # AI enrichment knobs
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"OpenAI model name (default {DEFAULT_MODEL}).")
    ap.add_argument("--ai-max-event-chars", type=int, default=DEFAULT_AI_MAX_EVENT_CHARS, help=f"Max chars sent to OpenAI per event (default {DEFAULT_AI_MAX_EVENT_CHARS}).")
    ap.add_argument("--ai-max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS_AI, help=f"Max OpenAI output tokens (default {DEFAULT_MAX_OUTPUT_TOKENS_AI}).")
    ap.add_argument("--ai-summary-max-chars", type=int, default=DEFAULT_AI_SUMMARY_MAX_CHARS, help=f"Max characters for AI_summary_text; summaries target 3-5 sentences and at least 35 words (default {DEFAULT_AI_SUMMARY_MAX_CHARS}).")
    ap.add_argument("--ai-batch-size", type=int, default=DEFAULT_AI_BATCH_SIZE, help=f"Events per AI worker batch (default {DEFAULT_AI_BATCH_SIZE}).")
    ap.add_argument("--ai-max-workers", type=int, default=DEFAULT_AI_MAX_WORKERS, help=f"Parallel AI worker threads (default {DEFAULT_AI_MAX_WORKERS}).")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries for OpenAI calls.")
    ap.add_argument("--sleep-ai", type=float, default=0.15, help="Base sleep seconds for retry backoff.")
    ap.add_argument("--dry-run", action="store_true", help="Do everything except call OpenAI; writes ai_enrichment as skipped.")
    ap.add_argument("--no-cache", action="store_true", help="Disable caching (by default, skips event_ids already in output file).")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input)
    out_path = Path(args.output)
    final_out_path = Path(args.final_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_out_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_out_path = Path(args.filtered_output) if args.filtered_output else None
    filtered_debug_out_path = Path(args.filtered_debug_output) if args.filtered_debug_output else None
    clean_out_path = Path(args.clean_output) if args.clean_output else None
    if clean_out_path:
        clean_out_path.parent.mkdir(parents=True, exist_ok=True)
    if filtered_out_path:
        filtered_out_path.parent.mkdir(parents=True, exist_ok=True)
    if filtered_debug_out_path:
        filtered_debug_out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        log_info(f"[Preprocess_AI] ERROR: input not found: {in_path}")
        return 2

    # OpenAI clients
    client_pool: Optional[ThreadLocalOpenAIClientPool] = None
    tracker = TokenTracker()
    if not args.dry_run:
        if OpenAI is None:
            log_info("[Preprocess_AI] ERROR: openai is not installed. Run: pip install openai")
            return 2
        if load_dotenv:
            load_dotenv()
        api_keys = load_openai_api_keys()
        if not api_keys:
            log_info("[Preprocess_AI] ERROR: no OpenAI API keys found. Set OPENAI_API_KEY_1..3 or OPENAI_API_KEY.")
            return 2
        client_pool = ThreadLocalOpenAIClientPool(api_keys)
        log_info(f"[Preprocess_AI] OpenAI key pool size: {len(api_keys)}")

    dedupe_enabled = not bool(args.nodedupe)
    seen: Set[str] = set()
    seen_urls: Set[str] = set()
    duplicate_url_count: int = 0

    # Optional caching by final output file
    cached_ids: Set[str] = set()
    if not bool(args.no_cache):
        cached_ids = load_existing_event_ids(final_out_path)
        if out_path.exists() and out_path != final_out_path:
            cached_ids |= load_existing_event_ids(out_path)

    now_dt = _dt.datetime.now(tz=_dt.timezone.utc)
    now_iso = now_dt.isoformat()

    try:
        from_dt = _parse_date_arg(args.from_date)
        to_dt = _parse_date_arg(args.to_date)
    except ValueError as e:
        log_info(f"[Preprocess_AI] ERROR: {e}")
        return 2

    default_range_used = False
    if from_dt is None and to_dt is None:
        default_range_used = True
        from_dt = (now_dt - _dt.timedelta(days=DEFAULT_MAX_AGE_DAYS)).date()
        to_dt = now_dt.date()

    fetch_cfg = FetchConfig(
        fetch_article=bool(args.fetch_article),
        fetch_contact_pages=bool(args.fetch_contact_pages),
        timeout=int(args.timeout),
        max_html_chars=int(args.max_html_chars),
        sleep_s=float(args.sleep_fetch),
    )

    total_read = 0
    total_cleaned = 0
    total_skipped_cached = 0
    total_filtered_old = 0
    total_filtered_noise = 0
    seen_event_urls = set()
    duplicate_event_url_count = 0
    total_filtered_kept = 0
    total_openai_calls = 0
    total_written_ai = 0
    total_written_filtered_out = 0

    clean_f = clean_out_path.open("w", encoding="utf-8") if clean_out_path else None
    filtered_f = filtered_out_path.open("w", encoding="utf-8") if filtered_out_path else None
    debug_f = (
        filtered_debug_out_path.open("w", encoding="utf-8")
        if bool(args.keep_filtered) and filtered_debug_out_path
        else None
    )
    started = time.time()

    try:
        # STAGE 1: Clean + filter + contact enrichment -> events_filtered.jsonl
        with requests.Session() as session:
            for item in iter_items_from_file(in_path):
                total_read += 1

                cleaned = clean_one_item(
                    item=item,
                    now_iso=now_iso,
                    max_body_chars=int(args.max_body_chars),
                    max_categorization_chars=int(args.max_categorization_chars),
                    max_summary_html_chars=int(args.max_summary_html_chars),
                )

                eid = str(cleaned.get("event_id") or "")
                if not eid:
                    # Should not happen, but keep safe.
                    eid = stable_event_id(
                        source=str(cleaned.get("source") or ""),
                        guid=str(cleaned.get("guid") or ""),
                        link=str(cleaned.get("link") or ""),
                        published_iso=str(cleaned.get("published_at") or ""),
                        title=str(cleaned.get("title_text") or ""),
                    )
                    cleaned["event_id"] = eid

                if dedupe_enabled:
                    if eid in seen:
                        continue
                    seen.add(eid)

                total_cleaned += 1
                if clean_f is not None:
                    tmp_clean = dict(cleaned)
                    tmp_clean["rss_summary_text"] = str(tmp_clean.get("summary_text") or "").strip()
                    tmp_clean.pop("summary_text", None)
                    write_jsonl_line(clean_f, tmp_clean)

                # Date range filter
                out_of_range, range_reason = is_outside_date_range(
                    published_iso=str(cleaned.get("published_at") or ""),
                    from_dt=from_dt,
                    to_dt=to_dt,
                )
                if out_of_range:
                    total_filtered_old += 1
                    if debug_f is not None:
                        cleaned["preprocess_filtered_out"] = True
                        cleaned["preprocess_filter_reason"] = range_reason or "date_out_of_range"
                        cleaned["contact"] = {"skipped": True}
                        cleaned["ai_enrichment"] = {"skipped": True, "skipped_reason": "filtered_out:date_range"}
                        cleaned["rss_summary_text"] = str(cleaned.get("summary_text") or "").strip()
                        cleaned.pop("summary_text", None)
                        cleaned["AI_summary_text"] = "No AI Response"
                        write_jsonl_line(debug_f, cleaned)
                        total_written_filtered_out += 1
                    continue
                # Dedupe by event URL (after date range filter, before noise filter)
                if dedupe_enabled:
                    event_url = (cleaned.get('event_url') or cleaned.get('url') or cleaned.get('link') or '').strip()
                    if event_url:
                        norm_url = normalize_url(event_url) or event_url.rstrip('/')
                        if norm_url in seen_event_urls:
                            duplicate_event_url_count += 1
                            if debug_f is not None:
                                cleaned.pop('summary_text', None)
                                cleaned['AI_summary_text'] = 'Filtered out (duplicate event url)'
                                write_jsonl_line(debug_f, cleaned)
                                total_written_filtered_out += 1
                            continue
                        seen_event_urls.add(norm_url)


                # Noise filter (lenient)
                if not bool(args.disable_noise_filter):
                    drop, why = is_obvious_noise_item(
                        title_text=str(cleaned.get("title_text") or ""),
                        summary_text=str(cleaned.get("summary_text") or ""),
                        link=str(cleaned.get("link") or ""),
                    )
                    if drop:
                        total_filtered_noise += 1
                        if debug_f is not None:
                            cleaned["preprocess_filtered_out"] = True
                            cleaned["preprocess_filter_reason"] = why or "noise"
                            cleaned["contact"] = {"skipped": True}
                            cleaned["ai_enrichment"] = {"skipped": True, "skipped_reason": "filtered_out:noise"}
                            cleaned["rss_summary_text"] = str(cleaned.get("summary_text") or "").strip()
                            cleaned.pop("summary_text", None)
                            cleaned["AI_summary_text"] = "No AI Response"
                            write_jsonl_line(debug_f, cleaned)
                            total_written_filtered_out += 1
                        continue

                # Contact enrichment
                enriched = enrich_contact_one(cleaned, session=session, cfg=fetch_cfg)

                # Preserve RSS summary before AI overrides summary_text
                # Always keep RSS summary separately, and do not emit summary_text in outputs.
                rss_summary = str(enriched.get("summary_text") or enriched.get("rss_summary_text") or "").strip()
                enriched["rss_summary_text"] = rss_summary
                enriched.pop("summary_text", None)

                # STAGE 1 output: filtered, contact-enriched events only
                if filtered_f is not None:
                    write_jsonl_line(filtered_f, enriched)
                    total_filtered_kept += 1

    finally:
        if clean_f is not None:
            clean_f.close()
        if filtered_f is not None:
            filtered_f.close()
        if debug_f is not None:
            debug_f.close()

    log_info(f"[Preprocess_AI] Total records read from events.json: {total_read}")

    print("\n" + "=" * 72)
    log_info("[Preprocess_AI] PREFILTER SUMMARY")
    print("=" * 72)
    print(f"Accepted (kept):           {total_filtered_kept}")
    print(f"Rejected due to date range:{total_filtered_old}")
    print(f"Rejected due to noise:     {total_filtered_noise}")
    print(f"Rejected due to duplicate urls:{duplicate_event_url_count}")
    print("=" * 72)

    # STAGE 2: AI enrichment from events_filtered.jsonl -> in-memory rows
    ai_input_path = filtered_out_path
    if ai_input_path is None or not ai_input_path.exists():
        log_info("[Preprocess_AI] ERROR: filtered input not found for AI stage.")
        return 2

    ai_records = list(iter_items_from_file(ai_input_path))
    ai_batch_size = max(1, int(args.ai_batch_size))
    ai_max_workers = max(1, int(args.ai_max_workers))
    ai_batches = [
        ai_records[i : i + ai_batch_size]
        for i in range(0, len(ai_records), ai_batch_size)
    ]
    log_info(
        f"[Preprocess_AI] AI stage batching: {len(ai_batches)} batches | "
        f"batch_size={ai_batch_size} | workers={ai_max_workers}"
    )

    def _default_combined() -> Dict[str, Any]:
        return {
            "summary": "",
            "event_type": "other",
            "entities": [],
            "tokens": [],
            "chains": [],
            "key_points": [],
            "why_it_matters": [],
            "recommended_action": "",
            "bd_signal": {
                "has_action_surface": False,
                "action_type": "none",
                "action_detail": None,
                "urgency": "none",
                "surface_type": "none",
            },
            "contact_leads": [],
            "confidence": 0.0,
        }

    def _process_ai_batch(batch_num: int, batch_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_started = time.time()
        local_rows: List[Dict[str, Any]] = []
        local_openai_calls = 0
        local_skipped_cached = 0
        local_client: Optional[OpenAI] = None
        thread_name = pretty_thread_name()

        log_info(
            f"[Preprocess_AI] Batch {batch_num}/{len(ai_batches)} assigned to {thread_name} "
            f"({len(batch_rows)} events)"
        )

        if not args.dry_run:
            assert client_pool is not None
            local_client = client_pool.get_client()

        for enriched in batch_rows:
            enriched = dict(enriched)
            eid = str(enriched.get("event_id") or "")
            if cached_ids and eid in cached_ids:
                local_skipped_cached += 1
                continue

            ai_text = build_event_text_for_ai(enriched, int(args.ai_max_event_chars))
            combined = _default_combined()
            meta = {"skipped": False, "skipped_reason": None}

            if args.dry_run:
                meta = {"skipped": True, "skipped_reason": "dry-run (no OpenAI call)"}
            else:
                assert local_client is not None
                try:
                    combined = call_openai_summary_and_enrichment(
                        client=local_client,
                        model=str(args.model),
                        event_text=ai_text,
                        summary_max_chars=int(args.ai_summary_max_chars),
                        max_output_tokens=int(args.ai_max_output_tokens),
                        max_retries=int(args.max_retries),
                        sleep_s=float(args.sleep_ai),
                        tracker=tracker,
                    )
                    local_openai_calls += 1
                except Exception as e:
                    meta = {"skipped": True, "skipped_reason": "openai_error:" + str(e)}

            enriched["AI_summary_text"] = finalize_ai_summary(
                str((combined or {}).get("summary", "") or ""),
                combined or {},
                int(args.ai_summary_max_chars),
            )

            enrichment_payload = {k: v for k, v in combined.items() if k != "summary"}
            enriched["ai_enrichment"] = {
                **enrichment_payload,
                **meta,
                "created_at": _now_iso(),
                "model": str(args.model) if not args.dry_run else "dry-run",
                "version": "preprocess_ai-enrich-v2",
                "max_event_chars": int(args.ai_max_event_chars),
                "max_output_tokens": int(args.ai_max_output_tokens),
            }

            if not (enriched.get("ingested_at") or "").__str__().strip():
                enriched["ingested_at"] = _now_iso()

            local_rows.append(enriched)

        batch_elapsed = time.time() - batch_started
        batch_events = len(local_rows) + local_skipped_cached
        batch_rate = (batch_events / batch_elapsed) if batch_elapsed > 0 else 0.0

        return {
            "batch_num": batch_num,
            "thread_name": thread_name,
            "rows": local_rows,
            "openai_calls": local_openai_calls,
            "skipped_cached": local_skipped_cached,
            "batch_rate": batch_rate,
        }

    results_by_batch: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=ai_max_workers) as ex:
        futures = [
            ex.submit(_process_ai_batch, batch_num, batch_rows)
            for batch_num, batch_rows in enumerate(ai_batches, 1)
        ]
        for fut in as_completed(futures):
            res = fut.result()
            results_by_batch[int(res["batch_num"])] = res
            thread_label = str(res["thread_name"]).replace("_", "")
            print(
                f"{_now_hms_utc()} [Preprocess_AI] {thread_label}: Completed batch "
                f"{res['batch_num']}/{len(ai_batches)} (written={len(res['rows'])}, "
                f"cached_skipped={res['skipped_cached']}, openai_calls={res['openai_calls']}, "
                f"time={float(res.get('batch_rate', 0.0)):.2f}ev/s)"
            )

    def _iter_rows_in_order() -> Iterable[Dict[str, Any]]:
        nonlocal total_openai_calls, total_skipped_cached, total_written_ai
        for batch_num in sorted(results_by_batch.keys()):
            res = results_by_batch[batch_num]
            total_openai_calls += int(res.get("openai_calls", 0))
            total_skipped_cached += int(res.get("skipped_cached", 0))
            for enriched in res.get("rows", []):
                total_written_ai += 1
                yield enriched

    total_written_final = write_final_events_from_rows(_iter_rows_in_order(), final_out_path)

    log_info(f"[Preprocess_AI] Input:  {in_path}")
    if clean_out_path:
        log_info(f"[Preprocess_AI] Clean output: {clean_out_path}")
    if filtered_out_path:
        log_info(f"[Preprocess_AI] Filtered output: {filtered_out_path}")
    if bool(args.keep_filtered) and filtered_debug_out_path:
        log_info(f"[Preprocess_AI] Filtered debug output: {filtered_debug_out_path}")
    log_info("[Preprocess_AI] AI-enriched output: in-memory (events_ai_enriched.jsonl not written)")
    log_info(f"[Preprocess_AI] Final events output: {final_out_path}")
    log_info(f"[Preprocess_AI] Records read:        {total_read}")
    log_info(f"[Preprocess_AI] Records cleaned:     {total_cleaned}")
    if filtered_out_path:
        log_info(f"[Preprocess_AI] Records kept (filtered): {total_filtered_kept}")
    if total_written_filtered_out:
        log_info(f"[Preprocess_AI] Filtered-out written: {total_written_filtered_out}")
    log_info(f"[Preprocess_AI] Records written:     {total_written_ai}")
    log_info(f"[Preprocess_AI] Final rows written:  {total_written_final}")
    log_info(f"[Preprocess_AI] Skipped (cached):    {total_skipped_cached}")
    if default_range_used:
        log_info(
            f"[Preprocess_AI] Date range:           default last {DEFAULT_MAX_AGE_DAYS} days "
            f"({from_dt.isoformat()} to {to_dt.isoformat()})"
        )
    else:
        log_info(
            f"[Preprocess_AI] Date range:           "
            f"{from_dt.isoformat() if from_dt else 'ANY'} to {to_dt.isoformat() if to_dt else 'ANY'}"
        )
    log_info(f"[Preprocess_AI] Filtered (date range): {total_filtered_old}")
    if not bool(args.disable_noise_filter):
        log_info(f"[Preprocess_AI] Filtered (noise):     {total_filtered_noise} (market/TA/prediction)")
        log_info(f"[Preprocess_AI] Filtered (duplicate urls): {duplicate_event_url_count}")
    else:
        log_info("[Preprocess_AI] Filtered (noise):    DISABLED")
    log_info(f"[Preprocess_AI] Dedupe: {'ON' if dedupe_enabled else 'OFF'}")
    log_info(f"[Preprocess_AI] OpenAI calls:        {total_openai_calls}")
    log_info(f"[Preprocess_AI] Fetch article:       {'ON' if fetch_cfg.fetch_article else 'OFF'}")
    log_info(f"[Preprocess_AI] Fetch contact pages: {'ON' if fetch_cfg.fetch_contact_pages else 'OFF'}")

    if bleach is None:
        log_info("[Preprocess_AI] NOTE: 'bleach' not installed. summary_html is plain-text escaped. Install bleach for HTML sanitization.")

    if not args.dry_run:
        tracker.print_summary()

    total_elapsed = time.time() - started
    log_info(f"[Preprocess_AI] Execution time: {total_elapsed:.2f}s ({total_elapsed / 60.0:.2f} minutes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())




