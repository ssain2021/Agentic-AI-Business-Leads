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
    - ai_summary (2â€“3 sentence summary)
    - ai_enrichment (event_type, entities, key_points, why_it_matters, contact_leads, etc.)

Defaults (no CLI required)
-------------------------
Input : ./Data/events.json
Output: ./Data/events_ai_enriched.jsonl

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
import time
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


# =============================================================================
# Defaults / paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data"

DEFAULT_INPUT = DATA_DIR / "events.json"
DEFAULT_CLEAN_OUTPUT = DATA_DIR / "events_clean.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "events_ai_enriched.jsonl"
DEFAULT_FILTERED_OUTPUT = DATA_DIR / "events_filtered.jsonl"

# Default window for date-range filtering if no explicit range is provided.
# (Per latest requirement: last 7 days)
DEFAULT_MAX_AGE_DAYS = 7

# Keep detailed texts for downstream (up to 15000 chars for cleaned body)
DEFAULT_MAX_BODY_CHARS = 15000

# AI request text cap (controls OpenAI cost). Uses title/summary/body + contact hints.
DEFAULT_AI_MAX_EVENT_CHARS = 6000

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_OUTPUT_TOKENS_AI = 800

# AI summary max chars (configurable; default 250)
DEFAULT_AI_SUMMARY_MAX_CHARS = 250


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

    def get_pricing(self, model: str):
        """Return (input_cost_per_1m, output_cost_per_1m) in USD per 1M tokens."""
        pricing = {
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
            "gpt-4.1-mini": (0.15, 0.6),
            "o3": (2.0, 8.0),
            "o3-mini": (1.1, 4.4),
            "gpt-5.1": (1.25, 10.0),
            "gpt-5-mini": (0.25, 2.0),
        }
        return pricing.get(model)

    def _ensure_model(self, model: str):
        if model not in self.model_usage:
            self.model_usage[model] = {"input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0}

    def add_usage(self, usage, model: str):
        if not usage:
            return

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
        print("\n" + "=" * 72)
        print("[Preprocess_AI] MODEL-WISE USAGE SUMMARY")
        print("=" * 72)
        for model, data in self.model_usage.items():
            total_tokens = data["input_tokens"] + data["output_tokens"]
            print(f"\nModel: {model}")
            print("-" * 72)
            print(f"  Input tokens:   {data['input_tokens']:,}")
            print(f"  Output tokens:  {data['output_tokens']:,}")
            print(f"  Total tokens:   {total_tokens:,}")
            if "estimated_cost" in data:
                print(f"  Estimated cost: ${data['estimated_cost']:.4f}")
        print("\n" + "=" * 72)
        print("[Preprocess_AI] TOTAL USAGE SUMMARY")
        print("=" * 72)
        total_tokens = self.total_input_tokens + self.total_output_tokens
        print(f"  Total input tokens:   {self.total_input_tokens:,}")
        print(f"  Total output tokens:  {self.total_output_tokens:,}")
        print(f"  Grand total tokens:   {total_tokens:,}")
        print(f"  Total estimated cost: ${self.total_estimated_cost:.4f}")
        print("=" * 72)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    return text[: max_chars - 1] + "â€¦"


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
        return text[: max_chars - 1] + "â€¦"
    return cut + "â€¦"


_SENT_SPLIT_RE = re.compile(r"(?<=[\.!\?])\s+")


def finalize_ai_summary(raw_summary: str, enrichment: Dict[str, Any], max_chars: int) -> str:
    """Ensure 2â€“3 sentences and length cap; fallback if missing."""
    raw = (raw_summary or "").strip()
    if not raw:
        return "No AI Response"

    raw = _WS_RE.sub(" ", raw).strip()
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(raw) if s.strip()]
    if not sentences:
        return "No AI Response"

    # Keep at most 3 sentences
    sentences = sentences[:3]

    # If only 1 sentence, append a second using why_it_matters or recommended_action
    if len(sentences) == 1:
        extra = ""
        why = enrichment.get("why_it_matters") or []
        if isinstance(why, list) and why:
            extra = str(why[0]).strip()
        if not extra:
            extra = str(enrichment.get("recommended_action") or "").strip()
        if extra:
            extra = _WS_RE.sub(" ", extra).strip()
            if extra and extra[-1] not in ".!?":
                extra += "."
            sentences.append(extra)

    # Re-join
    out = " ".join(sentences).strip()
    out = _WS_RE.sub(" ", out).strip()

    # Enforce max_chars
    if max_chars and len(out) > max_chars:
        out = smart_truncate(out, max_chars)

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


# Conservative, lenient noise filter: only drops obvious TA/prediction items.
NOISE_TITLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bprice\s+prediction\b", re.IGNORECASE),
    re.compile(r"\btechnical\s+analysis\b", re.IGNORECASE),
    re.compile(r"\bchart\s+analysis\b", re.IGNORECASE),
    re.compile(r"\btrading\s+signals\b", re.IGNORECASE),
    re.compile(r"\b(?:buy|sell)\s+signal\b", re.IGNORECASE),
]

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
    body_text_raw = item.get("body_text_raw") or ""

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
        "body_text_raw": body_text_raw,
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
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
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
    body_text_raw = str(fields["body_text_raw"] or "")
    source = str(fields["source"] or "")
    link = str(fields["link"] or "")
    guid = str(fields["guid"] or "")

    published_iso, published_src = best_effort_parse_datetime(fields["published_any"])
    if not published_iso:
        published_iso = now_iso
        published_src = "fallback_now"

    title_text = normalize_text(html_to_text(title_raw), max_chars=400)
    summary_text = normalize_text(html_to_text(description_raw), max_chars=1200)

    content_body_text = normalize_text(html_to_text(content_raw), max_chars=max_body_chars)
    body_text_raw_clean = normalize_text(html_to_text(body_text_raw), max_chars=max_body_chars)
    description_body_text = normalize_text(html_to_text(description_raw), max_chars=max_body_chars)

    body_text = content_body_text
    if len(content_body_text) < 400:
        fallback_candidates = [body_text_raw_clean, description_body_text]
        if all(len(candidate) >= 400 for candidate in fallback_candidates):
            body_text = max(fallback_candidates, key=len)
        elif not content_body_text and any(fallback_candidates):
            body_text = max(fallback_candidates, key=len)

    if len(body_text) > max_body_chars:
        body_text = body_text[:max_body_chars]

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
        f"You are a Web3 intelligence analyst. "
        f"For the given event, provide: "
        f"1) A concise 2-3 sentence summary (<= {summary_max_chars} characters) explaining what happened and why it matters. "
        f"2) Structured enrichment data (entities, tokens, chains, key points, etc.). "
        f"Do NOT invent facts. If a field is not present, return empty lists or nulls. "
        f"Keep key_points to 1-3 bullets, why_it_matters to 1-2 bullets, recommended_action to one sentence."
    )
    user = (
        f"Event text:\n{event_text}\n\n"
        "Rules:\n"
        f"- summary: 2-3 sentences (<= {summary_max_chars} chars) explaining what happened and why it matters.\n"
        "- event_type: must be one of the enum values.\n"
        "- entities: organization/project/protocol names only (no generic words).\n"
        "- tokens: symbols like BTC, ETH; omit if not present.\n"
        "- chains: network names like Ethereum, Solana; omit if not present.\n"
        "- key_points: 1-3 short bullets.\n"
        "- why_it_matters: 1-2 short bullets.\n"
        "- recommended_action: one sentence.\n"
        "- contact_leads: only include leads explicitly present (names/roles/emails/urls). If none, return [].\n"
        "- confidence: 0-1 reflecting certainty.\n"
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": openai_summary_and_enrichment_schema()},
                max_tokens=max_output_tokens,
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
            return data

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
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Final AI-enriched JSONL. Default: {DEFAULT_OUTPUT}")
    ap.add_argument("--filtered-output", default=str(DEFAULT_FILTERED_OUTPUT), help=f"Filtered events JSONL (post date-range/noise). Default: {DEFAULT_FILTERED_OUTPUT}")
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
    ap.add_argument("--ai-summary-max-chars", type=int, default=DEFAULT_AI_SUMMARY_MAX_CHARS, help=f"Max characters for AI_summary_text (default {DEFAULT_AI_SUMMARY_MAX_CHARS}).")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries for OpenAI calls.")
    ap.add_argument("--sleep-ai", type=float, default=0.15, help="Base sleep seconds for retry backoff.")
    ap.add_argument("--dry-run", action="store_true", help="Do everything except call OpenAI; writes ai_enrichment as skipped.")
    ap.add_argument("--no-cache", action="store_true", help="Disable caching (by default, skips event_ids already in output file).")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_out_path = Path(args.filtered_output) if args.filtered_output else None
    clean_out_path = Path(args.clean_output) if args.clean_output else None
    if clean_out_path:
        clean_out_path.parent.mkdir(parents=True, exist_ok=True)
    if filtered_out_path:
        filtered_out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[Preprocess_AI] ERROR: input not found: {in_path}")
        return 2

    # OpenAI client
    client: Optional[OpenAI] = None
    tracker = TokenTracker()
    if not args.dry_run:
        if OpenAI is None:
            print("[Preprocess_AI] ERROR: openai is not installed. Run: pip install openai")
            return 2
        if not os.getenv("OPENAI_API_KEY"):
            print("[Preprocess_AI] ERROR: OPENAI_API_KEY environment variable not set.")
            return 2
        client = OpenAI()

    dedupe_enabled = not bool(args.nodedupe)
    seen: Set[str] = set()
    seen_urls: Set[str] = set()
    duplicate_url_count: int = 0

    # Optional caching by output file
    cached_ids: Set[str] = set()
    if not bool(args.no_cache):
        cached_ids = load_existing_event_ids(out_path)

    now_dt = _dt.datetime.now(tz=_dt.timezone.utc)
    now_iso = now_dt.isoformat()

    try:
        from_dt = _parse_date_arg(args.from_date)
        to_dt = _parse_date_arg(args.to_date)
    except ValueError as e:
        print(f"[Preprocess_AI] ERROR: {e}")
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
    debug_f = out_path.open("a", encoding="utf-8") if bool(args.keep_filtered) else None
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
                        norm_url = event_url.rstrip('/')
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

    print(f"[Preprocess_AI] Total records read from events.json: {total_read}")

    print("\n" + "=" * 72)
    print("[Preprocess_AI] PREFILTER SUMMARY")
    print("=" * 72)
    print(f"Accepted (kept):           {total_filtered_kept}")
    print(f"Rejected due to date range:{total_filtered_old}")
    print(f"Rejected due to noise:     {total_filtered_noise}")
    print(f"Rejected due to duplicate urls:{duplicate_event_url_count}")
    print("=" * 72)

    # STAGE 2: AI enrichment from events_filtered.jsonl -> events_ai_enriched.jsonl
    ai_input_path = filtered_out_path
    if ai_input_path is None or not ai_input_path.exists():
        print("[Preprocess_AI] ERROR: filtered input not found for AI stage.")
        return 2

    with out_path.open("a", encoding="utf-8") as fout:
        for enriched in iter_items_from_file(ai_input_path):
            eid = str(enriched.get("event_id") or "")
            if cached_ids and eid in cached_ids:
                total_skipped_cached += 1
                continue

            ai_text = build_event_text_for_ai(enriched, int(args.ai_max_event_chars))
            combined = {
                "summary": "",
                "event_type": "other",
                "entities": [],
                "tokens": [],
                "chains": [],
                "key_points": [],
                "why_it_matters": [],
                "recommended_action": "",
                "contact_leads": [],
                "confidence": 0.0,
            }
            meta = {"skipped": False, "skipped_reason": None}

            if args.dry_run:
                meta = {"skipped": True, "skipped_reason": "dry-run (no OpenAI call)"}
            else:
                assert client is not None
                try:
                    combined = call_openai_summary_and_enrichment(
                        client=client,
                        model=str(args.model),
                        event_text=ai_text,
                        summary_max_chars=int(args.ai_summary_max_chars),
                        max_output_tokens=int(args.ai_max_output_tokens),
                        max_retries=int(args.max_retries),
                        sleep_s=float(args.sleep_ai),
                        tracker=tracker,
                    )
                    total_openai_calls += 1
                except Exception as e:
                    # Do not fail the pipeline; mark and continue.
                    meta = {"skipped": True, "skipped_reason": "openai_error:" + str(e)}

            # AI_summary_text is ALWAYS present. If no AI response, set explicit placeholder.
            enriched["AI_summary_text"] = finalize_ai_summary(
                str((combined or {}).get("summary", "") or ""),
                combined or {},
                int(args.ai_summary_max_chars),
            )

            # Store enrichment (excluding summary)
            enrichment_payload = {k: v for k, v in combined.items() if k != "summary"}
            enriched["ai_enrichment"] = {
                **enrichment_payload,
                **meta,
                "created_at": _now_iso(),
                "model": str(args.model) if not args.dry_run else "dry-run",
                "version": "preprocess_ai-enrich-v1",
                "max_event_chars": int(args.ai_max_event_chars),
                "max_output_tokens": int(args.ai_max_output_tokens),
            }

            # Housekeeping
            if not (enriched.get("ingested_at") or "").__str__().strip():
                enriched["ingested_at"] = _now_iso()

            write_jsonl_line(fout, enriched)
            total_written_ai += 1

            if total_written_ai > 0 and total_written_ai % 50 == 0:
                elapsed = time.time() - started
                rate = total_written_ai / elapsed if elapsed > 0 else 0.0
                print(
                    f"[Preprocess_AI] written={total_written_ai} openai_calls={total_openai_calls} "
                    f"cached_skipped={total_skipped_cached} rate={rate:.2f} ev/s"
                )

    print(f"[Preprocess_AI] Input:  {in_path}")
    if clean_out_path:
        print(f"[Preprocess_AI] Clean output: {clean_out_path}")
    if filtered_out_path:
        print(f"[Preprocess_AI] Filtered output: {filtered_out_path}")
    print(f"[Preprocess_AI] Output: {out_path}")
    print(f"[Preprocess_AI] Records read:        {total_read}")
    print(f"[Preprocess_AI] Records cleaned:     {total_cleaned}")
    if filtered_out_path:
        print(f"[Preprocess_AI] Records kept (filtered): {total_filtered_kept}")
    if total_written_filtered_out:
        print(f"[Preprocess_AI] Filtered-out written: {total_written_filtered_out}")
    print(f"[Preprocess_AI] Records written:     {total_written_ai}")
    print(f"[Preprocess_AI] Skipped (cached):    {total_skipped_cached}")
    if default_range_used:
        print(
            f"[Preprocess_AI] Date range:           default last {DEFAULT_MAX_AGE_DAYS} days "
            f"({from_dt.isoformat()} to {to_dt.isoformat()})"
        )
    else:
        print(
            f"[Preprocess_AI] Date range:           "
            f"{from_dt.isoformat() if from_dt else 'ANY'} to {to_dt.isoformat() if to_dt else 'ANY'}"
        )
    print(f"[Preprocess_AI] Filtered (date range): {total_filtered_old}")
    if not bool(args.disable_noise_filter):
        print(f"[Preprocess_AI] Filtered (noise):     {total_filtered_noise} (market/TA/prediction)")
        print(f"[Preprocess_AI] Filtered (duplicate urls): {duplicate_event_url_count}")
    else:
        print("[Preprocess_AI] Filtered (noise):    DISABLED")
    print(f"[Preprocess_AI] Dedupe: {'ON' if dedupe_enabled else 'OFF'}")
    print(f"[Preprocess_AI] OpenAI calls:        {total_openai_calls}")
    print(f"[Preprocess_AI] Fetch article:       {'ON' if fetch_cfg.fetch_article else 'OFF'}")
    print(f"[Preprocess_AI] Fetch contact pages: {'ON' if fetch_cfg.fetch_contact_pages else 'OFF'}")

    if bleach is None:
        print("[Preprocess_AI] NOTE: 'bleach' not installed. summary_html is plain-text escaped. Install bleach for HTML sanitization.")

    if not args.dry_run:
        tracker.print_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


