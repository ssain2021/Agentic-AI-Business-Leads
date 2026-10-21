#!/usr/bin/env python3
"""
RSS_Event_Collector.py

Purpose
------- collect_rss_events
Fetch RSS feeds from a configured list of sources and write a unified event stream.

This collector is intentionally "dumb":
- It does NOT try to detect funding/partnership/etc.
- It emits one record per RSS item (entry).
- It preserves raw RSS fields (often HTML) and adds minimal normalization.
- Optionally fetches full article body text (up to 15000 chars or whole body) for detailed content.
- Downstream steps (EventCleaner.py, EventCategorizerAI.py) perform cleaning + AI classification.

Outputs
-------
Default output: Data/events.json  (JSON array)
Optional output: Data/events.jsonl (JSON Lines)

Collected fields:
- event_id: stable hash-based ID
- source_name, feed_url, link, guid, published_at
- title_raw, description_raw, content_raw: raw RSS fields
- body_text_raw: full article body (up to 15000 chars or whole body, requires --fetch-html)
- page_title_raw, page_excerpt_raw: optional fetched page data
- created_at: collection timestamp

Config
------
Expects Web3_rss_sources.json in the same folder as this script.
Format:
[
  {"name": "CoinDesk", "url": "https://example.com/rss", "enabled": true},
  ...
]

Dependencies
------------
pip install feedparser requests beautifulsoup4 lxml python-dateutil

Usage
-----
python RSS_Event_Collector.py  (default: fetches HTML for full body text)
python RSS_Event_Collector.py --output-format jsonl --dedupe --max-items-per-feed 50
python RSS_Event_Collector.py --no-fetch-html  (skip HTML fetching for speed)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
import threading
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import feedparser
import requests
from requests.adapters import HTTPAdapter
import urllib3
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# ---------------- HTTP Headers (To Fix 406 errors - 40+ sources have this Error) ----------------

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

def configure_session(session: requests.Session) -> None:
    """Configure a requests Session with retries."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)


def session_get(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    headers: Dict[str, str],
    allow_insecure_ssl: bool,
) -> requests.Response:
    try:
        return session.get(url, timeout=timeout, headers=headers)
    except requests.exceptions.SSLError:
        if not allow_insecure_ssl:
            raise
        logger.warning(f"SSL error for {url}; retrying with certificate verification disabled.")
        return session.get(url, timeout=timeout, headers=headers, verify=False)


def _make_referer(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}/"
    except Exception:
        pass
    return ""


def _is_targeted_forum_host(host: str) -> bool:
    host = (host or "").lower()
    return host in {"dao.rocketpool.net", "forum.sky.money"}


def _build_targeted_forum_headers(url: str, headers: Dict[str, str], *, for_html: bool = False) -> Dict[str, str]:
    parsed = urllib.parse.urlparse(url)
    referer = _make_referer(url)
    targeted = dict(headers)
    targeted.update(
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
            if for_html else
            "application/rss+xml,application/xml;q=0.9,text/xml;q=0.8,text/html;q=0.7,*/*;q=0.6",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": DEFAULT_USER_AGENT,
        }
    )
    if parsed.scheme and parsed.netloc:
        targeted["Origin"] = f"{parsed.scheme}://{parsed.netloc}"
    if referer:
        targeted["Referer"] = referer
    return targeted


def request_with_403_fallback(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    headers: Dict[str, str],
    allow_insecure_ssl: bool,
    for_html: bool = False,
) -> requests.Response:
    """GET a URL; if blocked with 403, retry once with more browser-like headers."""
    resp = session_get(session, url, timeout=timeout, headers=headers, allow_insecure_ssl=allow_insecure_ssl)
    if resp.status_code != 403:
        return resp

    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").lower()
    if _is_targeted_forum_host(host):
        targeted_headers = _build_targeted_forum_headers(url, headers, for_html=for_html)
        targeted_resp = session_get(
            session,
            url,
            timeout=timeout,
            headers=targeted_headers,
            allow_insecure_ssl=allow_insecure_ssl,
        )
        if targeted_resp.status_code != 403:
            return targeted_resp

    referer = _make_referer(url)
    alt_headers = dict(headers)
    alt_headers.update(
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": alt_headers.get("Accept-Language", "en-US,en;q=0.9"),
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Upgrade-Insecure-Requests": "1",
        }
    )
    if referer:
        alt_headers["Referer"] = referer

    alt_headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
        "Gecko/20100101 Firefox/123.0"
    )

    return session_get(session, url, timeout=timeout, headers=alt_headers, allow_insecure_ssl=allow_insecure_ssl)


def build_request_headers(url: str, *, for_html: bool = False) -> Dict[str, str]:
    """Pick headers based on the target URL so GitHub Atom feeds do not return 406."""
    headers = dict(HEADERS)
    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    if for_html:
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        return headers

    if "github.com" in host:
        if path.endswith(".atom"):
            headers["Accept"] = "application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7"
            headers["Referer"] = "https://github.com/"
        else:
            headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

    return headers


def fetch_with_fallbacks(
    session: requests.Session,
    url: str,
    *,
    timeout: int,
    allow_insecure_ssl: bool,
    for_html: bool = False,
) -> requests.Response:
    """Fetch a URL with URL-aware headers and retries for 403/406 style rejections."""
    headers = build_request_headers(url, for_html=for_html)
    resp = request_with_403_fallback(
        session,
        url,
        timeout=timeout,
        headers=headers,
        allow_insecure_ssl=allow_insecure_ssl,
        for_html=for_html,
    )
    if resp.status_code not in (403, 406):
        return resp

    final_headers = dict(headers)
    final_headers.update(
        {
            "Accept": "application/atom+xml,application/rss+xml,application/xml,text/xml;q=0.9,text/html;q=0.8,*/*;q=0.7",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
        }
    )
    referer = _make_referer(url)
    if referer:
        final_headers["Referer"] = referer

    final_headers["User-Agent"] = (
        DEFAULT_USER_AGENT
    )
    return session_get(session, url, timeout=timeout, headers=final_headers, allow_insecure_ssl=allow_insecure_ssl)

HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}

# ---------------- Paths ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = BASE_DIR / "Web3_rss_sources.json"   # keep existing name to avoid breaking setup
EVENTS_JSON_PATH = DATA_DIR / "events.json"
EVENTS_JSONL_PATH = DATA_DIR / "events.jsonl"
FEED_HEALTH_REPORT_PATH = DATA_DIR / "feed_health_report.json"
FEED_DEBUG_SAMPLES_PATH = DATA_DIR / "feed_debug_samples.jsonl"


# ---------------- Logging ----------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RSS_Event_Collector")


UTC = timezone.utc
GITHUB_API_STATE_LOCK = threading.Lock()
GITHUB_API_DISABLED_REASON = ""
FEED_HEALTH_LOCK = threading.Lock()
FEED_HEALTH_RECORDS: List[Dict[str, Any]] = []
FEED_DEBUG_LOCK = threading.Lock()
FEED_DEBUG_SAMPLES: List[Dict[str, Any]] = []
GITHUB_API_REPO_ALIASES: Dict[Tuple[str, str], Tuple[str, str]] = {
    # Legacy GitHub path seen in the log; API fallback now lives on optimism.
    ("ethereum-optimism", "op-stack"): ("ethereum-optimism", "optimism"),
}


# ---------------- Helpers ----------------

def _dt_from_parsed_tuple(t: Optional[Tuple]) -> Optional[datetime]:
    if not t:
        return None
    try:
        return datetime(*t[:6], tzinfo=UTC)
    except Exception:
        return None


def parse_published_at(entry: dict) -> Tuple[Optional[str], str]:
    """
    Return (published_at_iso_utc, published_at_source).
    Prefer feedparser parsed tuples. Fall back to parsing strings.
    If no publish/update date can be parsed, return (None, "unparsed").
    """
    dt = _dt_from_parsed_tuple(entry.get("published_parsed")) or _dt_from_parsed_tuple(entry.get("updated_parsed"))
    if dt:
        return dt.astimezone(UTC).isoformat(), "feedparser_parsed"

    # Try common string fields
    for k in ("published", "updated", "pubDate", "date"):
        s = entry.get(k)
        if not s:
            continue
        try:
            parsed = dateparser.parse(str(s))
            if parsed is None:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC).isoformat(), f"dateutil:{k}"
        except Exception:
            continue

    return None, "unparsed"


def stable_event_id(source_name: str, guid: str, link: str, published_at: str, title: str) -> str:
    """
    Stable-ish unique ID for dedupe & downstream processing.
    """
    basis = "|".join([
        (source_name or "").strip().lower(),
        (guid or "").strip() or (link or "").strip(),
        (published_at or "").strip(),
        (title or "").strip()[:200],
    ])
    return hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()[:24]


def is_name_resolution_error(exc: Exception) -> bool:
    text = str(exc or "")
    return "NameResolutionError" in text or "getaddrinfo failed" in text or "Failed to resolve" in text


def classify_request_error(exc: Exception) -> Tuple[str, str]:
    text = str(exc or "")
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if is_name_resolution_error(exc):
        return "dns_error", "DNS resolution failed"
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout", f"Request timeout: {text}"
    if "SSLError" in text or "CERTIFICATE_VERIFY_FAILED" in text or "TLSV1_UNRECOGNIZED_NAME" in text:
        return "ssl_error", text
    if status_code == 404:
        return "not_found", "HTTP 404"
    if status_code == 403:
        return "forbidden", "HTTP 403"
    return "request_error", text


def merge_error_detail(*parts: str) -> str:
    cleaned = [str(part or "").strip() for part in parts if str(part or "").strip()]
    return " | ".join(cleaned)


def _truncate_text(value: Any, max_chars: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _redirect_chain(resp: Optional[requests.Response]) -> List[Dict[str, Any]]:
    if resp is None:
        return []
    chain: List[Dict[str, Any]] = []
    for item in list(getattr(resp, "history", []) or []) + [resp]:
        chain.append(
            {
                "status_code": int(getattr(item, "status_code", 0) or 0),
                "url": str(getattr(item, "url", "") or ""),
            }
        )
    return chain


def _response_preview(resp: Optional[requests.Response], max_chars: int = 500) -> str:
    if resp is None:
        return ""
    try:
        text = resp.text
    except Exception:
        try:
            text = resp.content.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    return _truncate_text(" ".join(str(text or "").split()), max_chars=max_chars)


def _diagnostic_flags(*, content_type: str, preview: str, final_url: str, feed_bozo: bool) -> List[str]:
    flags: List[str] = []
    ct = (content_type or "").lower()
    preview_l = (preview or "").lower()
    final_url_l = (final_url or "").lower()

    if "html" in ct:
        flags.append("content_type_html")
    if any(tag in ct for tag in ("rss", "atom", "xml", "text/xml", "application/xml")):
        flags.append("content_type_feedlike")
    if "cloudflare" in preview_l or "cf-ray" in preview_l:
        flags.append("cloudflare_marker")
    if "captcha" in preview_l:
        flags.append("captcha_marker")
    if "enable javascript" in preview_l or "javascript is required" in preview_l:
        flags.append("javascript_gate")
    if "access denied" in preview_l or "forbidden" in preview_l:
        flags.append("access_denied_text")
    if "attention required" in preview_l:
        flags.append("attention_required_text")
    if feed_bozo:
        flags.append("feed_bozo")
    if final_url_l and final_url_l.endswith((".html", "/")) and "xml" not in final_url_l and "rss" not in final_url_l and "feed" not in final_url_l:
        flags.append("redirected_non_feed_like_url")
    return flags


def classify_diagnostic_class(
    *,
    status: str,
    content_type: str = "",
    final_url: str = "",
    entry_count: int = 0,
    feed_bozo: bool = False,
    response_preview: str = "",
) -> str:
    if status == "ok":
        return "ok_rss"
    if status == "old_only":
        return "old_only_rss"
    if status == "unparsed_only":
        return "malformed_feed" if feed_bozo else "all_dates_unparsed"
    if status != "ok_zero":
        return status or "unknown"

    ct = (content_type or "").lower()
    preview_l = (response_preview or "").lower()
    final_url_l = (final_url or "").lower()

    if any(marker in preview_l for marker in ("captcha", "cloudflare", "attention required", "enable javascript", "access denied")):
        return "bot_block_html"
    if "html" in ct:
        return "html_instead_of_rss"
    if feed_bozo:
        return "malformed_feed"
    if entry_count == 0 and any(marker in ct for marker in ("rss", "atom", "xml", "text/xml", "application/xml")):
        return "empty_rss"
    if final_url_l and "xml" not in final_url_l and "rss" not in final_url_l and "feed" not in final_url_l and "html" in ct:
        return "redirected_non_feed"
    return "unknown_zero"


def record_feed_debug_sample(sample: Dict[str, Any]) -> None:
    if not sample:
        return
    with FEED_DEBUG_LOCK:
        FEED_DEBUG_SAMPLES.append(sample)


def record_feed_health(
    *,
    source_name: str,
    feed_url: str,
    source_type: str,
    status: str,
    collected_events: int,
    skipped_old: int,
    skipped_unparsed: int = 0,
    detail: str = "",
    content_type: str = "",
    final_url: str = "",
    entry_count: int = 0,
    feed_bozo: bool = False,
    diagnostic_class: str = "",
) -> None:
    with FEED_HEALTH_LOCK:
        FEED_HEALTH_RECORDS.append(
            {
                "source_name": source_name,
                "feed_url": feed_url,
                "source_type": source_type,
                "status": status,
                "collected_events": collected_events,
                "skipped_old": skipped_old,
                "skipped_unparsed": skipped_unparsed,
                "detail": detail,
                "content_type": content_type,
                "final_url": final_url,
                "entry_count": entry_count,
                "feed_bozo": feed_bozo,
                "diagnostic_class": diagnostic_class,
            }
        )


def build_feed_health_report(*, total_enabled_feeds: int, total_events: int) -> Dict[str, Any]:
    with FEED_HEALTH_LOCK:
        records = list(FEED_HEALTH_RECORDS)

    status_weights = {
        "dns_error": 100,
        "not_found": 95,
        "forbidden": 90,
        "ssl_error": 85,
        "timeout": 80,
        "github_api_unavailable": 75,
        "github_api_error": 70,
        "request_error": 65,
        "old_only": 30,
        "ok_zero": 20,
        "ok": 0,
    }

    summary = {
        "healthy_feeds": 0,
        "dead_or_error_feeds": 0,
        "low_value_feeds": 0,
    }

    ranked_records: List[Dict[str, Any]] = []
    for record in records:
        status = str(record.get("status") or "")
        collected_events = int(record.get("collected_events") or 0)
        skipped_old = int(record.get("skipped_old") or 0)
        skipped_unparsed = int(record.get("skipped_unparsed") or 0)
        if status == "ok" and collected_events > 0:
            summary["healthy_feeds"] += 1
            continue
        if status in {"ok_zero", "old_only"}:
            summary["low_value_feeds"] += 1
        else:
            summary["dead_or_error_feeds"] += 1

        ranked_record = dict(record)
        ranked_record["rank_score"] = status_weights.get(status, 50)
        ranked_record["note"] = (
            "No events and all matched items were older than the cutoff"
            if status == "old_only" else
            "No events and all matched items had unparseable publish dates"
            if status == "unparsed_only" else
            "No qualifying events collected"
            if status == "ok_zero" else
            str(record.get("detail") or "")
        )
        ranked_record["skipped_old"] = skipped_old
        ranked_record["skipped_unparsed"] = skipped_unparsed
        ranked_records.append(ranked_record)

    ranked_records.sort(
        key=lambda r: (
            -int(r.get("rank_score") or 0),
            int(r.get("collected_events") or 0),
            -int(r.get("skipped_old") or 0),
            -int(r.get("skipped_unparsed") or 0),
            str(r.get("source_name") or "").lower(),
        )
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_enabled_feeds": total_enabled_feeds,
        "total_events_saved": total_events,
        "summary": summary,
        "ranked_dead_or_low_value_feeds": ranked_records,
    }


def save_feed_health_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved feed health report to {path}")


def save_feed_debug_samples(path: Path) -> None:
    with FEED_DEBUG_LOCK:
        samples = list(FEED_DEBUG_SAMPLES)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} feed debug samples to {path}")


def clean_html_like(html: str) -> str:
    """
    Lightweight HTML -> text cleanup for collector-level excerpting.
    Full cleaning is done in EventCleaner.py; this is only to reduce noise in stored excerpts.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split()).strip()


class GitHubApiUnavailable(Exception):
    pass


def fetch_article_title_and_excerpt(
    session: requests.Session,
    url: str,
    *,
    timeout: int = 15,
    max_chars: int = 2000,
    allow_insecure_ssl: bool = False,
) -> Tuple[str, str]:
    """
    Optional: fetch article page HTML and extract title + excerpt.
    Note: This is best-effort and can be disabled for speed.
    """
    if not url:
        return "", ""
    try:
        resp = fetch_with_fallbacks(session, url, timeout=timeout, allow_insecure_ssl=allow_insecure_ssl, for_html=True)
        if resp.status_code >= 400:
            return "", ""
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        page_title = ""
        if soup.title and soup.title.string:
            page_title = soup.title.string.strip()

        # naive excerpt: collect paragraph text
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        excerpt = " ".join([p for p in paras if p])
        excerpt = " ".join(excerpt.split()).strip()
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip()

        return page_title, excerpt
    except Exception:
        return "", ""


def fetch_article_full_body(
    session: requests.Session,
    url: str,
    *,
    timeout: int = 15,
    max_chars: int = 15000,
    allow_insecure_ssl: bool = False,
) -> str:
    """
    Fetch article page HTML and extract full body text up to max_chars (or whole body).
    Extracts paragraphs, headings, list items, and other text content.
    Best-effort and can be slower; disable for speed.
    """
    if not url:
        return ""
    try:
        resp = fetch_with_fallbacks(session, url, timeout=timeout, allow_insecure_ssl=allow_insecure_ssl, for_html=True)
        if resp.status_code >= 400:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "noscript", "nav", "footer"]):
            tag.decompose()

        # Collect main content: paragraphs, headings, list items
        text_parts = []
        for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div"]):
            text = tag.get_text(" ", strip=True)
            if text and len(text.strip()) > 10:  # Skip very short fragments
                text_parts.append(text)

        body_text = " ".join(text_parts)
        body_text = " ".join(body_text.split()).strip()  # Normalize whitespace

        if len(body_text) > max_chars:
            body_text = body_text[:max_chars].rstrip()

        return body_text
    except Exception:
        return ""


def load_rss_sources(config_path: Path) -> List[Dict[str, Any]]:
    """Load RSS sources from JSON.

    Backwards compatible with the legacy format:
        [{"name": "...", "url": "..."}, ...]

    Extended format supported:
        {
          "name": "...",
          "url": "...",
          "enabled": true,
          "source_type": "media|governance|github_releases|...",
          "notes": "optional"
        }

    Only name+url are required. Missing 'enabled' defaults to True.
    """
    if not config_path.exists():
        logger.error(f"RSS sources config not found: {config_path}")
        return []
    try:
        with config_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("RSS sources config must be a JSON array.")
            return []

        sources: List[Dict[str, Any]] = []
        for obj in data:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name") or "").strip()
            url = str(obj.get("url") or "").strip()
            if not (name and url):
                continue

            enabled_val = obj.get("enabled", True)
            enabled = bool(enabled_val) if isinstance(enabled_val, (bool, int)) else str(enabled_val).strip().lower() not in {"false", "0", "no"}
            source_type = str(obj.get("source_type") or "").strip()
            notes = str(obj.get("notes") or "").strip()

            sources.append({
                "name": name,
                "url": url,
                "enabled": enabled,
                "source_type": source_type,
                "notes": notes,
            })
        return sources
    except Exception as e:
        logger.exception(f"Failed to load RSS sources: {e}")
        return []


def github_atom_feed_to_api(feed_url: str) -> Optional[Dict[str, str]]:
    parsed = urllib.parse.urlparse(feed_url)
    if (parsed.netloc or "").lower() != "github.com":
        return None

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) != 3:
        return None

    owner, repo, suffix = parts
    api_owner, api_repo = GITHUB_API_REPO_ALIASES.get((owner, repo), (owner, repo))
    quoted_owner = urllib.parse.quote(api_owner, safe="")
    quoted_repo = urllib.parse.quote(api_repo, safe="")
    source_repo = f"{owner}/{repo}"
    resolved_repo = f"{api_owner}/{api_repo}"

    if suffix == "issues.atom":
        return {
            "kind": "issues",
            "api_url": f"https://api.github.com/repos/{quoted_owner}/{quoted_repo}/issues?state=all&sort=updated&direction=desc&per_page=100",
            "source_repo": source_repo,
            "api_repo": resolved_repo,
        }
    if suffix == "pulls.atom":
        return {
            "kind": "pulls",
            "api_url": f"https://api.github.com/repos/{quoted_owner}/{quoted_repo}/pulls?state=all&sort=updated&direction=desc&per_page=100",
            "source_repo": source_repo,
            "api_repo": resolved_repo,
        }
    return None


def parse_dt_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = dateparser.parse(str(value))
        except Exception:
            dt = None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def resolve_github_item_timestamp(item: dict) -> str:
    return (
        item.get("updated_at")
        or item.get("created_at")
        or item.get("published_at")
        or ""
    )


def fetch_github_api_json(
    session: requests.Session,
    api_url: str,
    *,
    token: Optional[str] = None,
    timeout: int = 30,
):
    global GITHUB_API_DISABLED_REASON

    with GITHUB_API_STATE_LOCK:
        disabled_reason = GITHUB_API_DISABLED_REASON
    if disabled_reason:
        raise GitHubApiUnavailable(disabled_reason)

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": DEFAULT_USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = session.get(api_url, headers=headers, timeout=timeout)
    if resp.status_code in (403, 429):
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            reset_at = resp.headers.get("X-RateLimit-Reset")
            reason = "GitHub API rate limit reached"
            if reset_at and str(reset_at).isdigit():
                try:
                    reset_dt = datetime.fromtimestamp(int(reset_at), tz=UTC)
                    reason = f"GitHub API rate limit reached until {reset_dt.isoformat()}"
                except Exception:
                    pass
            with GITHUB_API_STATE_LOCK:
                GITHUB_API_DISABLED_REASON = reason
            raise GitHubApiUnavailable(reason)

    resp.raise_for_status()
    return resp.json()


def append_or_replace_query_param(url: str, key: str, value: Any) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(k, v) for k, v in query if k != key]
    filtered.append((key, str(value)))
    new_query = urllib.parse.urlencode(filtered)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment))


def github_item_to_event(
    *,
    source_name: str,
    feed_url: str,
    item: dict,
    canonical_published_at: str,
    fetch_html: bool,
    session: requests.Session,
    body_max_chars: int,
    allow_insecure_ssl: bool,
) -> Dict[str, Any]:
    title = (item.get("name") or item.get("title") or item.get("tag_name") or "").strip()
    link = (item.get("html_url") or "").strip()
    guid = str(item.get("id") or item.get("node_id") or link or title)
    description_raw = str(item.get("body") or item.get("description") or "")

    page_title = ""
    page_excerpt = ""
    body_text_raw = ""
    if fetch_html and link:
        page_title, page_excerpt = fetch_article_title_and_excerpt(
            session,
            link,
            allow_insecure_ssl=allow_insecure_ssl,
        )
        body_text_raw = fetch_article_full_body(
            session,
            link,
            max_chars=body_max_chars,
            allow_insecure_ssl=allow_insecure_ssl,
        )

    return {
        "event_id": stable_event_id(
            source_name=source_name,
            guid=guid,
            link=link or feed_url,
            published_at=canonical_published_at,
            title=title,
        ),
        "source_name": source_name,
        "feed_url": feed_url,
        "link": link,
        "guid": guid,
        "published_at": canonical_published_at,
        "published_at_source": "github_api_fallback",
        "title_raw": title,
        "description_raw": description_raw,
        "content_raw": "",
        "page_title_raw": page_title,
        "page_excerpt_raw": page_excerpt,
        "body_text_raw": body_text_raw,
        "created_at": datetime.now(UTC).isoformat(),
        "raw_entry": {
            "title": title,
            "link": link,
            "id": guid,
            "published": canonical_published_at,
        },
    }


def collect_github_atom_events(
    session: requests.Session,
    *,
    source_name: str,
    feed_url: str,
    max_items: int,
    fetch_html: bool,
    min_published_dt: datetime,
    timeout: int,
    body_max_chars: int,
    github_token: Optional[str],
    allow_insecure_ssl: bool,
) -> Tuple[List[Dict[str, Any]], int, int]:
    # Returns: (events, skipped_old, skipped_unparsed)
    mapping = github_atom_feed_to_api(feed_url)
    if not mapping:
        return [], 0, 0

    events: List[Dict[str, Any]] = []
    skipped_old = 0
    skipped_unparsed = 0
    page = 1
    while True:
        page_url = append_or_replace_query_param(mapping["api_url"], "page", page)
        items = fetch_github_api_json(session, page_url, token=github_token, timeout=timeout)
        if not isinstance(items, list) or not items:
            break

        page_had_newer_item = False
        for item in items:
            if not isinstance(item, dict):
                continue
            if mapping["kind"] == "issues" and item.get("pull_request"):
                continue

            published_at = resolve_github_item_timestamp(item)
            if not published_at:
                skipped_unparsed += 1
                continue
            published_dt = parse_dt_utc(published_at)
            if published_dt is None:
                skipped_unparsed += 1
                continue
            if published_dt < min_published_dt:
                skipped_old += 1
                continue

            page_had_newer_item = True
            events.append(
                github_item_to_event(
                    source_name=source_name,
                    feed_url=feed_url,
                    item=item,
                    canonical_published_at=published_at,
                    fetch_html=fetch_html,
                    session=session,
                    body_max_chars=body_max_chars,
                    allow_insecure_ssl=allow_insecure_ssl,
                )
            )
            if max_items > 0 and len(events) >= max_items:
                break

        if max_items > 0 and len(events) >= max_items:
            break
        if not page_had_newer_item:
            break
        page += 1

    logger.info(f"Collected {len(events)} items from {source_name} via GitHub API fallback")
    if skipped_old:
        logger.info(f"Skipped {skipped_old} items older than cutoff from {source_name}")
    if skipped_unparsed:
        logger.info(f"Skipped {skipped_unparsed} items with unparseable publish date from {source_name}")
    return events, skipped_old, skipped_unparsed


def entry_content_raw(entry: dict) -> str:
    """
    Extract HTML-ish content from a feedparser entry.
    Prefers content[0].value; falls back to summary/detail.
    """
    # content:encoded often lands here
    content = entry.get("content")
    if isinstance(content, list) and content:
        val = content[0].get("value")
        if val:
            return str(val)

    # summary/detail
    if entry.get("summary"):
        return str(entry.get("summary"))

    sd = entry.get("summary_detail") or {}
    if isinstance(sd, dict) and sd.get("value"):
        return str(sd.get("value"))

    # description sometimes exists
    if entry.get("description"):
        return str(entry.get("description"))

    return ""


def collect_feed_events(
    session: requests.Session,
    source_name: str,
    feed_url: str,
    source_type: str,
    max_items: int,
    fetch_html: bool,
    min_published_dt: datetime,
    timeout: int = 20,
    body_max_chars: int = 15000,
    progress_index: int = 0,
    progress_total: int = 0,
    github_token: Optional[str] = None,
    allow_insecure_ssl: bool = False,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    skipped_old = 0
    skipped_unparsed = 0
    github_fallback_detail = ""
    resp: Optional[requests.Response] = None

    progress_str = f"[{progress_index}/{progress_total}] " if progress_total > 0 else ""
    logger.info(f"{progress_str}Processing RSS feed: {source_name} ({feed_url})")
    github_atom_mapping = github_atom_feed_to_api(feed_url)

    if github_atom_mapping:
        try:
            if github_atom_mapping.get("api_repo") != github_atom_mapping.get("source_repo"):
                logger.info(
                    f"Using GitHub API repo override for {source_name}: "
                    f"{github_atom_mapping.get('source_repo')} -> {github_atom_mapping.get('api_repo')}"
                )
            events, skipped_old, skipped_unparsed = collect_github_atom_events(
                session,
                source_name=source_name,
                feed_url=feed_url,
                max_items=max_items,
                fetch_html=fetch_html,
                min_published_dt=min_published_dt,
                timeout=timeout,
                body_max_chars=body_max_chars,
                github_token=github_token,
                allow_insecure_ssl=allow_insecure_ssl,
            )
            record_feed_health(
                source_name=source_name,
                feed_url=feed_url,
                source_type=source_type,
                status="ok" if events else ("old_only" if skipped_old else ("unparsed_only" if skipped_unparsed else "ok_zero")),
                collected_events=len(events),
                skipped_old=skipped_old,
                skipped_unparsed=skipped_unparsed,
                detail="GitHub API fallback",
            )
            return events
        except GitHubApiUnavailable as e:
            logger.warning(f"GitHub API fallback unavailable for {source_name}: {e}")
            github_fallback_detail = f"GitHub API fallback unavailable: {e}"
            logger.info(f"Falling back to Atom feed for {source_name}")
        except requests.exceptions.Timeout:
            logger.warning(f"GitHub API fallback timeout for {source_name}: {feed_url} (timeout={timeout}s)")
            github_fallback_detail = f"GitHub API fallback timeout ({timeout}s)"
            logger.info(f"Falling back to Atom feed for {source_name}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"GitHub API fallback failed for {source_name}: {e}")
            status, detail = classify_request_error(e)
            github_fallback_detail = (
                f"GitHub API fallback {status}: {detail}"
                if status != "request_error" else
                f"GitHub API fallback failed: {detail}"
            )
            logger.info(f"Falling back to Atom feed for {source_name}")

    try:
        # Fetch feed with timeout to prevent hanging on unresponsive servers
        #resp = session.get(feed_url, timeout=timeout, headers={"User-Agent": "CalyxonAI-RSSCollector/1.0"})
        resp = fetch_with_fallbacks(
            session,
            feed_url,
            timeout=timeout,
            allow_insecure_ssl=allow_insecure_ssl,
        )
        if resp.status_code == 404:
            logger.warning(f"Feed not found (404) for {source_name}: {feed_url}")
            record_feed_health(
                source_name=source_name,
                feed_url=feed_url,
                source_type=source_type,
                status="not_found",
                collected_events=0,
                skipped_old=0,
                skipped_unparsed=0,
                detail=merge_error_detail(github_fallback_detail, "HTTP 404"),
            )
            return events
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except requests.exceptions.Timeout:
        logger.warning(f"Feed request timeout for {source_name}: {feed_url} (timeout={timeout}s)")
        record_feed_health(
            source_name=source_name,
            feed_url=feed_url,
            source_type=source_type,
            status="timeout",
            collected_events=0,
            skipped_old=0,
            skipped_unparsed=0,
            detail=merge_error_detail(github_fallback_detail, f"Feed timeout ({timeout}s)"),
        )
        return events
    except requests.exceptions.RequestException as e:
        status, detail = classify_request_error(e)
        if is_name_resolution_error(e):
            logger.warning(f"Feed DNS resolution failed for {source_name}: {feed_url}")
        elif getattr(getattr(e, "response", None), "status_code", None) == 404:
            logger.warning(f"Feed not found (404) for {source_name}: {feed_url}")
        else:
            logger.warning(f"Feed request error for {source_name}: {e}")
        record_feed_health(
            source_name=source_name,
            feed_url=feed_url,
            source_type=source_type,
            status=status,
            collected_events=0,
            skipped_old=0,
            skipped_unparsed=0,
            detail=merge_error_detail(github_fallback_detail, detail),
        )
        return events
    except Exception as e:
        logger.exception(f"Feed parse error for {source_name}: {e}")
        record_feed_health(
            source_name=source_name,
            feed_url=feed_url,
            source_type=source_type,
            status="request_error",
            collected_events=0,
            skipped_old=0,
            skipped_unparsed=0,
            detail=merge_error_detail(github_fallback_detail, f"Feed parse error: {e}"),
            content_type=str(getattr(getattr(resp, "headers", {}), "get", lambda *_: "")("Content-Type") or ""),
            final_url=str(getattr(resp, "url", "") or ""),
            entry_count=0,
            feed_bozo=False,
            diagnostic_class="request_error",
        )
        return events

    entries = feed.entries or []
    content_type = str(resp.headers.get("Content-Type") or "") if resp is not None else ""
    final_url = str(getattr(resp, "url", "") or "")
    entry_count = len(entries)
    feed_bozo = bool(getattr(feed, "bozo", 0))
    bozo_exception = _truncate_text(getattr(feed, "bozo_exception", "") or "", max_chars=500)
    response_preview = _response_preview(resp, max_chars=500)
    redirect_chain = _redirect_chain(resp)
    suspicion_flags = _diagnostic_flags(
        content_type=content_type,
        preview=response_preview,
        final_url=final_url,
        feed_bozo=feed_bozo,
    )
    if max_items > 0:
        entries = entries[:max_items]

    for entry in entries:
        title = str(entry.get("title") or "").strip()
        link = str(entry.get("link") or "").strip()
        guid = str(entry.get("id") or entry.get("guid") or "").strip()

        published_at, published_src = parse_published_at(entry)
        if not published_at:
            skipped_unparsed += 1
            continue

        # Filter out older items early (saves HTML fetch time)
        try:
            pub_dt = datetime.fromisoformat(str(published_at).replace('Z', '+00:00'))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=UTC)
        except Exception:
            skipped_unparsed += 1
            continue

        if pub_dt < min_published_dt:
            skipped_old += 1
            continue

        # RSS raw fields (often HTML)
        description_raw = str(entry.get("summary") or entry.get("description") or "")
        content_raw = entry_content_raw(entry)

        # Optional: fetch full article and store title + excerpt + full body text (best-effort)
        page_title = ""
        page_excerpt = ""
        body_text_raw = ""
        if fetch_html and link:
            page_title, page_excerpt = fetch_article_title_and_excerpt(
                session,
                link,
                allow_insecure_ssl=allow_insecure_ssl,
            )
            body_text_raw = fetch_article_full_body(
                session,
                link,
                max_chars=body_max_chars,
                allow_insecure_ssl=allow_insecure_ssl,
            )

        event_id = stable_event_id(
            source_name=source_name,
            guid=guid,
            link=link,
            published_at=published_at,
            title=title
        )

        events.append({
            "event_id": event_id,
            "source_name": source_name,
            "feed_url": feed_url,
            "link": link,
            "guid": guid,
            "published_at": published_at,
            "published_at_source": published_src,

            # Raw RSS payload (keep for traceability)
            "title_raw": title,
            "description_raw": description_raw,
            "content_raw": content_raw,

            # Optional fetched page data (may be empty)
            "page_title_raw": page_title,
            "page_excerpt_raw": page_excerpt,
            "body_text_raw": body_text_raw,  # Full article body text (up to 15000 chars or whole body)

            "created_at": datetime.now(UTC).isoformat(),

            # Original entry snapshot (small subset to avoid bloat)
            "raw_entry": {
                "title": title,
                "link": link,
                "id": guid,
                "published": entry.get("published"),
                "updated": entry.get("updated"),
            }
        })

    logger.info(f"Collected {len(events)} items from {source_name}")
    if skipped_old:
        logger.info(f"Skipped {skipped_old} items older than cutoff from {source_name}")
    if skipped_unparsed:
        logger.info(f"Skipped {skipped_unparsed} items with unparseable publish date from {source_name}")
    status = "ok" if events else ("old_only" if skipped_old else ("unparsed_only" if skipped_unparsed else "ok_zero"))
    diagnostic_class = classify_diagnostic_class(
        status=status,
        content_type=content_type,
        final_url=final_url,
        entry_count=entry_count,
        feed_bozo=feed_bozo,
        response_preview=response_preview,
    )
    record_feed_health(
        source_name=source_name,
        feed_url=feed_url,
        source_type=source_type,
        status=status,
        collected_events=len(events),
        skipped_old=skipped_old,
        skipped_unparsed=skipped_unparsed,
        detail=github_fallback_detail,
        content_type=content_type,
        final_url=final_url,
        entry_count=entry_count,
        feed_bozo=feed_bozo,
        diagnostic_class=diagnostic_class,
    )
    if status in {"ok_zero", "unparsed_only"} or feed_bozo or "html" in content_type.lower():
        record_feed_debug_sample(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "source_name": source_name,
                "feed_url": feed_url,
                "source_type": source_type,
                "status": status,
                "diagnostic_class": diagnostic_class,
                "content_type": content_type,
                "final_url": final_url,
                "entry_count": entry_count,
                "collected_events": len(events),
                "skipped_old": skipped_old,
                "skipped_unparsed": skipped_unparsed,
                "feed_bozo": feed_bozo,
                "bozo_exception": bozo_exception,
                "response_preview": response_preview,
                "redirect_chain": redirect_chain,
                "suspicion_flags": suspicion_flags,
                "detail": github_fallback_detail,
            }
        )
    return events


def chunked(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    if size <= 0:
        return [items[:]]
    return [items[i:i + size] for i in range(0, len(items), size)]


def process_rss_batch(
    batch_sources: List[Dict[str, Any]],
    *,
    batch_index: int,
    batch_total: int,
    max_items: int,
    fetch_html: bool,
    min_published_dt: datetime,
    timeout: int,
    body_max_chars: int,
    github_token: Optional[str],
    allow_insecure_ssl: bool,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    thread_name = threading.current_thread().name
    with requests.Session() as session:
        configure_session(session)
        logger.info(
            f"[RSS-BATCH {batch_index}/{batch_total}] Starting batch on thread {thread_name} "
            f"with {len(batch_sources)} feeds"
        )
        for local_index, src in enumerate(batch_sources, start=1):
            name = str(src.get("name") or "")
            url = str(src.get("url") or "")
            source_type = str(src.get("source_type") or "")
            events.extend(
                collect_feed_events(
                    session=session,
                    source_name=name,
                    feed_url=url,
                    source_type=source_type,
                    max_items=max_items,
                    fetch_html=fetch_html,
                    min_published_dt=min_published_dt,
                    timeout=timeout,
                    body_max_chars=body_max_chars,
                    progress_index=local_index,
                    progress_total=len(batch_sources),
                    github_token=github_token,
                    allow_insecure_ssl=allow_insecure_ssl,
                )
            )
        logger.info(
            f"[RSS-BATCH {batch_index}/{batch_total}] Finished batch on thread {thread_name} "
            f"with {len(events)} collected events"
        )
    return events


def dedupe_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for e in events:
        eid = e.get("event_id") or ""
        if not eid:
            continue
        if eid in seen:
            continue
        seen.add(eid)
        out.append(e)
    return out


def save_events_json(events: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(events)} events to {path}")


def save_events_jsonl(events: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(events)} events to {path}")


def main() -> None:
    global GITHUB_API_DISABLED_REASON, FEED_HEALTH_RECORDS, FEED_DEBUG_SAMPLES

    ap = argparse.ArgumentParser(description="Collect RSS items as events (one per entry).")
    ap.add_argument("--output-format", choices=["json", "jsonl"], default="json", help="Output file format")
    ap.add_argument("--max-items-per-feed", type=int, default=100, help="Max items per feed (0 = no limit)")
    ap.add_argument("--dedupe", action="store_true", help="De-duplicate by event_id")
    ap.add_argument("--no-fetch-html", dest="fetch_html", action="store_false", help="Disable HTML fetching entirely (faster, RSS-only)")
    ap.add_argument("--fetch-1000", action="store_true", help="Fetch article body but limit to 1,000 chars (default: 15,000 chars)")
    ap.add_argument("--feed-timeout", type=int, default=30, help="Feed fetch timeout in seconds (default: 30)")
    ap.add_argument("--rss-batch-size", type=int, default=20, help="Number of RSS feed URLs per worker batch (default: 20)")
    ap.add_argument("--worker-threads", type=int, default=3, help="Number of RSS worker threads (default: 3)")
    ap.add_argument("--min-published-date", type=str, default=None, help="Only include items with published_at >= this date (YYYY-MM-DD). Default: 30 days ago (UTC).")
    ap.add_argument("--github-token", type=str, default=os.getenv("GITHUB_TOKEN", ""), help="Optional GitHub token to improve issue/PR fallback coverage.")
    ap.add_argument("--allow-insecure-ssl", action="store_true", help="Retry SSL failures with certificate verification disabled (use with caution).")
    args = ap.parse_args()

    # Determine published-date cutoff (UTC). Default is 30 days ago at 00:00:00.
    if args.min_published_date:
        try:
            cutoff_date = datetime.strptime(args.min_published_date.strip(), "%Y-%m-%d").date()
        except Exception:
            raise SystemExit("--min-published-date must be in YYYY-MM-DD format")
        min_published_dt = datetime(cutoff_date.year, cutoff_date.month, cutoff_date.day, tzinfo=UTC)
    else:
        cutoff_date = (datetime.now(UTC) - timedelta(days=30)).date()
        min_published_dt = datetime(cutoff_date.year, cutoff_date.month, cutoff_date.day, tzinfo=UTC)
    logger.info(f"Using published-date cutoff (UTC): {min_published_dt.date().isoformat()}")
    with GITHUB_API_STATE_LOCK:
        GITHUB_API_DISABLED_REASON = ""
    with FEED_HEALTH_LOCK:
        FEED_HEALTH_RECORDS = []
    with FEED_DEBUG_LOCK:
        FEED_DEBUG_SAMPLES = []

    sources = load_rss_sources(CONFIG_PATH)
    if not sources:
        logger.error("No RSS sources configured. Exiting.")
        return

    github_token = args.github_token.strip() or None
    logger.info(f"GitHub API key read: {'Yes' if github_token else 'No'}")
    if github_token:
        logger.info("Authenticated GitHub API fallback enabled.")
    else:
        logger.info("GitHub API fallback may rate limit. Set GITHUB_TOKEN or --github-token if needed.")

    allow_insecure_ssl = bool(args.allow_insecure_ssl)
    if allow_insecure_ssl:
        urllib3.disable_warnings(InsecureRequestWarning)
    all_events: List[Dict[str, Any]] = []

    enabled_sources: List[Dict[str, Any]] = []
    for src in sources:
        name = str(src.get("name") or "")
        url = str(src.get("url") or "")
        enabled = bool(src.get("enabled", True))
        if not enabled:
            logger.info(f"Skipping disabled feed: {name} ({url})")
            continue
        enabled_sources.append(src)

    batch_size = max(1, int(args.rss_batch_size))
    worker_threads = max(1, int(args.worker_threads))
    body_max_chars = 1000 if args.fetch_1000 else 15000
    rss_batches = chunked(enabled_sources, batch_size)

    if rss_batches:
        logger.info(
            f"Processing {len(enabled_sources)} RSS feeds in {len(rss_batches)} batches "
            f"(batch_size={batch_size}, worker_threads={worker_threads})"
        )
        with ThreadPoolExecutor(max_workers=worker_threads, thread_name_prefix="rss") as executor:
            future_map = {
                executor.submit(
                    process_rss_batch,
                    batch,
                    batch_index=i,
                    batch_total=len(rss_batches),
                    max_items=args.max_items_per_feed,
                    fetch_html=args.fetch_html,
                    min_published_dt=min_published_dt,
                    timeout=args.feed_timeout,
                    body_max_chars=body_max_chars,
                    github_token=github_token,
                    allow_insecure_ssl=allow_insecure_ssl,
                ): i
                for i, batch in enumerate(rss_batches, start=1)
            }
            for future in as_completed(future_map):
                batch_num = future_map[future]
                try:
                    all_events.extend(future.result())
                except Exception as e:
                    logger.exception(f"RSS batch {batch_num} failed: {e}")

    if args.dedupe:
        all_events = dedupe_events(all_events)

    # Sort newest first
    def _key(e: Dict[str, Any]) -> str:
        return str(e.get("published_at") or "")

    all_events.sort(key=_key, reverse=True)

    if args.output_format == "jsonl":
        save_events_jsonl(all_events, EVENTS_JSONL_PATH)
    else:
        save_events_json(all_events, EVENTS_JSON_PATH)

    feed_health_report = build_feed_health_report(
        total_enabled_feeds=len(enabled_sources),
        total_events=len(all_events),
    )
    save_feed_health_report(feed_health_report, FEED_HEALTH_REPORT_PATH)
    save_feed_debug_samples(FEED_DEBUG_SAMPLES_PATH)

if __name__ == "__main__":
    main()




