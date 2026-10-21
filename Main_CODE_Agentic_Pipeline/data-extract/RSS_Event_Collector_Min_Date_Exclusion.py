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
import hashlib
import json
import logging
import os
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# ---------------- HTTP Headers (To Fix 406 errors - 40+ sources have this Error) ----------------

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
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
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
    for_html: bool = False,
) -> requests.Response:
    """GET a URL; if blocked with 403, retry once with more browser-like headers."""
    resp = session.get(url, timeout=timeout, headers=headers)
    if resp.status_code != 403:
        return resp

    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").lower()
    if _is_targeted_forum_host(host):
        targeted_headers = _build_targeted_forum_headers(url, headers, for_html=for_html)
        targeted_resp = session.get(url, timeout=timeout, headers=targeted_headers)
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

    return session.get(url, timeout=timeout, headers=alt_headers)


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
    for_html: bool = False,
) -> requests.Response:
    """Fetch a URL with URL-aware headers and retries for 403/406 style rejections."""
    headers = build_request_headers(url, for_html=for_html)
    resp = request_with_403_fallback(session, url, timeout=timeout, headers=headers, for_html=for_html)
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
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    return session.get(url, timeout=timeout, headers=final_headers)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# ---------------- Paths ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = BASE_DIR / "Web3_rss_sources.json"   # keep existing name to avoid breaking setup
GITHUB_API_SOURCES_PATH = BASE_DIR / "github_api_sources.json"
EVENTS_JSON_PATH = DATA_DIR / "events.json"
EVENTS_JSONL_PATH = DATA_DIR / "events.jsonl"


# ---------------- Logging ----------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RSS_Event_Collector")


UTC = timezone.utc


# ---------------- Helpers ----------------

def _dt_from_parsed_tuple(t: Optional[Tuple]) -> Optional[datetime]:
    if not t:
        return None
    try:
        return datetime(*t[:6], tzinfo=UTC)
    except Exception:
        return None


def parse_published_at(entry: dict) -> Tuple[str, str]:
    """
    Return (published_at_iso_utc, published_at_source).
    Prefer feedparser parsed tuples. Fall back to parsing strings. Final fallback = now.
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

    return datetime.now(UTC).isoformat(), "fallback_now"


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


def fetch_article_title_and_excerpt(session: requests.Session, url: str, timeout: int = 15, max_chars: int = 2000) -> Tuple[str, str]:
    """
    Optional: fetch article page HTML and extract title + excerpt.
    Note: This is best-effort and can be disabled for speed.
    """
    if not url:
        return "", ""
    try:
        #resp = fetch_with_fallbacks(session, url, timeout=timeout, for_html=True)
        resp = fetch_with_fallbacks(session, url, timeout=timeout, for_html=True)
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


def fetch_article_full_body(session: requests.Session, url: str, timeout: int = 15, max_chars: int = 15000) -> str:
    """
    Fetch article page HTML and extract full body text up to max_chars (or whole body).
    Extracts paragraphs, headings, list items, and other text content.
    Best-effort and can be slower; disable for speed.
    """
    if not url:
        return ""
    try:
        resp = fetch_with_fallbacks(session, url, timeout=timeout, for_html=True)
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
        with config_path.open("r", encoding="utf-8") as f:
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


def load_github_api_sources(path: Path) -> list[dict]:
    """Load GitHub API sources from github_api_sources.json.

    Supported formats:
    - list[dict]
    - {"sources": list[dict]}

    Each source dict should include:
    - source_name (or name)
    - api_url (or url)  # GitHub REST API endpoint returning JSON list
    Optional:
    - enabled (default True)
    """
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("sources"), list):
        items = data["sources"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unexpected github_api_sources.json structure at {path}")

    out: list[dict] = []
    for s in items:
        if not isinstance(s, dict):
            continue
        if s.get("enabled", True) is False:
            continue

        source_name = (s.get("source_name") or s.get("name") or "").strip()
        api_url = (s.get("api_url") or s.get("url") or "").strip()
        if not source_name or not api_url:
            continue

        out.append({**s, "source_name": source_name, "api_url": api_url})
    return out


def fetch_github_api_json(
    session: requests.Session,
    api_url: str,
    *,
    token: str | None = None,
    timeout: int = 30,
):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "rss-event-collector",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = session.get(api_url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def resolve_github_item_timestamp(item: dict) -> str:
    """
    Choose one canonical timestamp for GitHub fallback items.
    This must stay consistent between cutoff filtering and saved published_at.
    """
    return (
        item.get("updated_at")
        or item.get("created_at")
        or item.get("published_at")
        or ""
    )


def github_item_to_event(
    *,
    source_name: str,
    api_url: str,
    item: dict,
    canonical_published_at: Optional[str] = None,
    fetch_html: bool,
    session: requests.Session,
    body_max_chars: int,
) -> dict:
    """Convert a GitHub API item (release/issue/pr) into our canonical event dict."""

    title = (item.get("name") or item.get("title") or item.get("tag_name") or "").strip()
    link = (item.get("html_url") or "").strip()
    guid = str(item.get("id") or item.get("node_id") or link or title)

    published_at = canonical_published_at or resolve_github_item_timestamp(item)

    page_title = ""
    page_excerpt = ""
    body_text_raw = ""
    if fetch_html and link:
        page_title, page_excerpt = fetch_article_title_and_excerpt(session, link)
        body_text_raw = fetch_article_full_body(session, link, max_chars=body_max_chars)

    description_raw = (item.get("body") or item.get("description") or "").strip()

    event_id = stable_event_id(
        source_name=source_name,
        guid=guid,
        link=link or api_url,
        published_at=published_at,
        title=title,
    )

    return {
        "event_id": event_id,
        "source_name": source_name,
        "feed_url": api_url,
        "link": link,
        "guid": guid,
        "published_at": published_at,
        "published_at_source": "github_api",
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
            "published": published_at,
        }
    }


def parse_dt_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = dateparser.parse(str(value))
        except Exception:
            dt = None
    if dt is None:
        return datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def github_atom_feed_to_api(feed_url: str) -> Optional[Dict[str, str]]:
    parsed = urllib.parse.urlparse(feed_url)
    if (parsed.netloc or '').lower() != 'github.com':
        return None

    parts = [p for p in parsed.path.strip('/').split('/') if p]
    if len(parts) != 3:
        return None

    owner, repo, suffix = parts
    owner = urllib.parse.quote(owner, safe='')
    repo = urllib.parse.quote(repo, safe='')

    if suffix == 'issues.atom':
        return {
            'kind': 'issues',
            'api_url': f'https://api.github.com/repos/{owner}/{repo}/issues?state=all&sort=updated&direction=desc&per_page=100',
        }
    if suffix == 'pulls.atom':
        return {
            'kind': 'pulls',
            'api_url': f'https://api.github.com/repos/{owner}/{repo}/pulls?state=all&sort=updated&direction=desc&per_page=100',
        }
    return None


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
) -> List[Dict[str, Any]]:
    mapping = github_atom_feed_to_api(feed_url)
    if not mapping:
        return []

    api_url = mapping['api_url']
    kind = mapping['kind']
    items = fetch_github_api_json(session, api_url, token=github_token, timeout=timeout)
    if not isinstance(items, list):
        return []

    if max_items > 0:
        items = items[:max_items]

    events: List[Dict[str, Any]] = []
    skipped_old = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        if kind == 'issues' and item.get('pull_request'):
            continue

        published_at = resolve_github_item_timestamp(item)
        if parse_dt_utc(published_at) < min_published_dt:
            skipped_old += 1
            continue

        event = github_item_to_event(
            source_name=source_name,
            api_url=feed_url,
            item=item,
            canonical_published_at=published_at,
            fetch_html=fetch_html,
            session=session,
            body_max_chars=body_max_chars,
        )
        event['published_at_source'] = 'github_api_fallback'
        events.append(event)

    logger.info(f'Collected {len(events)} items from {source_name} via GitHub API fallback')
    if skipped_old:
        logger.info(f'Skipped {skipped_old} items older than cutoff from {source_name}')
    return events


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
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    skipped_old = 0

    progress_str = f"[{progress_index}/{progress_total}] " if progress_total > 0 else ""
    logger.info(f"{progress_str}Processing RSS feed: {source_name} ({feed_url})")
    github_atom_mapping = github_atom_feed_to_api(feed_url)
    if github_atom_mapping:
        try:
            return collect_github_atom_events(
                session,
                source_name=source_name,
                feed_url=feed_url,
                max_items=max_items,
                fetch_html=fetch_html,
                min_published_dt=min_published_dt,
                timeout=timeout,
                body_max_chars=body_max_chars,
                github_token=github_token,
            )
        except requests.exceptions.Timeout:
            logger.warning(f"GitHub API fallback timeout for {source_name}: {feed_url} (timeout={timeout}s)")
            return events
        except requests.exceptions.RequestException:
            pass

    try:
        # Fetch feed with timeout to prevent hanging on unresponsive servers
        #resp = session.get(feed_url, timeout=timeout, headers={"User-Agent": "CalyxonAI-RSSCollector/1.0"})
        resp = fetch_with_fallbacks(session, feed_url, timeout=timeout)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except requests.exceptions.Timeout:
        logger.warning(f"Feed request timeout for {source_name}: {feed_url} (timeout={timeout}s)")
        return events
    except requests.exceptions.RequestException as e:
        logger.warning(f"Feed request error for {source_name}: {e}")
        return events
    except Exception as e:
        logger.exception(f"Feed parse error for {source_name}: {e}")
        return events

    entries = feed.entries or []
    if max_items > 0:
        entries = entries[:max_items]

    for entry in entries:
        title = str(entry.get("title") or "").strip()
        link = str(entry.get("link") or "").strip()
        guid = str(entry.get("id") or entry.get("guid") or "").strip()

        published_at, published_src = parse_published_at(entry)

        # Filter out older items early (saves HTML fetch time)
        try:
            pub_dt = datetime.fromisoformat(str(published_at).replace('Z', '+00:00'))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=UTC)
        except Exception:
            pub_dt = datetime.now(UTC)

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
            page_title, page_excerpt = fetch_article_title_and_excerpt(session, link)
            body_text_raw = fetch_article_full_body(session, link, max_chars=body_max_chars)

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
    ap = argparse.ArgumentParser(description="Collect RSS items as events (one per entry).")
    ap.add_argument("--output-format", choices=["json", "jsonl"], default="json", help="Output file format")
    ap.add_argument("--max-items-per-feed", type=int, default=100, help="Max items per feed (0 = no limit)")
    ap.add_argument("--dedupe", action="store_true", help="De-duplicate by event_id")
    ap.add_argument("--no-fetch-html", dest="fetch_html", action="store_false", help="Disable HTML fetching entirely (faster, RSS-only)")
    ap.add_argument("--fetch-1000", action="store_true", help="Fetch article body but limit to 1,000 chars (default: 15,000 chars)")
    ap.add_argument("--feed-timeout", type=int, default=30, help="Feed fetch timeout in seconds (default: 30)")
    ap.add_argument("--min-published-date", type=str, default=None, help="Only include items with published_at >= this date (YYYY-MM-DD). Default: 30 days ago (UTC).")

    ap.add_argument("--github-api-sources", type=str, default=str(GITHUB_API_SOURCES_PATH),
                    help="Path to github_api_sources.json (GitHub REST API endpoints). If missing, GitHub API ingestion is skipped.")
    ap.add_argument("--github-token", type=str, default=os.getenv("GITHUB_TOKEN", ""),
                    help="Optional GitHub token (or set env var GITHUB_TOKEN) to avoid rate limits.")
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

    sources = load_rss_sources(CONFIG_PATH)
    if not sources:
        logger.error("No RSS sources configured. Exiting.")
        return

    github_token = args.github_token.strip() or None
    all_events: List[Dict[str, Any]] = []
    with requests.Session() as session:
        configure_session(session)
        total_sources = len([s for s in sources if bool(s.get("enabled", True))])
        current_index = 0
        
        for src in sources:
            name = str(src.get("name") or "")
            url = str(src.get("url") or "")
            enabled = bool(src.get("enabled", True))
            source_type = str(src.get("source_type") or "")

            if not enabled:
                logger.info(f"Skipping disabled feed: {name} ({url})")
                continue

            current_index += 1

            all_events.extend(collect_feed_events(
                session=session,
                source_name=name,
                feed_url=url,
                source_type=source_type,
                max_items=args.max_items_per_feed,
                fetch_html=args.fetch_html,
                min_published_dt=min_published_dt,
                timeout=args.feed_timeout,
                body_max_chars=1000 if args.fetch_1000 else 15000,
                progress_index=current_index,
                progress_total=total_sources,
                github_token=github_token,
            ))

        # -----
        # GitHub API sources (JSON) ingestion
        # -----
        github_api_path = Path(args.github_api_sources)

        github_sources = []
        try:
            github_sources = load_github_api_sources(github_api_path)
        except Exception as e:
            logger.warning(f"Failed to load GitHub API sources from {github_api_path}: {e}")

        if github_sources:
            logger.info(f"Processing GitHub API sources: {len(github_sources)}")
            for i, s in enumerate(github_sources, start=1):
                source_name = s.get("source_name", "")
                api_url = s.get("api_url", "")
                if not source_name or not api_url:
                    continue

                logger.info(f"[GITHUB {i}/{len(github_sources)}] {source_name} ({api_url})")
                try:
                    data = fetch_github_api_json(session, api_url, token=github_token, timeout=args.feed_timeout)
                    if isinstance(data, dict) and isinstance(data.get("items"), list):
                        items = data["items"]
                    elif isinstance(data, list):
                        items = data
                    else:
                        logger.warning(f"Unexpected GitHub API response shape for {source_name}: {type(data)}")
                        continue

                    new_count = 0
                    old_count = 0
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        published_at = resolve_github_item_timestamp(item)
                        dt = parse_dt_utc(published_at) if published_at else None
                        if dt and dt < min_published_dt:
                            old_count += 1
                            continue

                        all_events.append(
                            github_item_to_event(
                                source_name=source_name,
                                api_url=api_url,
                                item=item,
                                canonical_published_at=published_at,
                                fetch_html=args.fetch_html,
                                session=session,
                                body_max_chars=1000 if args.fetch_1000 else 15000,
                            )
                        )
                        new_count += 1

                    logger.info(f"Collected {new_count} items from {source_name} (skipped {old_count} older than min date)")
                except Exception as e:
                    logger.warning(f"GitHub API request error for {source_name}: {e}")

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


if __name__ == "__main__":
    main()




