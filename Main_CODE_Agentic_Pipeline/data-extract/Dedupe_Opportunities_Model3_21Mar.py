#!/usr/bin/env python3
"""
Deduplicate opportunities by Event ID (pipe-delimited) and write winners to opportunities_unique_filter.

UPDATED RULE (per your latest requirement)
------------------------------------------
- Dedupe key is the COMPLETE SET of event IDs found in event_ids (pipe-delimited).
- If the complete set repeats with different opportunities, keep ONE best one:
  1) higher confidence
  2) stronger BD-relevant fields
  3) lower bd_weeks (more immediate horizon)
  4) newer time_found
  5) lexicographically smaller opportunity_id (UUID string)

Notes
-----
- event_ids is treated as pipe-delimited ONLY, e.g. "ev1|ev2|ev3"
- The dedupe key is normalized as a set: tokens are trimmed, de-duped, sorted, and re-joined with '|'
  so "b|a|a" and "a|b" are treated as the same set.

Database Configuration
----------------------
Connection details are read from "Database_Connection.csv" in the same directory.
The CSV must have columns: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SCHEMA

Optional CLI
------------
--truncate-target   : TRUNCATE opportunities_unique_filter before writing
--dry-run           : do not write; only print counts
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import re

try:
    from openai import OpenAI
    from openai import APIError, APIConnectionError, RateLimitError, BadRequestError
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    APIError = APIConnectionError = RateLimitError = BadRequestError = Exception  # type: ignore


try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
except Exception:
    psycopg2 = None


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

        # Get the first row as config
        config_row = next(reader, None)
        if config_row is None:
            raise ValueError("Database config file has no data rows")

    # Validate required fields
    required_fields = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    for field in required_fields:
        if field not in config_row or not config_row[field]:
            raise ValueError(f"Missing or empty required field in database config: {field}")

    # Return config with defaults
    config = {
        "DB_HOST": config_row["DB_HOST"].strip(),
        "DB_PORT": config_row["DB_PORT"].strip(),
        "DB_USER": config_row["DB_USER"].strip(),
        "DB_PASSWORD": config_row["DB_PASSWORD"].strip(),
        "DB_NAME": config_row["DB_NAME"].strip(),
        "DB_SCHEMA": config_row.get("DB_SCHEMA", "public").strip() or "public",
    }

    return config


def _parse_event_ids_pipe_only(event_ids_text: Optional[str]) -> List[str]:
    """
    Parse event_ids from a TEXT column where multiple IDs are joined ONLY by '|'.
    Example: "id1|id2|id3"

    Returns a de-duplicated list of tokens preserving first-seen order.
    """
    s = (event_ids_text or "").strip()
    if not s:
        return []

    parts = [p.strip() for p in s.split("|")]
    parts = [p for p in parts if p]  # drop empty tokens

    # Preserve order but unique
    seen = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _normalize_event_ids_set(event_ids_text: Optional[str]) -> str:
    """
    Normalize event_ids as a SET so equivalent sets match deterministically.

    - Split on '|'
    - Trim whitespace
    - Drop empty tokens
    - De-duplicate tokens
    - Sort tokens (order-independent set semantics)
    - Re-join with '|'
    """
    toks = _parse_event_ids_pipe_only(event_ids_text)
    if not toks:
        return ""
    return "|".join(sorted(set(toks)))


def _parse_time_found(v: Any) -> datetime:
    """
    Normalize time_found for deterministic comparisons.
    If missing/unparseable: use epoch (1970-01-01 UTC).
    """
    if v is None:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)

    return datetime(1970, 1, 1, tzinfo=timezone.utc)


_CONTACT_HINT_RE = re.compile(
    r"\b(?:@[\w.\-]+|https?://|www\.|telegram|email|mailto:|discord|forum|github|linkedin|contact)\b",
    re.IGNORECASE,
)

_ACTION_HINT_RE = re.compile(
    r"\b(?:apply|submit|join|launch|integrat|contact|reach out|pilot|deploy|migrat|vendor|rfp|program|"
    r"onboard|intake|audit|evaluate|feedback|proposal)\b",
    re.IGNORECASE,
)


def _text_or_empty(v: Any) -> str:
    return str(v or "").strip()


def _bounded_length_score(text: str, cap: int) -> float:
    if not text:
        return 0.0
    return min(len(text), cap) / float(cap)


def _bd_relevance_strength_score(row: Dict[str, Any]) -> float:
    """
    Prefer the duplicate variant with richer BD-relevant content.

    This rewards rows that preserve clearer contact paths, outreach guidance,
    and more concrete actionable context. It is a tie-breaker after confidence,
    not a replacement for confidence.
    """
    title = _text_or_empty(row.get("title"))
    summary = _text_or_empty(row.get("summary"))
    reason = _text_or_empty(row.get("reason"))
    outreach = _text_or_empty(row.get("suggested_outreach_angle"))
    who_to_contact = _text_or_empty(row.get("who_to_contact"))
    details = _text_or_empty(row.get("opportunity_details"))

    score = 0.0
    score += 0.8 * _bounded_length_score(title, 160)
    score += 1.2 * _bounded_length_score(summary, 500)
    score += 1.0 * _bounded_length_score(reason, 500)
    score += 1.5 * _bounded_length_score(outreach, 320)
    score += 1.4 * _bounded_length_score(who_to_contact, 220)
    score += 0.9 * _bounded_length_score(details, 900)

    if outreach:
        score += 1.0
    if who_to_contact:
        score += 1.2
    if details:
        score += 0.4

    if _CONTACT_HINT_RE.search(who_to_contact):
        score += 1.0
    if _CONTACT_HINT_RE.search(outreach):
        score += 0.6
    if _CONTACT_HINT_RE.search(summary) or _CONTACT_HINT_RE.search(reason):
        score += 0.3

    combined = " ".join([title, summary, reason, outreach, who_to_contact, details])
    action_hits = len(set(m.group(0).lower() for m in _ACTION_HINT_RE.finditer(combined)))
    score += min(action_hits, 6) * 0.2

    return score



# ---------------------------------------------------------------------------
# LLM-based dedupe helpers (target_company-level dedupe across different events)
# ---------------------------------------------------------------------------

class TokenTracker:
    """Minimal token/cost tracker (mirrors the pattern in Preprocess_AI_GPT41.py)."""

    PRICING_USD_PER_1M = {
        # NOTE: update these if your OpenAI pricing changes.
        # Chosen to be consistent with your recent gpt-5.2 run cost.
        "gpt-5.2": {"input": 1.25, "output": 12.50},
        "gpt-5.1": {"input": 1.25, "output": 10.00},
        "gpt-4.1": {"input": 2.50, "output": 10.00},
    }

    def __init__(self) -> None:
        self.model_usage: Dict[str, Dict[str, int]] = {}

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        mu = self.model_usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
        mu["input_tokens"] += int(input_tokens or 0)
        mu["output_tokens"] += int(output_tokens or 0)

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING_USD_PER_1M.get(model)
        if not pricing:
            return 0.0
        return (input_tokens / 1_000_000.0) * pricing["input"] + (output_tokens / 1_000_000.0) * pricing["output"]

    def print_summary(self) -> None:
        if not self.model_usage:
            print("\n[LLM_DEDUPE] No OpenAI usage.\n")
            return

        print("\n" + "=" * 72)
        print("[LLM_DEDUPE] MODEL-WISE USAGE SUMMARY")
        print("=" * 72)
        total_in = total_out = 0
        total_cost = 0.0
        for model, usage in self.model_usage.items():
            inp = usage["input_tokens"]
            out = usage["output_tokens"]
            cost = self.estimate_cost(model, inp, out)
            total_in += inp
            total_out += out
            total_cost += cost
            print(f"\nModel: {model}")
            print("-" * 72)
            print(f"  Input tokens:   {inp:,}")
            print(f"  Output tokens:  {out:,}")
            print(f"  Total tokens:   {(inp+out):,}")
            print(f"  Estimated cost: ${cost:,.4f}")

        print("\n" + "=" * 72)
        print("[LLM_DEDUPE] TOTAL USAGE SUMMARY")
        print("=" * 72)
        print(f"  Total input tokens:   {total_in:,}")
        print(f"  Total output tokens:  {total_out:,}")
        print(f"  Grand total tokens:   {(total_in+total_out):,}")
        print(f"  Total estimated cost: ${total_cost:,.4f}")
        print("=" * 72 + "\n")


def _normalize_company(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _is_retryable_openai_error(exc: Exception) -> bool:
    return isinstance(exc, (RateLimitError, APIConnectionError, APIError))


LLM_DEDUPE_SCHEMA = {
    "name": "company_level_dedupe",
    "schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "duplicate_ids": {"type": "array", "items": {"type": "string"}},
                        "keep_id": {"type": "string"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                        "merged_fields": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "reason": {"type": "string"},
                                "suggested_outreach_angle": {"type": "string"},
                            },
                            "required": ["title", "summary", "reason", "suggested_outreach_angle"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["duplicate_ids", "keep_id", "reason", "confidence", "merged_fields"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    },
    "strict": True,
}


def llm_company_level_dedupe(
    client: OpenAI,
    tracker: TokenTracker,
    opportunities: List[Dict[str, Any]],
    model_primary: str = "gpt-5.2",
    model_fallback: str = "gpt-5.1",
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Ask an LLM to dedupe opportunities for a single target_company group."""

    compact: List[Dict[str, Any]] = []
    for o in opportunities:
        compact.append(
            {
                "opportunity_id": str(o.get("opportunity_id") or o.get("id") or ""),
                "event_id": str(o.get("event_id") or ""),
                "target_company": str(o.get("target_company") or ""),
                "confidence": float(o.get("confidence") or 0),
                "bd_weeks": int(o.get("bd_weeks") or 0),
                "title": str(o.get("title") or o.get("opportunity_title") or ""),
                "summary": str(o.get("summary") or ""),
                "reason": str(o.get("reason") or ""),
                "suggested_outreach_angle": str(o.get("suggested_outreach_angle") or ""),
                "opportunity_title": str(o.get("opportunity_title") or o.get("title") or ""),
                "opportunity_description": str(o.get("opportunity_description") or o.get("description") or ""),
                "source": str(o.get("source") or o.get("event_source") or ""),
                "published_at": str(o.get("published_at") or o.get("event_published_at") or ""),
            }
        )

    system = (
        "You are a Web3 BD analyst performing final deduplication of business development opportunities.\n\n"

        "Your job is to identify opportunities that represent the SAME ACTIONABLE BD MOTION and keep only the best version.\n\n"

        "PRIMARY USE CASE � Cross-horizon duplicates:\n"
        "The same opportunity may have been generated multiple times across different BD horizon passes "
        "(e.g. 1-10 weeks, 1-20 weeks, 1-30 weeks) with slightly different titles or framing. "
        "These are duplicates if a BD person would send the same outreach email for both.\n\n"

        "SECONDARY USE CASE � Cross-source duplicates:\n"
        "The same real-world event may have been scraped from two different sources, generating two opportunities "
        "with different titles, event_ids, or source references. These are duplicates if the underlying "
        "action surface is identical.\n\n"

        "MATCHING RULE - use these 4 fields (moderate fuzzy match):\n"
        "- title\n"
        "- summary\n"
        "- reason\n"
        "- suggested_outreach_angle\n"
        "Mark duplicates only when these four fields are the same or near-paraphrases of the same action.\n"
        "If any field implies a different program, action, or audience, keep both.\n\n"

        "SAME ACTION TEST � mark as duplicate if ALL of the following are true:\n"
        "1) Same target_company (or same DAO / ecosystem entity, even if named slightly differently).\n"
        "2) Same action surface: the BD team would send the same outreach email, apply to the same program, "
        "or contact the same team about the same opportunity.\n"
        "3) Same time window: both opportunities are actionable in the same general period "
        "(do not deduplicate a 10-week and a 30-week opportunity if the earlier one has a genuine deadline "
        "and the later one is a different phase of the same program).\n\n"

        "DIFFERENT INITIATIVE � keep both if ANY of the following are true:\n"
        "- Different programs, products, grant rounds, or RFPs (even at the same company).\n"
        "- Different regions, timelines, or eligibility criteria.\n"
        "- A BD person would write different outreach emails for each.\n\n"

        "WINNER SELECTION � when duplicates are identified, keep the version that is:\n"
        "1) Highest confidence score (primary criterion).\n"
        "2) Clearest, most specific title and description (secondary criterion).\n"
        "3) Shortest BD horizon / most immediately actionable (tertiary criterion � prefer 10w over 20w over 30w if tied).\n\n"

        "WINNER ENFORCEMENT:\n"
        "- The keep_id must be the highest confidence among the duplicates.\n\n"

        "MERGE REPHRASE (PREFERRED):\n"
        "- If duplicates are found, you should include merged_fields with improved phrasing of the four fields.\n"
        "- Only use information already present across the duplicate items.\n"
        "- Do NOT add new facts, claims, or action steps.\n"
        "- If no safe merge is possible, omit merged_fields and explain why in the decision reason.\n\n"

        "OUTPUT RULES:\n"
        "- Return valid JSON only. No explanations outside the JSON.\n"
        "- For each duplicate group, output one decision object.\n"
        "- If no duplicates are found, return an empty decisions array.\n"
        "- Do not mark opportunities as duplicates speculatively. If in doubt, keep both.\n\n"

        "OUTPUT SCHEMA:\n"
        "{\n"
        "  \"decisions\": [\n"
        "    {\n"
        "      \"duplicate_ids\": [\"id_to_discard_1\", \"id_to_discard_2\"],\n"
        "      \"keep_id\": \"id_to_keep\",\n"
        "      \"reason\": \"one sentence explaining why these are duplicates and why this version was kept\",\n"
        "      \"confidence\": 0.0-1.0,\n"
        "      \"merged_fields\": {\n"
        "        \"title\": \"optional\",\n"
        "        \"summary\": \"optional\",\n"
        "        \"reason\": \"optional\",\n"
        "        \"suggested_outreach_angle\": \"optional\"\n"
        "      }\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    user = (
        "Deduplicate the following opportunities for the same target_company. "
        "Return JSON with 'decisions'. Each decision groups truly duplicate opportunities and selects one keep_id.\n\n"
        f"OPPORTUNITIES_JSON:\n{json.dumps(compact, ensure_ascii=False)}"
    )

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        for model in (model_primary, model_fallback):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_schema", "json_schema": LLM_DEDUPE_SCHEMA},
                )

                usage = getattr(resp, "usage", None)
                if usage:
                    tracker.add_usage(model, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))

                content = resp.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:
                last_exc = e
                if attempt >= max_retries or not _is_retryable_openai_error(e):
                    break

        if attempt < max_retries:
            time.sleep(1.0 * (attempt + 1))

    # Fail safe: return no decisions rather than crashing the whole dedupe
    print(f"[LLM_DEDUPE] WARNING: LLM dedupe failed; proceeding without company-level dedupe. Last error: {last_exc}")
    return {"decisions": []}

class DatabaseManager:
    def __init__(self, host: str, port: int, user: str, password: str, database: str, schema: str = "public"):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is not installed. Install with: pip install psycopg2-binary")

        self.schema = (schema or "public").strip()
        self.conn = psycopg2.connect(host=host, port=port, user=user, password=password, database=database)
        self.conn.autocommit = False
        self._set_search_path()

    def _set_search_path(self) -> None:
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SET search_path TO {self.schema}")
        except Exception:
            pass

    def fetch_all_opportunities(self) -> List[Dict[str, Any]]:
        q = f"""
            SELECT
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
            FROM {self.schema}.opportunities_filter
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q)
            return cur.fetchall()

    def truncate_unique(self, target_table: str = "opportunities_unique_filter") -> None:
        q = f"TRUNCATE TABLE {self.schema}.{target_table}"
        with self.conn.cursor() as cur:
            cur.execute(q)

    def upsert_unique(self, rows: List[Dict[str, Any]], target_table: str = "opportunities_unique_filter") -> int:
        if not rows:
            return 0

        q = f"""
            INSERT INTO {self.schema}.{target_table} (
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
            )
            VALUES %s
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
                time_found = EXCLUDED.time_found,
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

        values = [
            (
                r["opportunity_id"],
                r["title"],
                r.get("summary"),
                r.get("reason").replace("_", " ") if isinstance(r.get("reason"), str) else r.get("reason"),
                r.get("who_to_contact"),
                r.get("suggested_outreach_angle"),
                r.get("categories"),
                r.get("filter_chain"),
                r.get("filter_sector"),
                r.get("filter_seeking"),
                r.get("time_found"),
                r.get("confidence"),
                r.get("tags"),
                r.get("target_company"),
                r.get("sources"),
                r.get("event_ids"),
                r.get("event_titles"),
                r.get("event_url"),
                r.get("bd_weeks"),
                r.get("opportunity_details"),
            )
            for r in rows
        ]

        with self.conn.cursor() as cur:
            execute_values(cur, q, values, page_size=1000)

        return len(rows)

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def dedupe_by_full_event_ids_set(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Returns:
      - unique winner rows (deduped by COMPLETE normalized set of event_ids)
      - mapping: normalized_event_ids_set -> winning opportunity_id
    """
    winners: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        opp_id = str(r["opportunity_id"])
        conf = float(r.get("confidence") or 0.0)
        bd_strength = _bd_relevance_strength_score(r)
        bd_weeks = int(r.get("bd_weeks") or 999999)
        tf = _parse_time_found(r.get("time_found"))

        key = _normalize_event_ids_set(r.get("event_ids"))

        # If event_ids is blank, keep it isolated by its own surrogate key (no dedupe possible)
        if not key:
            key = f"__NO_EVENT_IDS_SET__::{opp_id}"

        cur = winners.get(key)
        if cur is None:
            winners[key] = r
            continue

        cur_conf = float(cur.get("confidence") or 0.0)
        cur_bd_strength = _bd_relevance_strength_score(cur)
        cur_bd_weeks = int(cur.get("bd_weeks") or 999999)
        cur_tf = _parse_time_found(cur.get("time_found"))
        cur_opp = str(cur["opportunity_id"])

        better = False
        if conf > cur_conf:
            better = True
        elif conf == cur_conf:
            if bd_strength > cur_bd_strength:
                better = True
            elif bd_strength == cur_bd_strength:
                if bd_weeks < cur_bd_weeks:
                    better = True
                elif bd_weeks == cur_bd_weeks:
                    if tf > cur_tf:
                        better = True
                    elif tf == cur_tf:
                        if opp_id < cur_opp:
                            better = True

        # Safety valve for near-identical confidence values: prefer the richer BD version.
        elif abs(conf - cur_conf) <= 0.01:
            if bd_strength > cur_bd_strength:
                better = True
            elif bd_strength == cur_bd_strength and bd_weeks < cur_bd_weeks:
                better = True
            elif bd_strength == cur_bd_strength and bd_weeks == cur_bd_weeks and tf > cur_tf:
                better = True
            elif bd_strength == cur_bd_strength and bd_weeks == cur_bd_weeks and tf == cur_tf:
                if opp_id < cur_opp:
                    better = True

        if better:
            winners[key] = r

    unique_rows = list(winners.values())

    # Stable output order (optional but useful)
    unique_rows.sort(
        key=lambda x: (
            -(float(x.get("confidence") or 0.0)),
            int(x.get("bd_weeks") or 999999),
            _parse_time_found(x.get("time_found")),
            str(x["opportunity_id"]),
        ),
        reverse=False,
    )

    key_to_opp: Dict[str, str] = {k: str(v["opportunity_id"]) for k, v in winners.items()}
    return unique_rows, key_to_opp


def main() -> int:
    print("[Deduper] Starting dedupe job...")

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-table", default="opportunities_unique_filter", help="Target table for final unique opportunities (default: opportunities_unique_filter)")
    ap.add_argument("--truncate-target", action="store_true", help="TRUNCATE target table before insert")
    ap.add_argument("--dry-run", action="store_true", help="Do not write to DB; only print counts")
    ap.add_argument("--llm-dedupe", action="store_true", default=True, help="Enable LLM company-level dedupe (default: ON)")
    ap.add_argument("--no-llm-dedupe", dest="llm_dedupe", action="store_false", help="Disable LLM company-level dedupe")
    ap.add_argument("--llm-model-primary", default="gpt-4.1", help="Primary model for LLM dedupe")
    ap.add_argument("--llm-model-fallback", default="gpt-5.1", help="Fallback model for LLM dedupe")
    ap.add_argument("--llm-max-companies", type=int, default=0, help="Optional cap: process only first N duplicate target_company groups (0=all)")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "Database_Connection.csv")

    try:
        config = _read_database_config(config_file)
    except Exception as e:
        print(f"[Deduper] ERROR reading database config: {e}", file=sys.stderr)
        return 1

    host = config["DB_HOST"]
    port = int(config["DB_PORT"])
    user = config["DB_USER"]
    password = config["DB_PASSWORD"]
    dbname = config["DB_NAME"]
    schema = config["DB_SCHEMA"]

    print(f"[Deduper] Connecting to DB host={host} port={port} db={dbname} schema={schema} user={user} ...")
    db = DatabaseManager(host, port, user, password, dbname, schema=schema)
    print(f"[Deduper] Connected to DB: host={host} port={port} db={dbname} schema={schema} user={user}")

    try:
        print(f"[Deduper] Fetching opportunities from {schema}.opportunities_filter ...")
        rows = db.fetch_all_opportunities()
        print(f"[Deduper] Fetched opportunities: {len(rows)}")

        unique_rows, set_map = dedupe_by_full_event_ids_set(rows)

        # -----------------------------------------------------------------
        # Additional company-level dedupe (across different event_id sets)
        # -----------------------------------------------------------------
        tracker = TokenTracker()
        if args.llm_dedupe:
            if OpenAI is None:
                raise RuntimeError("openai package not available in this environment")
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")

            client = OpenAI(api_key=api_key)

            # Group by normalized target_company
            groups = {}
            for r in unique_rows:
                tc = r.get("target_company") or ""
                k = _normalize_company(str(tc))
                if not k:
                    continue
                groups.setdefault(k, []).append(r)

            candidates = {k: v for k, v in groups.items() if len(v) > 1}
            print(f"[Deduper] Company-level LLM dedupe candidates (target_company appearing >1): {len(candidates)}")

            drop_ids = set()
            merged_applied = 0
            for k, items in candidates.items():
                # Optional safety cap
                if args.llm_max_companies and len(items) > args.llm_max_companies:
                    items = sorted(items, key=lambda x: float(x.get('confidence') or 0), reverse=True)[: args.llm_max_companies]

                id_to_row = {}
                for it in items:
                    oid = str(it.get("opportunity_id") or it.get("id") or "")
                    if oid:
                        id_to_row[oid] = it

                decision_obj = llm_company_level_dedupe(
                    client=client,
                    tracker=tracker,
                    opportunities=items,
                    model_primary=args.llm_model_primary,
                    model_fallback=args.llm_model_fallback,
                )

                for d in (decision_obj or {}).get('decisions', []):
                    dup_ids = [str(x) for x in d.get('duplicate_ids', []) if str(x)]
                    keep_id = str(d.get('keep_id') or '')
                    if not dup_ids or not keep_id:
                        continue

                    candidate_ids = set(dup_ids + [keep_id])
                    # Enforce confidence-first winner, then prefer the richer BD-ready row.
                    best_id = keep_id
                    best_conf = -1.0
                    best_bd_strength = -1
                    best_bd_weeks = 999999
                    best_time_found = datetime.min.replace(tzinfo=timezone.utc)
                    for oid in candidate_ids:
                        row = id_to_row.get(oid)
                        if row is None:
                            continue
                        conf = float(row.get("confidence") or 0)
                        bd_strength = _bd_relevance_strength_score(row)
                        bd_weeks = int(row.get("bd_weeks") or 999999)
                        time_found = _parse_time_found(row.get("time_found"))
                        if conf > best_conf:
                            best_conf = conf
                            best_bd_strength = bd_strength
                            best_bd_weeks = bd_weeks
                            best_time_found = time_found
                            best_id = oid
                        elif conf == best_conf:
                            if bd_strength > best_bd_strength:
                                best_conf = conf
                                best_bd_strength = bd_strength
                                best_bd_weeks = bd_weeks
                                best_time_found = time_found
                                best_id = oid
                            elif bd_strength == best_bd_strength:
                                if bd_weeks < best_bd_weeks:
                                    best_conf = conf
                                    best_bd_strength = bd_strength
                                    best_bd_weeks = bd_weeks
                                    best_time_found = time_found
                                    best_id = oid
                                elif bd_weeks == best_bd_weeks and time_found > best_time_found:
                                    best_conf = conf
                                    best_bd_strength = bd_strength
                                    best_bd_weeks = bd_weeks
                                    best_time_found = time_found
                                    best_id = oid
                        elif abs(conf - best_conf) <= 0.01 and bd_strength > best_bd_strength:
                            best_conf = conf
                            best_bd_strength = bd_strength
                            best_bd_weeks = bd_weeks
                            best_time_found = time_found
                            best_id = oid

                    merged = d.get("merged_fields") if isinstance(d, dict) else None
                    if merged and isinstance(merged, dict):
                        row = id_to_row.get(best_id)
                        if row is not None:
                            for field in ("title", "summary", "reason", "suggested_outreach_angle"):
                                val = merged.get(field)
                                if isinstance(val, str) and val.strip():
                                    row[field] = val.strip()
                            merged_applied += 1

                    for oid in candidate_ids:
                        if oid != best_id:
                            drop_ids.add(oid)

            if drop_ids:
                before = len(unique_rows)
                unique_rows = [r for r in unique_rows if str(r.get('opportunity_id') or r.get('id') or '') not in drop_ids]
                after = len(unique_rows)
                print(f"[Deduper] Company-level LLM dedupe removed {before-after} opportunities (kept {after}).")
            if merged_applied:
                print(f"[Deduper] Company-level LLM dedupe merged fields on {merged_applied} winners.")

            tracker.print_summary()

        print(f"[Deduper] Read rows: {len(rows)} from {schema}.opportunities_filter")
        print(f"[Deduper] Unique winners: {len(unique_rows)} candidates for {schema}.{args.target_table}")
        print(f"[Deduper] Distinct event-id SET keys with winners: {len(set_map)}")

        if args.dry_run:
            print("[Deduper] Dry-run enabled; no DB writes performed.")
            return 0

        if args.truncate_target:
            print(f"[Deduper] Truncating {schema}.{args.target_table} ...")
            db.truncate_unique(target_table=args.target_table)
            print(f"[Deduper] Truncate complete.")

        print(f"[Deduper] Upserting {len(unique_rows)} rows into {schema}.{args.target_table} ...")
        inserted = db.upsert_unique(unique_rows, target_table=args.target_table)

        print("[Deduper] Committing transaction ...")
        db.commit()

        print("[Deduper] Done.")
        return 0

    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        print(f"[Deduper] ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

