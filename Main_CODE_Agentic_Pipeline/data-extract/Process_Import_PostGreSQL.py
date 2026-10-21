#!/usr/bin/env python3
"""
Process_Import_PostGreSQL.py

Purpose
-------
1) Read an AI-enriched events JSONL file (jsonl rows output by your RSS/AI enrichment pipeline).
2) Transform each row into a DB-ready JSON object for `public.final_events` and write:
      ./Data/Final_events.jsonl
3) Import (UPSERT) that Final_events.jsonl into PostgreSQL table `public.final_events`.

Scope (per request)
-------------------
- ONLY touches `final_events`.
- Removes/avoids all other table/file references (final_categories, final_category_events, etc.).

Database Configuration

----------------------
Connection details are read from "Database_Connection.csv" in the same directory.
The CSV must have columns: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SCHEMA

Notes
-----
- No OpenAI calls.
- Uses psycopg2 + execute_batch for performance.

Typical usage
-------------
Process + Import (default):
  python Process_Import_PostGreSQL.py --input ./Data/events_ai_enriched.jsonl

Process only:
  python Process_Import_PostGreSQL.py --input ./Data/events_ai_enriched.jsonl --process-only

Import only (assumes ./Data/Final_events.jsonl exists):
  python Process_Import_PostGreSQL.py --import-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_batch, Json
from dateutil import parser as dtparser


# ==========================================================
# CONFIG: PostgreSQL connection from CSV file
# ==========================================================
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


def get_pg_config_from_csv() -> Dict[str, Any]:
    """Read PostgreSQL connection config from Database_Connection.csv"""
    script_dir = Path(__file__).resolve().parent
    config_file = script_dir / "Database_Connection.csv"
    
    try:
        config = _read_database_config(str(config_file))
    except Exception as e:
        print(f"[Process_Import] ERROR reading database config: {e}")
        raise
    
    return {
        "host": config["DB_HOST"],
        "port": int(config["DB_PORT"]),
        "dbname": config["DB_NAME"],
        "user": config["DB_USER"],
        "password": config["DB_PASSWORD"],
        "sslmode": "disable",
        "connect_timeout": 10,
    }


# ==========================================================
# Paths
# ==========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = SCRIPT_DIR / "Data"
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT = LOCAL_DATA_DIR / "events_ai_enriched.jsonl"
DEFAULT_FINAL_EVENTS = LOCAL_DATA_DIR / "Final_events.jsonl"


# ==========================================================
# JSONL read/write (robust)
# ==========================================================
def read_jsonl_robust(path: Path) -> Iterable[Dict[str, Any]]:
    """Tolerant JSONL reader that also handles occasional concatenated JSON objects."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    dec = JSONDecoder()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            # Fast path
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
                continue
            except json.JSONDecodeError:
                pass

            # Slow path: decode multiple JSON objects if concatenated
            idx = 0
            n = len(s)
            while idx < n:
                while idx < n and s[idx].isspace():
                    idx += 1
                if idx >= n:
                    break
                try:
                    obj, end = dec.raw_decode(s, idx)
                    if isinstance(obj, dict):
                        yield obj
                    idx = end
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Invalid JSON at {path}:{line_no} (concat decode): {e}") from e


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



# ==========================================================
# JSON safety helpers (keep all original event fields in output JSONL)
# ==========================================================
def make_json_safe(value: Any) -> Any:
    """Best-effort conversion to a JSON-serializable structure."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    # Fallback: stringify unknown objects (e.g., datetime)
    return str(value)

# ==========================================================
# Transformation: AI-enriched event -> DB row for final_events
# ==========================================================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def event_to_final_events_row(ev: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the output row for Final_events.jsonl.

    Requirement:
    - Keep ALL original event fields from the AI-enriched JSONL (including raw fields, full body, summary, etc.)
    - Also provide/overwrite standardized keys that the PostgreSQL importer expects.

    The importer will ignore unknown keys, so this preserves fidelity in Final_events.jsonl without impacting DB writes.
    """
    # Start from the full original event payload (json-safe copy)
    row: Dict[str, Any] = make_json_safe(ev)

    contact = ev.get("contact") if isinstance(ev.get("contact"), dict) else {}
    ai_cat = ev.get("ai_category") if isinstance(ev.get("ai_category"), dict) else {}

    # Prefer already-flattened fields but fall back safely.
    primary_id = ev.get("ai_primary_category_id") or ai_cat.get("primary_category_id")
    secondary_ids = ev.get("ai_secondary_category_ids") or ai_cat.get("secondary_category_ids") or []

    # Standardized keys used by the DB importer (set/overwrite for consistency)
    row["event_id"] = ev.get("event_id")
    row["opportunity_flag"] = ev.get("opportunity_flag")
    row["source"] = ev.get("source") or ev.get("source_name")
    row["title"] = ev.get("title_text") or ev.get("title_raw") or ev.get("title")
    row["url"] = ev.get("link") or ev.get("url")

    # Timestamps (keep as strings; importer parses to timestamptz)
    row["published_at"] = ev.get("published_at")
    # Prefer explicit ingested field; else fall back to created_at if present
    row["ingested_at"] = ev.get("ingested_at") or ev.get("created_at")

    # Author/contact
    row["author_name"] = ev.get("author_name") or contact.get("author_name")
    row["author_url"] = ev.get("author_url") or contact.get("author_url")

    # Summary/body fidelity: preserve what you already have, but add robust fallbacks
    row["summary_text"] = ev.get("summary_text") or ev.get("page_excerpt_raw") or ev.get("description_raw")
    row["summary_html"] = ev.get("summary_html") or ev.get("description_raw")
    row["body_text"] = ev.get("body_text") or ev.get("body_text_raw") or ev.get("content_raw")

    # Enrichment fields (kept both as flattened helpers and nested objects)
    row["ai_primary_category_id"] = int(primary_id) if primary_id is not None else None
    row["ai_secondary_category_ids"] = [int(x) for x in secondary_ids] if isinstance(secondary_ids, list) else []
    row["ai_category_confidence"] = safe_float(ev.get("ai_category_confidence") or ai_cat.get("confidence"), 0.0)
    row["ai_category_reasoning"] = ai_cat.get("reasoning")
    row["ai_category"] = ai_cat if isinstance(ai_cat, dict) else None
    row["ai_enrichment"] = ev.get("ai_enrichment")

    return make_json_safe(row)


# ==========================================================
# Importer: UPSERT into final_events
# ==========================================================
def parse_timestamptz(value: Any) -> Optional[str]:
    """Accepts ISO8601 string or None. Returns ISO string or None."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return dtparser.isoparse(value).isoformat()
        except Exception:
            return value
    return str(value)


def ensure_schema(cur) -> None:
    """Best-effort schema alignment for new pipeline fields."""
    cur.execute("ALTER TABLE final_events ADD COLUMN IF NOT EXISTS opportunity_flag text;")


UPSERT_EVENTS_SQL = """
INSERT INTO final_events (
  event_id,
  opportunity_flag,
  source,
  title,
  url,
  published_at,
  ingested_at,
  author_name,
  author_url,
  summary_text,
  summary_html,
  body_text,
  ai_category,
  ai_enrichment,
  updated_at
)
VALUES (
  %s, %s, %s, %s, %s,
  %s, %s,
  %s, %s,
  %s, %s,
  %s,
  %s, %s,
  now()
)
ON CONFLICT (event_id) DO UPDATE SET
  opportunity_flag = EXCLUDED.opportunity_flag,
  source = EXCLUDED.source,
  title = EXCLUDED.title,
  url = EXCLUDED.url,
  published_at = EXCLUDED.published_at,
  ingested_at = EXCLUDED.ingested_at,
  author_name = EXCLUDED.author_name,
  author_url = EXCLUDED.author_url,
  summary_text = EXCLUDED.summary_text,
  summary_html = EXCLUDED.summary_html,
  body_text = EXCLUDED.body_text,
  ai_category = EXCLUDED.ai_category,
  ai_enrichment = EXCLUDED.ai_enrichment,
  updated_at = now();
"""


def upsert_events(cur, rows: List[Dict[str, Any]], page_size: int = 500) -> int:
    params: List[Tuple[Any, ...]] = []
    for r in rows:
        event_id = (r.get("event_id") or "").strip()
        if not event_id:
            continue

        source = r.get("source") or r.get("source_name")
        opportunity_flag = r.get("opportunity_flag")
        title = r.get("title") or r.get("title_text") or r.get("title_raw")
        url = r.get("url") or r.get("link")

        published_at = parse_timestamptz(r.get("published_at"))
        ingested_at = parse_timestamptz(r.get("ingested_at"))

        author_name = r.get("author_name")
        author_url = r.get("author_url")

        summary_text = r.get("summary_text")
        summary_html = r.get("summary_html")
        body_text = r.get("body_text")

        ai_category = r.get("ai_category")
        ai_enrichment = r.get("ai_enrichment")

        params.append(
            (
                event_id,
                opportunity_flag,
                source,
                title,
                url,
                published_at,
                ingested_at,
                author_name,
                author_url,
                summary_text,
                summary_html,
                body_text,
                Json(ai_category) if ai_category is not None else None,
                Json(ai_enrichment) if ai_enrichment is not None else None,
            )
        )

    if params:
        try:
            # Get count of unique event_ids we're about to insert
            event_ids_to_insert = {p[0] for p in params}  # p[0] is event_id
            
            # Process in smaller chunks for better performance
            total_processed = 0
            for i in range(0, len(params), page_size):
                batch = params[i:i+page_size]
                print(f"[Process_Import] Processing batch {i//page_size + 1}: {len(batch)} rows")
                execute_batch(cur, UPSERT_EVENTS_SQL, batch, page_size=len(batch))
                total_processed += len(batch)
            
            print(f"[Process_Import] Upsert completed: {total_processed} rows processed")
            
            # Verify the data was actually inserted/updated
            placeholders = ','.join(['%s'] * len(event_ids_to_insert))
            verify_sql = f"SELECT COUNT(*) FROM public.final_events WHERE event_id IN ({placeholders})"
            cur.execute(verify_sql, tuple(event_ids_to_insert))
            verified_count = cur.fetchone()[0]
            
            if verified_count == len(event_ids_to_insert):
                print(f"[Process_Import] ✓ Verified: All {verified_count} events are in the database")
            else:
                print(f"[Process_Import] ⚠ WARNING: Expected {len(event_ids_to_insert)} events, but only {verified_count} found in database")
                print(f"[Process_Import] This may indicate silent failures or duplicate event_ids in input")
                
        except psycopg2.IntegrityError as e:
            print(f"[Process_Import] ❌ ERROR: Integrity constraint violation")
            print(f"  Details: {e}")
            print(f"  This usually means: duplicate primary keys, foreign key violations, or constraint violations")
            raise
        except psycopg2.DatabaseError as e:
            print(f"[Process_Import] ❌ ERROR: Database error during upsert")
            print(f"  Details: {e}")
            raise
        except Exception as e:
            print(f"[Process_Import] ❌ ERROR: Unexpected error during upsert ({type(e).__name__})")
            print(f"  Details: {e}")
            raise
    return len(params)


# ==========================================================
# Pipeline steps
# ==========================================================
def process_ai_enriched_to_final_events(in_path: Path, out_path: Path) -> int:
    events: List[Dict[str, Any]] = list(read_jsonl_robust(in_path))
    out_rows = [event_to_final_events_row(ev) for ev in events]
    write_jsonl(out_path, out_rows)
    return len(out_rows)


def import_final_events_to_postgres(final_events_path: Path, page_size: int = 500) -> int:
    cfg = get_pg_config_from_csv()
    try:
        conn = psycopg2.connect(
            host=cfg["host"],
            port=cfg["port"],
            dbname=cfg["dbname"],
            user=cfg["user"],
            password=cfg["password"],
            sslmode=cfg["sslmode"],
            connect_timeout=cfg["connect_timeout"],
        )
    except psycopg2.OperationalError as e:
        print("[Process_Import] ERROR: PostgreSQL connection failed.")
        print(f"  host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} user={cfg['user']}")
        print(f"  Details: {e}")
        print("  Hint: verify Database_Connection.csv has correct credentials.")
        return 0

    print(f"[Process_Import] Connected to database: host={cfg['host']} port={cfg['port']} dbname={cfg['dbname']} user={cfg['user']}")
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            ensure_schema(cur)
            conn.commit()

            rows = list(read_jsonl_robust(final_events_path))
            n = upsert_events(cur, rows, page_size=page_size)
            conn.commit()
            return n
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print("[Process_Import] ERROR: Failed to insert/upsert data into database.")
        print(f"  Details: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        conn.close()


# ==========================================================
# Main
# ==========================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Process AI-enriched events JSONL and import into PostgreSQL final_events")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help=f"Input AI-enriched events JSONL (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", default=str(DEFAULT_FINAL_EVENTS), help=f"Output Final_events.jsonl (default: {DEFAULT_FINAL_EVENTS})")
    ap.add_argument("--process-only", action="store_true", help="Only process input -> Final_events.jsonl (no DB import)")
    ap.add_argument("--import-only", action="store_true", help="Only import Final_events.jsonl to PostgreSQL (no processing)")
    ap.add_argument("--page-size", type=int, default=500, help="PostgreSQL execute_batch page size")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if args.process_only and args.import_only:
        print("[Process_Import] ERROR: choose only one of --process-only or --import-only.")
        return 2

    if not args.import_only:
        if not in_path.exists():
            print(f"[Process_Import] ERROR: input not found: {in_path}")
            return 2
        t0 = time.time()
        n = process_ai_enriched_to_final_events(in_path, out_path)
        print(f"[Process_Import] Processed {n} rows -> {out_path} ({time.time()-t0:.2f}s)")

    if not args.process_only:
        if not out_path.exists():
            print(f"[Process_Import] ERROR: Final_events.jsonl not found: {out_path}")
            return 2
        t1 = time.time()
        n2 = import_final_events_to_postgres(out_path, page_size=args.page_size)
        print(f"[Process_Import] Imported {n2} rows into public.final_events ({time.time()-t1:.2f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
