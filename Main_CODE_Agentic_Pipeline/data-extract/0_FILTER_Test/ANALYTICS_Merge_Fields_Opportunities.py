#!/usr/bin/env python3
"""
ANALYTICS_Merge_Fields_Oppor.py

Purpose
-------
Fetch all records from the opportunities_events_centric table and export them to a CSV file
with additional RSS source metadata columns.

Process
-------
1. Read database configuration from Database_Connection.csv
2. Connect to PostgreSQL database
3. Load RSS sources from Web3_rss_sources.json
4. Load events body text from Data/events_ai_enriched.jsonl
5. Fetch all opportunities from opportunities_events_centric table
6. Enrich records with RSS_source_name, RSS_source_type, RSS_event_body, and RSS_event_body2 columns
7. Write enriched data to CSV file in 0_TEMP folder

Database Configuration
----------------------
Connection details are read from "Database_Connection.csv" in the same directory.
The CSV must have columns: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SCHEMA

Output
------
CSV file: opportunities_with_rss_sources_YYYYMMDD_HHMMSS.csv
Location: C:/Sukhendu/UPWORK-WORK/Fiverr-CAM-Python/0_TEMP (or script directory if 0_TEMP doesn't exist)
Columns: All original opportunity fields + RSS_source_name + RSS_source_type + RSS_event_body + RSS_event_body2
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import re

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None


# ========== Path and Configuration ==========

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "Web3_rss_sources.json"

SCRIPT_DIR = Path(__file__).resolve().parent
FIXED_OUTPUT_DIR = Path(r"C:\Sukhendu\UPWORK-WORK\Fiverr-CAM-Python\0_TEMP")
OUTPUT_DIR = FIXED_OUTPUT_DIR if FIXED_OUTPUT_DIR.exists() else SCRIPT_DIR
EVENTS_FILE_PATH = BASE_DIR / "Data" / "events_ai_enriched.jsonl"


# ========== Logging Setup ==========

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ANALYTICS_Merge_Fields")


# ========== Events and JSON Loading ==========

def load_events_body_text(events_file_path: Path) -> Dict[str, str]:
    """Load event excerpt text from JSONL file.
    
    Creates a mapping of event_id to their page_excerpt_raw content.
    Returns a dictionary: event_id -> page_excerpt_raw
    """
    body_text_map: Dict[str, str] = {}
    
    if not events_file_path.exists():
        logger.warning(f"Events file not found: {events_file_path}")
        return body_text_map
    
    try:
        with events_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event_id = str(event.get("event_id") or "").strip()
                    page_excerpt = str(event.get("page_excerpt_raw") or "").strip()
                    if not page_excerpt:
                        for key in (
                            "body_text_raw",
                            "body_text",
                            "rss_summary_text",
                            "description_raw",
                            "content_raw",
                            "summary_html",
                        ):
                            candidate = str(event.get(key) or "").strip()
                            if candidate:
                                page_excerpt = candidate
                                break
                    if event_id and page_excerpt:
                        body_text_map[event_id] = page_excerpt
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(body_text_map)} event excerpts from {events_file_path}")
    except Exception as e:
        logger.exception(f"Failed to load events file: {e}")
    
    return body_text_map


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


# ========== Record Enrichment and CSV Writing ==========

def add_rss_source_columns(records: List[Dict[str, Any]], rss_sources: List[Dict[str, Any]], body_text_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Add five new columns to each record:
    - RSS_source_name: extracted from 'sources' column (split by ':')
    - RSS_source_type: looked up from Web3_rss_sources.json by matching name
    - RSS_event_body: excerpt from first event (matched by event_id)
    - RSS_event_body2: excerpt from second event (if exists), else empty
    
    Args:
        records: List of opportunity dictionaries
        rss_sources: List of RSS source dictionaries from Web3_rss_sources.json
        body_text_map: Dictionary mapping event_id to page_excerpt_raw content
    
    Returns:
        Enriched records with the new columns populated.
    """
    def _normalize_source_name(value: str) -> str:
        name = (value or "").strip()
        if name.startswith("['"):
            name = name[2:]
        if name.endswith("']"):
            name = name[:-2]
        name = name.strip().strip("'\"")
        name = name.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
        return name.strip()

    def _parse_event_ids(value: str) -> List[str]:
        """
        Parse event_ids from the opportunity record.
        Event IDs can be separated by pipe symbol ('|').
        Returns a list of event_id strings.
        """
        if not value:
            return []

        raw = str(value).strip()
        if not raw:
            return []

        # Split by pipe symbol ('|') as the primary separator
        event_ids = [eid.strip() for eid in raw.split("|") if eid.strip()]
        return event_ids

    # Build a lookup dictionary: normalized name -> source_type
    source_type_map: Dict[str, str] = {}
    for source in rss_sources:
        source_name = _normalize_source_name(str(source.get("name", "")))
        source_type = str(source.get("source_type", "")).strip()
        if source_name:
            source_type_map[source_name] = source_type
    
    enriched = []
    for record in records:
        # Create a copy to avoid modifying the original
        enriched_record = dict(record)
        
        # Extract RSS_source_name from 'sources' field by splitting on ':'
        sources_field = str(record.get("sources") or "").strip()
        rss_source_name = ""
        if sources_field and ":" in sources_field:
            # Split by ':' and take the first part
            rss_source_name = sources_field.split(":", 1)[0].strip()
        rss_source_name = _normalize_source_name(rss_source_name)
        
        # Look up RSS_source_type from the JSON map
        rss_source_type = source_type_map.get(rss_source_name, "")
        
        # Extract event IDs from 'event_ids' field and get body text for up to 2 events
        event_ids_field = str(record.get("event_ids") or "").strip()
        rss_event_body = ""
        rss_event_body2 = ""
        
        if event_ids_field:
            event_id_list = _parse_event_ids(event_ids_field)

            if len(event_id_list) > 0:
                rss_event_body = body_text_map.get(event_id_list[0], "")

            if len(event_id_list) > 1:
                rss_event_body2 = body_text_map.get(event_id_list[1], "")
        
        # Add the populated columns
        enriched_record["RSS_source_name"] = rss_source_name
        enriched_record["RSS_source_type"] = rss_source_type
        enriched_record["RSS_event_body"] = rss_event_body
        enriched_record["RSS_event_body2"] = rss_event_body2
        
        enriched.append(enriched_record)
    
    return enriched


def write_opportunities_to_csv(records: List[Dict[str, Any]], output_dir: Path = OUTPUT_DIR) -> str:
    """
    Write opportunities records to a CSV file with all fields including RSS source columns.
    
    Args:
        records: List of opportunity dictionaries
        output_dir: Directory where CSV should be written (defaults to OUTPUT_DIR)
    
    Returns:
        Path to the written CSV file
    """
    if not records:
        logger.warning("No records to write to CSV.")
        return ""
    
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_filename = f"opportunities_with_rss_sources_{ts}.csv"
    csv_path = output_dir / csv_filename
    
    try:
        # Get all fieldnames from the first record, maintaining order
        fieldnames = list(records[0].keys())
        
        # Ensure RSS columns are at the end: RSS_source_name, RSS_source_type, RSS_event_body, RSS_event_body2
        if "RSS_source_name" in fieldnames:
            fieldnames.remove("RSS_source_name")
        if "RSS_source_type" in fieldnames:
            fieldnames.remove("RSS_source_type")
        if "RSS_event_body" in fieldnames:
            fieldnames.remove("RSS_event_body")
        if "RSS_event_body2" in fieldnames:
            fieldnames.remove("RSS_event_body2")
        fieldnames.append("RSS_source_name")
        fieldnames.append("RSS_source_type")
        fieldnames.append("RSS_event_body")
        fieldnames.append("RSS_event_body2")
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(records)
        
        logger.info(f"Wrote {len(records)} records to: {csv_path}")
        return str(csv_path)
    
    except Exception as e:
        logger.exception(f"Failed to write CSV to {csv_path}: {e}")
        return ""


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
                time_found,
                confidence,
                tags,
                target_company,
                sources,
                event_ids,
                event_titles,
                event_url,
                bd_weeks
            FROM {self.schema}.opportunities_unique_old
        """
## TABLES: opportunities_events_centric, opportunities_unique_month, opportunities_unique_old
##  TABLES::  opportunities_unique_TEMP

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q)
            return cur.fetchall()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def main() -> int:
    print("[Analytics] Starting to fetch opportunities and export to CSV...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "Database_Connection.csv")

    try:
        config = _read_database_config(config_file)
    except Exception as e:
        print(f"[Analytics] ERROR reading database config: {e}", file=sys.stderr)
        return 1

    host = config["DB_HOST"]
    port = int(config["DB_PORT"])
    user = config["DB_USER"]
    password = config["DB_PASSWORD"]
    dbname = config["DB_NAME"]
    schema = config["DB_SCHEMA"]

    print(f"[Analytics] Connecting to DB host={host} port={port} db={dbname} schema={schema} user={user} ...")
    db = DatabaseManager(host, port, user, password, dbname, schema=schema)
    print(f"[Analytics] Connected to DB: host={host} port={port} db={dbname} schema={schema} user={user}")

    try:
        # Load RSS sources from JSON
        print(f"[Analytics] Loading RSS sources from {CONFIG_PATH} ...")
        rss_sources = load_rss_sources(CONFIG_PATH)
        print(f"[Analytics] Loaded {len(rss_sources)} RSS sources")

        # Load events excerpts (matched by event_id)
        print(f"[Analytics] Loading event excerpts from {EVENTS_FILE_PATH} ...")
        body_text_map = load_events_body_text(EVENTS_FILE_PATH)
        print(f"[Analytics] Loaded {len(body_text_map)} event excerpts")

        # Fetch opportunities from database
        print(f"[Analytics] Fetching opportunities from {schema}.opportunities_unique_old ...")
        rows = db.fetch_all_opportunities()
        print(f"[Analytics] Fetched opportunities: {len(rows)}")

        # Enrich records with RSS source columns and event excerpts
        print(f"[Analytics] Enriching records with RSS source information and event excerpts ...")
        enriched_rows = add_rss_source_columns(rows, rss_sources, body_text_map)
        
        # Write enriched opportunities to CSV
        print(f"[Analytics] Writing enriched opportunities to CSV ...")
        csv_path = write_opportunities_to_csv(enriched_rows, OUTPUT_DIR)
        if csv_path:
            print(f"[Analytics] CSV written to: {csv_path}")
            print(f"[Analytics] Total records exported: {len(enriched_rows)}")
        
        print("[Analytics] Done.")
        return 0

    except Exception as e:
        print(f"[Analytics] ERROR: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
