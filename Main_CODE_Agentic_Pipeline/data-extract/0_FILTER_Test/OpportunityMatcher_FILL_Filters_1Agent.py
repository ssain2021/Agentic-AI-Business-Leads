#!/usr/bin/env python3

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import psycopg2
from psycopg2.extras import execute_batch
from openai import OpenAI


# =====================================================
# CONFIG
# =====================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

INPUT_CSV = DATA_DIR / "Unique_Opportunities_04_Feb_Published.csv"
FILTER_MASTER_JSON = SCRIPT_DIR / "filters_master.json"
DB_CONFIG_CSV = SCRIPT_DIR / "Database_Connection.csv"

MODEL = "gpt-5.1"
BATCH_SIZE = 5


# =====================================================
# ROBUST CSV READER (UTF-8 → CP1252 FALLBACK)
# =====================================================

def read_csv_with_fallback(path: Path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            print("Reading CSV using UTF-8...")
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        print("UTF-8 failed. Retrying with cp1252...")
        with open(path, newline="", encoding="cp1252") as f:
            return list(csv.DictReader(f))


# =====================================================
# TOKEN + COST TRACKER
# =====================================================

class TokenTracker:

    PRICING = {
        "gpt-5.1": (1.25, 10.00),  # input_per_1M, output_per_1M
    }

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.total_cost = 0.0

    def calculate_cost(self, input_tokens, output_tokens, model):
        pricing = self.PRICING.get(model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]
        return input_cost + output_cost

    def add_batch(self, usage, model):
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)

        batch_cost = self.calculate_cost(input_tokens, output_tokens, model)

        self.total_input += input_tokens
        self.total_output += output_tokens
        self.total_cost += batch_cost

        return input_tokens, output_tokens, batch_cost

    def print_summary(self):
        total_tokens = self.total_input + self.total_output

        print("\n" + "=" * 70)
        print("FINAL LLM USAGE SUMMARY (gpt-5.1)")
        print("=" * 70)
        print(f"Total input tokens:   {self.total_input:,}")
        print(f"Total output tokens:  {self.total_output:,}")
        print(f"Grand total tokens:   {total_tokens:,}")
        print(f"Total estimated cost: ${self.total_cost:.4f}")
        print("=" * 70)


# =====================================================
# DB CONFIG
# =====================================================

def load_db_config(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    return {
        "host": row["DB_HOST"].strip(),
        "port": int(row["DB_PORT"]),
        "user": row["DB_USER"].strip(),
        "password": row["DB_PASSWORD"].strip(),
        "database": row["DB_NAME"].strip(),
        "schema": row.get("DB_SCHEMA", "public").strip() or "public"
    }


def connect_db(cfg):
    conn = psycopg2.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {cfg['schema']}")
    return conn


# =====================================================
# FILTER MASTER
# =====================================================

def load_filter_master(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chains, sectors, seeking = {}, {}, {}

    for item in data.get("filters", []):
        if not item.get("enabled", False):
            continue

        fid = item["filter_text_id"]

        if item["filter_type"] == "chain":
            chains[fid] = item
        elif item["filter_type"] == "sector":
            sectors[fid] = item
        elif item["filter_type"] == "seeking":
            seeking[fid] = item

    return chains, sectors, seeking


# =====================================================
# OPENAI SCHEMA
# =====================================================

def filter_schema():
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
                            "row_index": {"type": "integer"},
                            "filter_chain": {"type": "array", "items": {"type": "string"}},
                            "filter_sector": {"type": "array", "items": {"type": "string"}},
                            "filter_seeking": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["row_index","filter_chain","filter_sector","filter_seeking"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        }
    }


# =====================================================
# FILTER AGENT CALL
# =====================================================

def call_filter_agent(client, payload, chains, sectors, seeking, tracker):

    allowed_chain_ids = list(chains.keys())
    allowed_sector_ids = list(sectors.keys())
    allowed_seeking_ids = list(seeking.keys())

    system_prompt = f"""
You are a strict filter classification agent.

Use ALL fields:
- title
- summary
- reason
- suggested_outreach_angle
- event_body_1
- event_body_2

Allowed IDs only.

filter_chain: {allowed_chain_ids}
filter_sector: {allowed_sector_ids}
filter_seeking: {allowed_seeking_ids}

Return JSON only.
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_schema", "json_schema": filter_schema()},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"opportunities": payload})}
        ],
        max_completion_tokens=2500,
    )

    input_tokens, output_tokens, batch_cost = tracker.add_batch(response.usage, MODEL)

    print(f"Batch usage → Input: {input_tokens:,} | Output: {output_tokens:,} | Cost: ${batch_cost:.4f}")

    return json.loads(response.choices[0].message.content)["results"]


# =====================================================
# UPSERT
# =====================================================

def normalize_tags(value: str) -> str:
    """
    Normalize tags: comma-separated, spaces replaced with hyphens.
    Handles regular CSV, Python-like list/set literals, and wrapped DB-style strings.
    """
    if not value or not isinstance(value, str):
        return ""

    source = value.strip()
    source = source.replace('""', '"')

    while len(source) >= 2 and ((source[0] == '"' and source[-1] == '"') or (source[0] == "'" and source[-1] == "'")):
        source = source[1:-1].strip()

    if len(source) >= 2 and ((source[0] == "[" and source[-1] == "]") or (source[0] == "{" and source[-1] == "}")):
        source = source[1:-1].strip()

    if not source:
        return ""

    if "','" in source or source.startswith("'"):
        parts = next(csv.reader([source], delimiter=",", quotechar="'", skipinitialspace=True))
    elif '","' in source or source.startswith('"'):
        parts = next(csv.reader([source], delimiter=",", quotechar='"', skipinitialspace=True))
    else:
        parts = [p.strip() for p in source.split(",")]

    cleaned: List[str] = []

    def _strip_wrappers(text: str) -> str:
        t = (text or "").strip()
        changed = True
        while changed and len(t) >= 2:
            changed = False
            if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
                t = t[1:-1].strip()
                changed = True
            if len(t) >= 2 and ((t.startswith("[") and t.endswith("]")) or (t.startswith("{") and t.endswith("}"))):
                t = t[1:-1].strip()
                changed = True
        return t

    for part in parts:
        item = _strip_wrappers(part)
        if not item:
            continue
        item = item.replace(" ", "-")
        if item:
            cleaned.append(item)

    return ",".join(cleaned)


def normalize_sources(value: str) -> str:
    """
    Normalize sources: clean up wrappers/quotes and return comma-separated values.
    Handles regular CSV, Python-like list/set literals, and wrapped DB-style strings.
    Examples:
    - "['source1','source2']" -> "source1,source2"
    - "{'source1','source2'}" -> "source1,source2"
    - "source1,source2" -> "source1,source2"
    - 'source1' -> "source1"
    """
    if not value or not isinstance(value, str):
        return ""
    
    source = value.strip()

    # Handle CSV-escaped double quotes, e.g. ""text""
    source = source.replace('""', '"')

    # Remove outer wrapping quotes repeatedly
    while len(source) >= 2 and ((source[0] == '"' and source[-1] == '"') or (source[0] == "'" and source[-1] == "'")):
        source = source[1:-1].strip()

    # Remove one level of container wrappers if present
    if len(source) >= 2 and ((source[0] == "[" and source[-1] == "]") or (source[0] == "{" and source[-1] == "}")):
        source = source[1:-1].strip()

    if not source:
        return ""

    # Parse comma-separated tokens while respecting quoted commas.
    if "','" in source or source.startswith("'"):
        parts = next(csv.reader([source], delimiter=",", quotechar="'", skipinitialspace=True))
    elif '","' in source or source.startswith('"'):
        parts = next(csv.reader([source], delimiter=",", quotechar='"', skipinitialspace=True))
    else:
        parts = [p.strip() for p in source.split(",")]

    cleaned: List[str] = []

    def _strip_wrappers(text: str) -> str:
        t = (text or "").strip()
        changed = True
        while changed and len(t) >= 2:
            changed = False
            if (t.startswith("\"") and t.endswith("\"")) or (t.startswith("'") and t.endswith("'")):
                t = t[1:-1].strip()
                changed = True
            if len(t) >= 2 and ((t.startswith("[") and t.endswith("]")) or (t.startswith("{") and t.endswith("}"))):
                t = t[1:-1].strip()
                changed = True
        return t

    for part in parts:
        item = _strip_wrappers(part)
        if not item:
            continue
        item = _strip_wrappers(item)
        if item:
            cleaned.append(item)

    return ",".join(cleaned)


def parse_time_found(value) -> str:
    """
    Parse time_found from CSV. If it's a string datetime, return as-is.
    If it's None/empty, return empty string.
    """
    if not value:
        return ""
    
    value_str = str(value).strip()
    if not value_str or value_str.lower() in ("none", "null", ""):
        return ""
    
    return value_str


def to_postgres_array(value: str) -> str:
    """
    Convert comma-separated string to PostgreSQL array format with quoted values.
    Example: "val1,val2,val3" -> {"val1","val2","val3"}
    If value is empty, return empty array {}.
    """
    if not value or not isinstance(value, str):
        return "{}"
    
    value = value.strip()
    if not value:
        return "{}"
    
    # Split and clean items
    items = [s.strip() for s in value.split(",") if s.strip()]
    
    if not items:
        return "{}"
    
    # Add double quotes around each item
    quoted_items = [f'"{item}"' for item in items]
    
    return "{" + ",".join(quoted_items) + "}"


def upsert_batch(conn, schema, rows):

    query = f"""
        INSERT INTO {schema}.opportunities_unique_filter (
            opportunity_id, title, summary, reason, who_to_contact,
            suggested_outreach_angle, categories,
            filter_chain, filter_sector, filter_seeking,
            time_found, confidence, tags, target_company,
            sources, event_ids, event_titles, event_url, bd_weeks
        )
        VALUES (
            %(opportunity_id)s, %(title)s, %(summary)s, %(reason)s, %(who_to_contact)s,
            %(suggested_outreach_angle)s, %(categories)s,
            %(filter_chain)s, %(filter_sector)s, %(filter_seeking)s,
            %(time_found)s, %(confidence)s, %(tags)s, %(target_company)s,
            %(sources)s, %(event_ids)s, %(event_titles)s, %(event_url)s, %(bd_weeks)s
        )
        ON CONFLICT (opportunity_id)
        DO UPDATE SET
            filter_chain = EXCLUDED.filter_chain,
            filter_sector = EXCLUDED.filter_sector,
            filter_seeking = EXCLUDED.filter_seeking
    """

    with conn.cursor() as cur:
        execute_batch(cur, query, rows)


# =====================================================
# MAIN
# =====================================================

def main():

    client = OpenAI()
    tracker = TokenTracker()

    chains, sectors, seeking = load_filter_master(FILTER_MASTER_JSON)
    db_cfg = load_db_config(DB_CONFIG_CSV)
    conn = connect_db(db_cfg)

    rows = read_csv_with_fallback(INPUT_CSV)
    print(f"Loaded {len(rows)} opportunities.")

    for i in range(0, len(rows), BATCH_SIZE):

        batch = rows[i:i+BATCH_SIZE]
        payload = []

        for idx, r in enumerate(batch):

            keys = list(r.keys())
            event_body_1 = r.get("event_body_1")
            event_body_2 = r.get("event_body_2")

            if event_body_1 is None and len(keys) >= 1:
                event_body_1 = r.get(keys[-1], "")
            if event_body_2 is None and len(keys) >= 2:
                event_body_2 = r.get(keys[-2], "")

            payload.append({
                "row_index": i + idx,
                "title": r.get("title",""),
                "summary": r.get("summary",""),
                "reason": r.get("reason",""),
                "suggested_outreach_angle": r.get("suggested_outreach_angle",""),
                "event_body_1": event_body_1 or "",
                "event_body_2": event_body_2 or ""
            })

        results = call_filter_agent(client, payload, chains, sectors, seeking, tracker)
        result_map = {r["row_index"]: r for r in results}

        db_rows = []

        for idx in range(i, i + len(batch)):
            row = rows[idx]
            r = result_map.get(idx)

            if r:
                row["filter_chain"] = ",".join(r["filter_chain"]) or "chain-oth"
                row["filter_sector"] = ",".join(r["filter_sector"]) or "sect-oth"
                row["filter_seeking"] = ",".join(r["filter_seeking"]) or "seek-oth"
            else:
                row["filter_chain"] = "chain-oth"
                row["filter_sector"] = "sect-oth"
                row["filter_seeking"] = "seek-oth"

            db_rows.append({
                **row,
                "tags": to_postgres_array(normalize_tags(row.get("tags", ""))),
                "sources": to_postgres_array(normalize_sources(row.get("sources", ""))),
                "categories": normalize_sources(row.get("categories", "")),
                "event_ids": to_postgres_array(row.get("event_ids", "")),
                "event_titles": to_postgres_array(row.get("event_titles", "")),
                "filter_chain": row.get("filter_chain", ""),
                "filter_sector": row.get("filter_sector", ""),
                "filter_seeking": row.get("filter_seeking", ""),
                "time_found": parse_time_found(row.get("time_found")),
                "confidence": float(row.get("confidence") or 0),
            })

        upsert_batch(conn, db_cfg["schema"], db_rows)
        conn.commit()
        time.sleep(1)

    tracker.print_summary()
    conn.close()
    print("Backfill complete → opportunities_unique_filter updated.")


if __name__ == "__main__":
    main()

    