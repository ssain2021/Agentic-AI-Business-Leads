#!/usr/bin/env python3
"""
Read opportunities from CSV file and insert into opportunities_unique table.

The opportunities CSV file should be in the same directory as this script.
All records from the CSV are inserted directly without deduplication.

Database Configuration
----------------------
Connection details are read from "Database_Connection.csv" in the same directory.
The CSV must have columns: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SCHEMA
"""

from __future__ import annotations

import csv
import os
import sys
import codecs
from typing import Any, Dict, List, Optional
import ast
import uuid

try:
    import psycopg2
    from psycopg2.extras import execute_values
except Exception:
    psycopg2 = None


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


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


def _read_csv_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Read opportunities from a CSV file.
    
    Returns a list of dictionaries with one row per opportunity.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(filepath, "rb") as fb:
        prefix = fb.read(4)

    if prefix.startswith(codecs.BOM_UTF8):
        encodings = ("utf-8-sig", "cp1252")
    elif prefix.startswith(codecs.BOM_UTF16_LE) or prefix.startswith(codecs.BOM_UTF16_BE):
        encodings = ("utf-16",)
    else:
        encodings = ("utf-8", "cp1252")

    def _read_rows(encoding: str, errors: str = "strict") -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(filepath, "r", encoding=encoding, errors=errors, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty or has no headers")

            valid_fieldnames = [
                field.strip()
                for field in reader.fieldnames
                if field is not None and field.strip() != ""
            ]

            if not valid_fieldnames:
                raise ValueError("CSV file has no valid headers")

            for row_dict in reader:
                cleaned_row: Dict[str, Any] = {}
                for field in valid_fieldnames:
                    cleaned_row[field] = row_dict.get(field)
                if not any(
                    value is not None and str(value).strip() not in ("", "NULL")
                    for value in cleaned_row.values()
                ):
                    continue
                rows.append(cleaned_row)

        return rows

    attempted_errors = []
    for encoding in encodings:
        try:
            return _read_rows(encoding)
        except UnicodeDecodeError as exc:
            attempted_errors.append(f"{encoding}: {exc}")
            continue

    # Some manually edited / Excel-saved CSVs contain mixed UTF-8 and Windows bytes
    # in otherwise valid text fields. Fall back to permissive decoding so a few bad
    # punctuation characters do not block the whole import.
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            print(
                f"[CSV Inserter] Warning: falling back to permissive {encoding} decoding "
                f"for mixed-encoding CSV content."
            )
            return _read_rows(encoding, errors="replace")
        except Exception as exc:
            attempted_errors.append(f"{encoding} (replace): {exc}")
            continue

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        "Failed to decode CSV with supported encodings: "
        + ", ".join(attempted_errors if attempted_errors else ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252"]),
    )


def _normalize_pg_text_array_value(value: Any) -> Optional[List[str]]:
    if value is None:
        return None

    def _clean_items(items: List[Any]) -> Optional[List[str]]:
        cleaned = [str(item).strip() for item in items if str(item).strip() not in ("", "NULL")]
        return cleaned if cleaned else None

    if isinstance(value, (list, tuple, set)):
        return _clean_items(list(value))

    value_str = str(value).strip()
    if value_str == "" or value_str.upper() == "NULL":
        return None

    try:
        parsed = ast.literal_eval(value_str)
        if parsed is None:
            return None
        if isinstance(parsed, (list, tuple, set)):
            return _clean_items(list(parsed))
        parsed_str = str(parsed).strip()
        return [parsed_str] if parsed_str and parsed_str.upper() != "NULL" else None
    except Exception:
        # Fallback for malformed literals: split comma-separated strings and strip wrappers.
        parts = [
            part.strip().strip("'\"")
            for part in value_str.strip("[]{}").split(",")
            if part.strip() != ""
        ]
        cleaned = [part for part in parts if part.upper() != "NULL"]
        return cleaned if cleaned else None


def _normalize_uuid_value(value: Any, row: Dict[str, Any]) -> str:
    """
    Ensure the value is a valid UUID string.

    If the CSV contains a malformed or missing opportunity_id, generate a deterministic
    UUID from the best available row identity so imports remain stable across reruns.
    """
    value_str = "" if value is None else str(value).strip()
    if value_str:
        try:
            return str(uuid.UUID(value_str))
        except Exception:
            pass

    identity_parts = [
        str(row.get("event_url") or "").strip(),
        str(row.get("title") or "").strip(),
        str(row.get("event_ids") or "").strip(),
        str(row.get("time_found") or "").strip(),
    ]
    identity = " | ".join(part for part in identity_parts if part)
    if not identity:
        identity = value_str or str(uuid.uuid4())

    normalized = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))

    if value_str and value_str != normalized:
        print(
            f"[CSV Inserter] Warning: invalid opportunity_id '{value_str}' "
            f"normalized to '{normalized}' for title='{str(row.get('title') or '').strip()}'"
        )
    elif not value_str:
        print(
            f"[CSV Inserter] Warning: missing opportunity_id normalized to '{normalized}' "
            f"for title='{str(row.get('title') or '').strip()}'"
        )

    return normalized


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

    def insert_opportunities(self, rows: List[Dict[str, Any]]) -> int:
        """
        Insert opportunities from CSV into  opportunities_unique table.
        """
        if not rows:
            return 0

        # Get the column names from the first row
        columns = [col for col in rows[0].keys() if col is not None and str(col).strip() != ""]
        if not columns:
            raise ValueError("No valid CSV columns found for insert")
        columns_str = ", ".join(columns)
        
        # Build the VALUES clause with proper placeholders
        placeholders = ", ".join(["%s"] * len(columns))
        q = f"""
            INSERT INTO {self.schema}.opportunities_unique (
                {columns_str}
            )
            VALUES %s
        """

        # Prepare values preserving the column order
        # Convert empty strings to None for array columns
        values = []
        for r in rows:
            row_values = []
            for col in columns:
                val = r.get(col)
                if isinstance(val, str):
                    val = val.strip()
                    if val == "" or val.upper() == "NULL":
                        val = None
                # Convert empty strings to None (NULL) for array columns
                if col == "tags":
                    if val is None:
                        val = None
                    else:
                        val_str = str(val).strip()
                        if val_str == "" or val_str.upper() == "NULL":
                            val = None
                        else:
                            # Try to parse Python-style list literal from CSV, e.g. "['a', 'b']"
                            try:
                                parsed = ast.literal_eval(val_str)
                                # Ensure parsed is a list/tuple
                                if isinstance(parsed, (list, tuple)):
                                    # normalize elements to strings
                                    val = [str(x).strip() for x in parsed]
                                else:
                                    # fallback to single-element list
                                    val = [str(parsed)]
                            except Exception:
                                # Fallback: split by comma, strip quotes/spaces
                                parts = [p.strip().strip("'\"") for p in val_str.strip('[]{}').split(',') if p.strip() != '']
                                val = parts if parts else None
                elif col == "sources":
                    val = _normalize_pg_text_array_value(val)
                elif col == "opportunity_id":
                    val = _normalize_uuid_value(val, r)
                row_values.append(val)
            values.append(tuple(row_values))

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




def main() -> int:
    print("[CSV Inserter] Starting CSV import job...")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read database configuration from CSV
    config_file = os.path.join(script_dir, "Database_Connection.csv")
    print(f"[CSV Inserter] Reading database configuration from: {config_file}")
    
    try:
        config = _read_database_config(config_file)
    except Exception as e:
        print(f"[CSV Inserter] ERROR reading database config: {e}", file=sys.stderr)
        return 1
    
    host = config["DB_HOST"]
    port = int(config["DB_PORT"])
    user = config["DB_USER"]
    password = config["DB_PASSWORD"]
    dbname = config["DB_NAME"]
    schema = config["DB_SCHEMA"]
    
    # Path to opportunities CSV file
    csv_file = os.path.join(script_dir, "Unique_Opportunities_06May_Published.csv") 
 

    print(f"[CSV Inserter] Looking for CSV file at: {csv_file}")
    print(f"[CSV Inserter] Connecting to DB host={host} port={port} db={dbname} schema={schema} user={user} ...")
    
    db = DatabaseManager(host, port, user, password, dbname, schema=schema)
    print("[CSV Inserter] Connected to DB.")

    try:
        print(f"[CSV Inserter] Reading CSV file: {csv_file} ...")
        rows = _read_csv_file(csv_file)
        print(f"[CSV Inserter] Read {len(rows)} rows from CSV file.")

        if len(rows) == 0:
            print("[CSV Inserter] No rows to insert. Exiting.")
            return 0

        print(f"[CSV Inserter] Inserting {len(rows)} rows into {schema}.opportunities_unique ...")
        inserted = db.insert_opportunities(rows)

        print("[CSV Inserter] Committing transaction ...")
        db.commit()

        print(f"[CSV Inserter] Successfully inserted {inserted} rows.")
        print("[CSV Inserter] Done.")
        return 0

    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        print(f"[CSV Inserter] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

