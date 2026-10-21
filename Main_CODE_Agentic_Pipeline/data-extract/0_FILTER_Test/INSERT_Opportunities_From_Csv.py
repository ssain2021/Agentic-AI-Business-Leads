#!/usr/bin/env python3
"""
Read opportunities from CSV file and insert into opportunities_unique_filter table.

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
from typing import Any, Dict, List
import ast

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

    for encoding in ("utf-8", "cp1252"):
        try:
            rows: List[Dict[str, Any]] = []
            with open(filepath, "r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("CSV file is empty or has no headers")

                for row_dict in reader:
                    rows.append(row_dict)

            return rows
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        "Failed to decode CSV with utf-8 or cp1252",
    )


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
        Insert opportunities from CSV into  opportunities_unique_filter table.
        """
        if not rows:
            return 0

        # Get the column names from the first row
        columns = list(rows[0].keys())
        columns_str = ", ".join(columns)
        
        # Build the VALUES clause with proper placeholders
        placeholders = ", ".join(["%s"] * len(columns))
        q = f"""
            INSERT INTO {self.schema}.opportunities_unique_filter (
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
                # Convert empty strings to None (NULL) for array columns
                if col in ("tags", "sources"):
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
    csv_file = os.path.join(script_dir, "Unique_Opportunities_11Mar_Publish.csv")

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

        print(f"[CSV Inserter] Inserting {len(rows)} rows into {schema}.opportunities_unique_filter ...")
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

