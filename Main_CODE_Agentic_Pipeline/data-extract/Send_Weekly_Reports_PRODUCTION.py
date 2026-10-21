import re
import re
def format_reason_bullets(reason_text: str) -> str:
    """Render 'Why this matters' from opportunities_unique.reason.

    We display ONLY:
      • Evidence snippet
      • Reason

    Parse from full text; then truncate per-section to keep emails readable.
    """
    if not reason_text:
        return ""

    txt = str(reason_text).strip()

    snippet_match = re.search(r"Evidence\s+snippet:\s*(.+?)(?:\n\s*Reason:|\s+Reason:|$)", txt, re.IGNORECASE | re.DOTALL)
    reason_match = re.search(r"Reason:\s*(.+)$", txt, re.IGNORECASE | re.DOTALL)

    snippet = (snippet_match.group(1).strip() if snippet_match else "")
    reason = (reason_match.group(1).strip() if reason_match else "")

    if not snippet and not reason:
        return "<div style='margin-top:6px;'>" + esc(txt) + "</div>"

    def _truncate_section(s: str, max_chars: int) -> str:
        s = (s or "").strip()
        if len(s) <= max_chars:
            return s
        # Add ASCII ellipsis when truncated
        if max_chars <= 3:
            return "..."[:max_chars]
        return s[: max_chars - 3].rstrip() + "..."

    snippet = _truncate_section(snippet, 160)
    reason = _truncate_section(reason, 320)

    items = []
    if snippet:
        items.append(f"<li><span style='font-weight:700;'>Evidence snippet:</span> \"{esc(snippet)}\"</li>")
    if reason:
        items.append(f"<li><span style='font-weight:700;'>Reason:</span> {esc(reason)}</li>")

    return "<ul style='margin:6px 0 0 18px;padding:0;'>" + "".join(items) + "</ul>"

#!/usr/bin/env python3
"""
Send_Daily_Reports.py

Sends daily category-wise opportunity reports to ALL users in public.users.

Key behaviors:
- Users opt into categories via users.categories (comma-separated category_text_id values like: "web3,defi,exchange")
- Categories are resolved via final_categories.category_text_id
- Opportunities come from public.opportunities_unique
- Only confidence >= REPORT_THRESHOLD are included (default 0.75)
- Events older than REPORT_MAX_AGE_DAYS are excluded (default 10)
- Top REPORT_MAX_PER_CATEGORY per category (default 8)

Email formatting (client-facing):
- SINGLE COLUMN (no left label column)
- Category header in purple + underline
- Company/partnership bold
- Opportunity heading larger + bold
- Summary with "Summary:" label
- Opportunity Type line
- Confidence shown as: "Confidence: 90%" (NO duplication)
  - >= 90% uses #A2FFDC
  - 75–89% uses #FFB454
- CTA goes to hardcoded dashboard URL
"""

import csv
import os
import smtplib
import ssl
import sys
import psycopg2
import psycopg2.extras
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# -----------------------
# Logging
# -----------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -----------------------
# Database Configuration Helper
# -----------------------
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


# -----------------------
# Config
# -----------------------

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.hostinger.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "support@calyxonai.com")
SMTP_PASS = os.getenv("SMTP_PASS", "XEy%9pGhL7YQ7")
MAIL_FROM = os.getenv("MAIL_FROM", "[CalyxonAI] Weekly Opportunities Report <support@calyxonai.com>")

# Optional allowlist for testing (comma-separated emails). If empty -> send to all users.
ALLOW_EMAILS = {e.strip().lower() for e in os.getenv("ALLOW_EMAILS", "").split(",") if e.strip()}

REPORT_DATE = os.getenv("REPORT_DATE")  # optional YYYY-MM-DD

REPORT_THRESHOLD = float(os.getenv("REPORT_THRESHOLD", "0.65"))
REPORT_MAX_AGE_DAYS = int(os.getenv("REPORT_MAX_AGE_DAYS", "10"))
REPORT_MAX_PER_CATEGORY = int(os.getenv("REPORT_MAX_PER_CATEGORY", "8"))

DEFAULT_FIRST_NAME = os.getenv("DEFAULT_FIRST_NAME", "there")

# Hardcoded dashboard link
DASHBOARD_URL = "https://calyxonai.com/dashboard"


# -----------------------
# SQL
# -----------------------
SQL_USERS = """
SELECT id, email, name, categories
FROM public.users
WHERE email IS NOT NULL AND email <> ''
ORDER BY created_at ASC;
"""

SQL_CATEGORIES_BY_TEXT_IDS = """
SELECT category_id, category_name, category_text_id
FROM public.final_categories
WHERE category_text_id = ANY(%s)
ORDER BY category_name;
"""

SQL_TOP_CATEGORY_EVENTS_FOR_CATEGORY_IDS = """
SELECT DISTINCT ON (ou.opportunity_id)
  btrim(ctext_raw) AS category_text_id,
  ou.opportunity_id AS opportunity_id,
    ou.confidence,
    ou.reason,
    ou.target_company AS primary_company,
    NULL::text[] AS related_companies,
    NULL::text[] AS opportunity_types,
    ou.tags,
    ou.title,
    ou.event_url AS url,
    NULL::text AS source,
    ou.time_found AS published_at,
    ou.summary AS summary_text
FROM public.opportunities_unique ou
CROSS JOIN LATERAL unnest(string_to_array(COALESCE(ou.categories, ''), ',')) AS ctext_raw
WHERE ou.confidence >= %(threshold)s
  AND btrim(ctext_raw) = ANY(%(category_ids)s)
  AND (
        ou.time_found IS NULL
        OR ou.time_found >= (NOW() - (%(max_age_days)s || ' days')::interval)
      )
ORDER BY ou.opportunity_id, ou.confidence DESC, ou.time_found DESC NULLS LAST
LIMIT 10;
"""


# -----------------------
# Helpers
# -----------------------
def esc(s) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def truncate(s, max_chars: int) -> str:
    if not s:
        return ""
    s = str(s).strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def ensure_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def parse_category_text_ids(raw: str) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def first_name_from_user(full_name: str) -> str:
    if not full_name:
        return DEFAULT_FIRST_NAME
    first = full_name.strip().split()[0]
    return first if first else DEFAULT_FIRST_NAME


def format_company_partnership(primary_company: str, related_companies) -> str:
    pc = (primary_company or "").strip()
    rel = [str(x).strip() for x in ensure_list(related_companies) if x is not None and str(x).strip()]
    rel = [x for x in rel if x and (not pc or x.lower() != pc.lower())]
    if pc and rel:
        rel_show = rel[:2]
        if len(rel_show) == 1:
            return f"{pc} ↔ {rel_show[0]}"
        return f"{pc} ↔ {', '.join(rel_show)}"
    if pc:
        return pc
    if rel:
        return ", ".join(rel[:2])
    return ""


def format_opportunity_types(opportunity_types, tags) -> str:
  ot = [str(x).strip() for x in ensure_list(opportunity_types) if x is not None and str(x).strip()]
  tg = [str(x).strip() for x in ensure_list(tags) if x is not None and str(x).strip()]
  parts = ot if ot else tg
  # First Cap only for each tag
  def first_cap(s):
    return s[:1].upper() + s[1:].lower() if s else s
  return ", ".join(first_cap(x) for x in parts[:6])


def confidence_html(conf) -> str:
    """
    Returns colored HTML for BOTH label and value:
      'Confidence: 90%'

    Color rules:
      >= 80%  -> #A2FFDC
      60–80%  -> #FFB454
    """
    try:
        v = float(conf)
    except Exception:
        return "<span>Confidence: —</span>"

    pct_val = round(v * 100)
    color = "#44BD85" if pct_val >= 80 else "#DF983B"
    return f"<span style='font-weight:700;color:{color};'>Confidence: {pct_val}%</span>"





# -----------------------
# Email styling constants
# -----------------------
PURPLE = "#9900FF"
ORANGE = "#FF9900"
CYAN = "#00FFFF"
TEXT = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
BG = "#FFFFFF"


# -----------------------
# HTML renderer (single column)
# -----------------------
def build_email_html(report_dt: date, first_name: str, categories, rows_by_cat):
    subject = f"Calyxon AI - Your Web3 Opportunity Report - {report_dt.isoformat()}"

    parts = []
    parts.append(f"""\
<!doctype html>
<html>
  <body style="margin:0;padding:0;background:{BG};font-family:Arial,Helvetica,sans-serif;color:{TEXT};">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background:{BG};">
      <tr>
        <td align="center" style="padding:24px 12px;">
          <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="720" style="width:720px;max-width:720px;">

            <tr>
              <td style="padding:0 0 10px 0;">
                <div style="font-size:26px;font-weight:700;line-height:1.2;">
                  {esc("Calyxon AI - Your Web3 Opportunity Report")}
                </div>
                <div style="margin-top:8px;font-size:14px;color:{MUTED};">
                  Report date: {esc(report_dt.isoformat())}
                </div>
              </td>
            </tr>

            <tr>
              <td style="padding:18px 0 0 0;font-size:16px;line-height:1.6;">
                <div>Dear {esc(first_name)},</div>
                <div style="margin-top:10px;">
                  Here is your latest Calyxon AI opportunity report based on your selected information categories.
                </div>
                <div style="margin-top:10px;">
                  Below you’ll find the most relevant partnership, integration, and other opportunities identified across the Web3 ecosystem.
                </div>
                <div style="margin-top:10px;">
                  Each opportunity links to full context and source material inside your Calyxon AI dashboard.
                </div>
              </td>
            </tr>

            <style>
              .dashboardLink {{
                  text-decoration: none;
                  color: {PURPLE};
                  display: inline-block;
                  transition: color 200ms ease, transform 200ms ease;
              }}
              .dashboardLink:hover {{
                  color: {CYAN};
                  text-decoration: underline;
                  transform: scale(1.05);
              }}
            </style>

            <tr>
              <td style="padding:12px 0 0 0;">
                <a href="{DASHBOARD_URL}" class="dashboardLink" style="text-decoration:none;color:{ORANGE};display:inline-block;transition:color 200ms ease,transform 200ms ease;;font-weight:700;font-size:14px;">
                  👉 Go to your Dashboard
                </a>
              </td>
            </tr>

            <tr>
              <td style="padding:18px 0 0 0;">
                <div style="height:2px;background:{TEXT};width:100%;"></div>
              </td>
            </tr>
""")

    for c in categories:
        cid = str(c["category_text_id"])
        cname = str(c["category_name"])
        opps = rows_by_cat.get(cid, [])

        if not opps:
            # parts.append(f"""\
            #             <tr>
            #               <td style="padding:6px 0 14px 0;color:{MUTED};font-size:13px;line-height:1.6;">
            #                 No opportunities met the confidence threshold today.
            #               </td>
            #             </tr>
            # """)
            continue

        # Category header (single column)
        parts.append(f"""\
            <tr>
              <td style="padding:18px 0 8px 0;">
                <span style="color:{PURPLE};font-weight:700;font-size:20px;text-decoration:underline;">
                  {esc(cname)} ({len(opps)} {"opportunity" if len(opps)==1 else "opportunities"})
                </span>
              </td>
            </tr>
""")

        for idx, r in enumerate(opps, start=1):
            if idx > 1:
                parts.append(f"""\
            <tr>
              <td style="padding:10px 0;">
                <div style="height:1px;background:{BORDER};width:100%;"></div>
              </td>
            </tr>
""")

            company_line = format_company_partnership(
                r.get("primary_company") or "",
                r.get("related_companies")
            ) or "—"

            title_txt = r.get("title") or ""
            summary = truncate(r.get("summary_text") or "", 650)
            reason = ""
            reason_html = ""
            opp_types = format_opportunity_types(r.get("opportunity_types"), r.get("tags")) or "—"
            conf_html = confidence_html(r.get("confidence"))
            published_at = r.get("published_at")
            date_str = ""
            if published_at:
                if isinstance(published_at, str):
                    date_str = published_at[:10]
                else:
                    try:
                        date_str = published_at.strftime("%Y-%m-%d")
                    except Exception:
                        date_str = str(published_at)

            parts.append(f"""\
            <tr>
              <td style="padding:14px 0 18px 0;">

                <div style="font-size:16px;font-weight:700;line-height:1.35;">
                  {esc(company_line)}
                </div>

                <div style="margin-top:6px;font-size:18px;font-weight:700;line-height:1.3;">
                  {esc(title_txt)}
                </div>


                <div style="margin-top:8px;line-height:1.6;">
                  {conf_html}
                  <span style="color:{MUTED};"> • </span>
                  <span style="font-weight:700;">{esc(opp_types)}</span>
                  {f'<span style="color:#636872;"> • {date_str}</span>' if date_str else ''}
                </div>

                <div style="margin-top:12px;font-weight:700;">Summary:</div>
                <div style="margin-top:4px;line-height:1.6;">
                  {esc(summary) if summary else "<span style='color:"+MUTED+";'>—</span>"}
                </div>

                {"<div style='margin-top:10px;line-height:1.6;'><div style='font-weight:700;color:"+TEXT+";'>Why this matters</div>"+reason_html+"</div>" if reason_html else ""}

                <div style="margin-top:12px;">
                  <a href="{esc(f"{DASHBOARD_URL}#/opportunity?id={r.get("opportunity_id")}")}"
                     style="color:{ORANGE};font-weight:700;text-decoration:none;font-size:14px;">
                     👉 View full opportunity &amp; outreach guidance
                  </a>
                </div>

              </td>
            </tr>
""")

        # extra spacing after each category
        parts.append(f"""\
            <tr><td style="padding:6px 0;"></td></tr>
""")

    # Footer
    parts.append(f"""\
            <tr>
              <td style="padding:18px 0 0 0;">
                <div style="height:1px;background:{BORDER};width:100%;"></div>
                <div style="margin-top:12px;font-size:14px;line-height:1.6;color:{TEXT};">
                  <span style="font-weight:700;color:{PURPLE};">What happens next</span>
                </div>
                <div style="margin-top:8px;font-size:14px;line-height:1.6;color:{TEXT};">
                  Manage your preferences or explore full opportunity details in your dashboard.
                </div>
                <div style="margin-top:10px;font-size:14px;line-height:1.6;">
                  <a href="{esc(DASHBOARD_URL)}" style="color:{ORANGE};font-weight:700;text-decoration:none;">
                    Go to your Calyxon AI Dashboard
                  </a>
                </div>
                <div style="margin-top:14px;font-size:12px;color:{MUTED};line-height:1.5;">
                  © {report_dt.year} Calyxon AI — Opportunity Intelligence for Web3 Teams
                </div>
              </td>
            </tr>

          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
""")

    return "\n".join(parts), subject


# -----------------------
# SMTP session (reuse one connection for the whole batch)
# -----------------------
class SmtpSession:
    def __init__(self):
        self.server = None

    def __enter__(self):
        log(f"Connecting to SMTP {SMTP_HOST}:{SMTP_PORT} ...")
        if SMTP_PORT == 465:
            context = ssl.create_default_context()
            self.server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30, context=context)
        else:
            self.server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
            self.server.ehlo()
            self.server.starttls(context=ssl.create_default_context())
            self.server.ehlo()

        if SMTP_USER and SMTP_PASS:
            log("Logging into SMTP...")
            self.server.login(SMTP_USER, SMTP_PASS)
            log("SMTP login successful")
        return self

    def send(self, to_addr: str, subject: str, html_body: str):
        msg = MIMEMultipart("alternative")
        msg["From"] = MAIL_FROM
        msg["To"] = to_addr
        msg["Subject"] = subject

        text_fallback = (
            "Calyxon AI - Your Web3 Opportunity Report\n\n"
            "Your email client does not support HTML.\n"
            f"Please view this report in the Calyxon AI dashboard: {DASHBOARD_URL}"
        )
        msg.attach(MIMEText(text_fallback, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        self.server.sendmail(MAIL_FROM, [to_addr], msg.as_string())

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.server:
                self.server.quit()
        except Exception:
            pass


# -----------------------
# Main (loop over all users)
# -----------------------
def main() -> None:
    log("START: Daily reports (all users)")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read database configuration from CSV
    config_file = os.path.join(script_dir, "Database_Connection.csv")
    log(f"Reading database configuration from: {config_file}")
    
    try:
        config = _read_database_config(config_file)
    except Exception as e:
        log(f"ERROR reading database config: {e}")
        raise
    
    host = config["DB_HOST"]
    port = int(config["DB_PORT"])
    user = config["DB_USER"]
    password = config["DB_PASSWORD"]
    dbname = config["DB_NAME"]
    schema = config["DB_SCHEMA"]

    if REPORT_DATE:
        report_dt = datetime.strptime(REPORT_DATE, "%Y-%m-%d").date()
    else:
        report_dt = date.today() 
    log(f"Threshold={REPORT_THRESHOLD} MaxAgeDays={REPORT_MAX_AGE_DAYS} MaxPerCategory={REPORT_MAX_PER_CATEGORY}")
    if ALLOW_EMAILS:
        log(f"ALLOW_EMAILS enabled: {sorted(ALLOW_EMAILS)}")

    sent = 0
    skipped = 0
    errors = 0

    log(f"Connecting to PostgreSQL host={host} port={port} db={dbname} schema={schema} user={user}...")
    conn_str = f"host={host} port={port} user={user} password={password} dbname={dbname}"
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            log("Fetching users...")
            cur.execute(SQL_USERS)
            users = cur.fetchall()
            log(f"Fetched {len(users)} users")

            with SmtpSession() as smtp:
                for u in users:
                    email = (u.get("email") or "").strip()
                    if not email:
                        skipped += 1
                        continue

                    if ALLOW_EMAILS and email.lower() not in ALLOW_EMAILS:
                        skipped += 1
                        continue

                    cat_text_ids = parse_category_text_ids(u.get("categories"))
                    if not cat_text_ids:
                        log(f"SKIP: {email} has no categories selected")
                        skipped += 1
                        continue

                    # Resolve category_text_id -> category_id/category_name
                    cur.execute(SQL_CATEGORIES_BY_TEXT_IDS, (cat_text_ids,))
                    categories = cur.fetchall()
                    if not categories:
                        log(f"SKIP: {email} categories did not resolve in final_categories: {cat_text_ids}")
                        skipped += 1
                        continue

                    category_ids = [str(c["category_text_id"]) for c in categories]

                    # Pull top 10 opportunities overall for user's selected category_ids
                    cur.execute(SQL_TOP_CATEGORY_EVENTS_FOR_CATEGORY_IDS, {
                        "threshold": REPORT_THRESHOLD,
                        "max_age_days": REPORT_MAX_AGE_DAYS,
                        "category_ids": category_ids,
                    })
                    rows = cur.fetchall()
                    
                    # Sort by confidence descending
                    rows = sorted(rows, key=lambda r: (r.get("confidence") or 0), reverse=True)

                    rows_by_cat = {}
                    for r in rows:
                      cid = str(r["category_text_id"])
                      rows_by_cat.setdefault(cid, []).append(r)

                    first_name = first_name_from_user(u.get("name"))
                    html_body, subject = build_email_html(report_dt, first_name, categories, rows_by_cat)

                    try:
                        smtp.send(email, subject, html_body)
                        sent += 1
                        log(f"SENT: {email} categories={len(categories)} items={len(rows)}")
                    except Exception as e:
                        errors += 1
                        log(f"ERROR: sending to {email}: {e}")

    log(f"END: sent={sent} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()

