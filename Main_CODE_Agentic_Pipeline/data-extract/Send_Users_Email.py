#!/usr/bin/env python3
"""
Send_Email.py

Option A schema:
- final_categories
- final_events
- final_category_events

Email output:
- Category-wise sections
- Two-column Excel-like layout:
    Left column = labels (Category, Company/partnership, etc.)
    Right column = values
- Brand colors aligned to Excel sample:
    Purple: #9900FF
    Orange: #FF9900
    Cyan:   #00FFFF

Hardcoded dashboard link:
- https://calyxonai.com/dashboard
(All CTAs and links route users to dashboard.)
"""

import os
import smtplib
import ssl
import psycopg2
import psycopg2.extras
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -----------------------
# Config
# -----------------------
PG_DSN = os.getenv(
    "PG_DSN",
    "dbname=testdb user=devmain password=devmain#db004@calyxonai host=srv1172911.hstgr.cloud port=5432",
)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.hostinger.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "support@calyxonai.com")
SMTP_PASS = os.getenv("SMTP_PASS", "XEy%9pGhL7YQ7")
MAIL_FROM = os.getenv("MAIL_FROM", "calyxon ai reports <support@calyxonai.com>")
MAIL_TO = [e.strip() for e in os.getenv("MAIL_TO", "s_sain@hotmail.com").split(",") if e.strip()]

REPORT_DATE = os.getenv("REPORT_DATE")  # optional YYYY-MM-DD

# Option-A knobs
REPORT_THRESHOLD = float(os.getenv("REPORT_THRESHOLD", "0.75"))
REPORT_MAX_AGE_DAYS = int(os.getenv("REPORT_MAX_AGE_DAYS", "10"))
REPORT_MAX_PER_CATEGORY = int(os.getenv("REPORT_MAX_PER_CATEGORY", "3"))  # set to 8 for default Option-A

FIRST_NAME = os.getenv("FIRST_NAME", "there")

# Hardcoded Dashboard Link (as requested)
DASHBOARD_URL = "https://calyxonai.com/dashboard"


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


def pct(conf) -> str:
    try:
        v = float(conf)
    except Exception:
        return ""
    v = max(0.0, min(1.0, v))
    return f"{round(v * 100)}%"


def build_event_link(event_id: str, article_url: str) -> str:
    # Always route users through the Calyxon AI dashboard (hardcoded)
    return DASHBOARD_URL


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
    return ", ".join(parts[:6])


# -----------------------
# Email rendering (Excel-style colors + 2-column table)
# -----------------------
PURPLE = "#9900FF"
ORANGE = "#FF9900"
CYAN = "#00FFFF"
TEXT = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
BG = "#FFFFFF"


def build_email_html(report_dt: date, categories, rows_by_cat):
    subject = f"Calyxon AI - Your Web3 Opportunity Report - {report_dt.isoformat()}"

    LEFT_W = "240"
    RIGHT_W = "480"

    # ✅ This is the corrected row() (Category big+bold; others small+not bold)
    def row(label: str, value_html: str) -> str:
        is_category = (label.strip().lower() == "category")

        if is_category:
            left_style = (
                f"padding:12px 12px 12px 0;"
                f"font-size:16px;"
                f"font-weight:700;"
                f"color:{TEXT};"
            )
            right_padding = "padding:12px 0;"
        else:
            left_style = (
                f"padding:8px 12px 8px 0;"
                f"font-size:13px;"
                f"font-weight:400;"
                f"color:{TEXT};"
            )
            right_padding = "padding:8px 0;"

        return f"""\
<tr>
  <td valign="top" width="{LEFT_W}" style="{left_style}">
    {esc(label)}
  </td>
  <td valign="top" width="{RIGHT_W}" style="{right_padding}font-size:14px;color:{TEXT};">
    {value_html}
  </td>
</tr>
"""

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
                <div style="font-size:22px;font-weight:700;line-height:1.2;">
                  {esc("Calyxon AI - Your Web3 Opportunity Report")}
                </div>
                <div style="margin-top:8px;font-size:14px;color:{MUTED};">
                  Report date: {esc(report_dt.isoformat())}
                </div>
              </td>
            </tr>

            <tr>
              <td style="padding:18px 0 0 0;font-size:14px;line-height:1.6;">
                <div>Dear {esc(FIRST_NAME)},</div>
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

            <tr>
              <td style="padding:18px 0 0 0;">
                <div style="height:2px;background:{CYAN};width:100%;"></div>
              </td>
            </tr>
""")

    for c in categories:
        cid = int(c["category_id"])
        cname = str(c["category_name"])
        opps = rows_by_cat.get(cid, [])

        parts.append("""\
            <tr>
              <td style="padding:18px 0 0 0;">
                <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="border-collapse:collapse;">
""")

        cat_value = f"""\
<span style="color:{PURPLE};font-weight:700;font-size:16px;text-decoration:underline;">
  {esc(cname)} ({len(opps)} {"opportunity" if len(opps)==1 else "opportunities"})
</span>
"""
        parts.append(row("Category", cat_value))

        if not opps:
            parts.append(row("", f"<span style='color:{MUTED};'>No opportunities met the confidence threshold today.</span>"))
            parts.append("""\
                </table>
              </td>
            </tr>
""")
            continue

        for idx, r in enumerate(opps, start=1):
            event_id = str(r.get("event_id") or "").strip()
            article_url = r.get("url") or ""
            link = build_event_link(event_id, article_url)

            company_line = format_company_partnership(r.get("primary_company") or "", r.get("related_companies"))
            opp_types = format_opportunity_types(r.get("opportunity_types"), r.get("tags"))
            confidence_str = pct(r.get("confidence"))
            summary = truncate(r.get("summary_text") or "", 650)
            reason = truncate(r.get("reason") or "", 160)

            # separator between opportunities
            if idx > 1:
                parts.append(f"""\
<tr>
  <td colspan="2" style="padding:10px 0;">
    <div style="height:1px;background:{BORDER};width:100%;"></div>
  </td>
</tr>
""")

            parts.append(row(
                "Company/partnership",
                f"<span style='font-weight:700;font-size:16px;'>{esc(company_line) if company_line else '—'}</span>"
            ))

            parts.append(row(
                "Opportunity heading",
                f"<span style='font-weight:700;font-size:16px;'>{esc(r.get('title') or '')}</span>"
            ))

            if summary:
                summary_html = (
                    "<div style='font-weight:700;'>Summary:</div>"
                    f"<div style='margin-top:2px;line-height:1.6;'>{esc(summary)}</div>"
                )
            else:
                summary_html = f"<span style='color:{MUTED};'>—</span>"

            parts.append(row("Summary of Opportunity", summary_html))

            parts.append(row(
                "Opportunity Type (tags)",
                f"<span style='font-weight:700;'>{esc(opp_types) if opp_types else '—'}</span>"
            ))

            parts.append(row(
                "Confidence",
                f"<span style='font-weight:700;'>Confidence: {esc(confidence_str)}</span>"
            ))

            if reason:
                parts.append(row(
                    "Why this matches",
                    f"<span style='color:{MUTED};'>{esc(reason)}</span>"
                ))

            parts.append(row(
                "Button (opens dashboard)",
                f"""<a href="{esc(link)}" style="color:{ORANGE};font-weight:700;text-decoration:none;font-size:14px;">
                      👉 View full opportunity &amp; outreach guidance
                    </a>"""
            ))

        parts.append("""\
                </table>
              </td>
            </tr>
""")

    parts.append(f"""\
            <tr>
              <td style="padding:22px 0 0 0;">
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


def send_email(subject: str, html_body: str) -> None:
    log(f"Preparing email: subject='{subject}'")
    log(f"Recipients: {MAIL_TO}")

    msg = MIMEMultipart("alternative")
    msg["From"] = MAIL_FROM
    msg["To"] = ", ".join(MAIL_TO)
    msg["Subject"] = subject

    text_fallback = (
        "Calyxon AI - Your Web3 Opportunity Report\n\n"
        "Your email client does not support HTML.\n"
        f"Please view this report in the Calyxon AI dashboard: {DASHBOARD_URL}"
    )
    msg.attach(MIMEText(text_fallback, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    log(f"Connecting to SMTP {SMTP_HOST}:{SMTP_PORT} ...")

    if SMTP_PORT == 465:
        context = ssl.create_default_context()
        log("Using SMTP_SSL (implicit TLS) for port 465")
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30, context=context) as server:
            if SMTP_USER and SMTP_PASS:
                log("Logging in...")
                server.login(SMTP_USER, SMTP_PASS)
            log("Sending email...")
            server.sendmail(MAIL_FROM, MAIL_TO, msg.as_string())
            log("Email sent (SMTP_SSL).")
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            if SMTP_USER and SMTP_PASS:
                log("Logging in...")
                server.login(SMTP_USER, SMTP_PASS)
            log("Sending email...")
            server.sendmail(MAIL_FROM, MAIL_TO, msg.as_string())
            log("Email sent (STARTTLS).")


# -----------------------
# Data access (Option A)
# -----------------------
SQL_CATEGORIES = """
SELECT category_id, category_name
FROM final_categories
ORDER BY category_name;
"""

SQL_TOP_CATEGORY_EVENTS = """
WITH ranked AS (
    SELECT
        fce.category_id,
        fce.event_id,
        fce.confidence,
        fce.reason,
        fce.primary_company,
        fce.related_companies,
        fce.opportunity_types,
        fce.tags,
        e.title,
        e.url,
        e.source,
        e.published_at,
        e.summary_text,
        ROW_NUMBER() OVER (
            PARTITION BY fce.category_id
            ORDER BY fce.confidence DESC, e.published_at DESC NULLS LAST, fce.event_id ASC
        ) AS rn
    FROM final_category_events fce
    JOIN final_events e
      ON e.event_id = fce.event_id
    WHERE fce.confidence >= %(threshold)s
      AND (
            e.published_at IS NULL
            OR e.published_at >= (NOW() - (%(max_age_days)s || ' days')::interval)
          )
)
SELECT *
FROM ranked
WHERE rn <= %(max_per_category)s
ORDER BY category_id, rn;
"""


def main() -> None:
    log("START: Daily email generation")

    if not MAIL_TO:
        raise RuntimeError("MAIL_TO is empty.")
    if not PG_DSN:
        raise RuntimeError("PG_DSN is not set.")

    if REPORT_DATE:
        report_dt = datetime.strptime(REPORT_DATE, "%Y-%m-%d").date()
    else:
        report_dt = date.today()

    log(f"Report date = {report_dt}")
    log(f"Threshold={REPORT_THRESHOLD} MaxAgeDays={REPORT_MAX_AGE_DAYS} MaxPerCategory={REPORT_MAX_PER_CATEGORY}")

    log("Connecting to PostgreSQL...")
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            log("Fetching categories...")
            cur.execute(SQL_CATEGORIES)
            categories = cur.fetchall()
            log(f"Fetched {len(categories)} categories")

            log("Fetching opportunities per category...")
            cur.execute(SQL_TOP_CATEGORY_EVENTS, {
                "threshold": REPORT_THRESHOLD,
                "max_age_days": REPORT_MAX_AGE_DAYS,
                "max_per_category": REPORT_MAX_PER_CATEGORY,
            })
            rows = cur.fetchall()
            log(f"Fetched {len(rows)} category-event rows")

            rows_by_cat = {}
            for r in rows:
                cid = int(r["category_id"])
                rows_by_cat.setdefault(cid, []).append(r)

    log("Rendering HTML email...")
    html_body, subject = build_email_html(report_dt, categories, rows_by_cat)

    log("Sending email...")
    send_email(subject, html_body)

    log("END: Daily email generation")


if __name__ == "__main__":
    main()

