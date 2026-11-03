
import sqlite3
import math
import pandas as pd
from config import ROOT

DB = ROOT / "database/intentguard.db"


def init():
    DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS assessments(
                id INTEGER PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                user_ref TEXT,
                recipient_ref TEXT,
                amount REAL,
                risk REAL,
                ids REAL,
                action TEXT,
                probability REAL
            )
        """)


def validate_number(value, field, minimum, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a valid number.")

    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")

    if number < minimum:
        raise ValueError(f"{field} cannot be below {minimum}.")

    if maximum is not None and number > maximum:
        raise ValueError(f"{field} cannot exceed {maximum}.")

    return number


def save(u, r, a, risk, ids, act, p):
    if not isinstance(u, str) or not u.strip():
        raise ValueError("User reference cannot be empty.")

    if not isinstance(r, str) or not r.strip():
        raise ValueError("Recipient reference cannot be empty.")

    a = validate_number(a, "Amount", 0)
    risk = validate_number(risk, "Risk score", 0, 100)
    ids = validate_number(ids, "Intent Drift Score", 0, 100)
    p = validate_number(p, "Probability", 0, 100)

    if act not in ["ALLOW", "VERIFY", "PROTECT"]:
        raise ValueError("Invalid payment decision.")

    init()

    with sqlite3.connect(DB) as c:
        c.execute("""
            INSERT INTO assessments(
                user_ref, recipient_ref, amount,
                risk, ids, action, probability
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (u.strip(), r.strip(), a, risk, ids, act, p))


def history():
    init()

    with sqlite3.connect(DB) as c:
        return pd.read_sql_query(
            "SELECT * FROM assessments ORDER BY id DESC", c
        )
