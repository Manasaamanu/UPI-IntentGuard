
import sqlite3
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


def save(u, r, a, risk, ids, act, p):
    # Validate transaction amount
    if float(a) < 0:
        raise ValueError("Transaction amount cannot be negative.")

    # Validate risk score
    if not 0 <= float(risk) <= 100:
        raise ValueError("Risk score must be between 0 and 100.")

    # Validate Intent Drift Score
    if not 0 <= float(ids) <= 100:
        raise ValueError("Intent Drift Score must be between 0 and 100.")

    # Validate model probability (percentage)
    if not 0 <= float(p) <= 100:
        raise ValueError("Probability must be between 0 and 100.")

    if act not in ["ALLOW", "VERIFY", "PROTECT"]:
        raise ValueError("Invalid payment decision.")

    init()

    with sqlite3.connect(DB) as c:
        c.execute(
            """
            INSERT INTO assessments(
                user_ref, recipient_ref, amount,
                risk, ids, action, probability
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (u, r, a, risk, ids, act, p)
        )


def history():
    init()

    with sqlite3.connect(DB) as c:
        return pd.read_sql_query(
            "SELECT * FROM assessments ORDER BY id DESC", c
        )

