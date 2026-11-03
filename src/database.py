import sqlite3,pandas as pd
from config import ROOT
DB=ROOT/"database/intentguard.db"
def init():
    with sqlite3.connect(DB) as c:c.execute("CREATE TABLE IF NOT EXISTS assessments(id INTEGER PRIMARY KEY,created_at TEXT DEFAULT CURRENT_TIMESTAMP,user_ref TEXT,recipient_ref TEXT,amount REAL,risk REAL,ids REAL,action TEXT,probability REAL)")
def save(u,r,a,risk,ids,act,p):
    init()
    with sqlite3.connect(DB) as c:c.execute("INSERT INTO assessments(user_ref,recipient_ref,amount,risk,ids,action,probability) VALUES(?,?,?,?,?,?,?)",(u,r,a,risk,ids,act,p))
def history():
    init()
    with sqlite3.connect(DB) as c:return pd.read_sql_query("SELECT * FROM assessments ORDER BY id DESC",c)
