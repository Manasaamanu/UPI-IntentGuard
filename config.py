from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
for p in ["data/raw","artifacts","reports","database"]: (ROOT/p).mkdir(parents=True,exist_ok=True)
RAW=ROOT/"data/raw/synthetic_transactions.csv"
ART=ROOT/"artifacts"
REPORTS=ROOT/"reports"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN=10
FEATURES=["log_amount","amount_deviation","hour_sin","hour_cos","new_recipient","device_changed","location_changed","velocity","category_unusual","recipient_prior"]
