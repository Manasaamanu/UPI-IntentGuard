# UPI IntentGuard
Intent-aware deep learning demo for detecting potentially manipulated but authorized payments.

## Run on Windows
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data.generate_data
python -m src.training.train
streamlit run app.py
```
Open the local Streamlit URL, normally http://localhost:8501.

## PDF-aligned architecture
- GRU: learns recent user transaction sequences.
- Graph message-passing branch: learns recipient co-occurrence relationships.
- Context MLP: analyses amount, time, device, location and category.
- Neural fusion head: combines branch embeddings to estimate risk.
- Intent Drift Score: weighted score of amount deviation (25%), behavioural anomaly (25%), recipient unfamiliarity (20%), unusual timing (15%) and payment context (15%).
- Adaptive response: ALLOW (<40), VERIFY (40–74.9), PROTECT (>=75).

## UI
Overview, Live Risk Assessment, Transaction Explorer, Model Evaluation and Assessment History.

## Data and responsible use
The dataset is synthetic. Positive labels simulate social-engineering scenarios; they are not observed UPI fraud. Metrics are demonstration results only. No live payment integration or automatic blocking is implemented. Never enter credentials or real personal financial information.
