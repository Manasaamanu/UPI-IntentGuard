def _unit(value):
    return max(0.0, min(1.0, float(value)))


def intent_drift_score(amount, behaviour, recipient, timing, context):
    values = {
        "Amount deviation": amount,
        "Behaviour anomaly": behaviour,
        "Recipient unfamiliarity": recipient,
        "Unusual timing": timing,
        "Payment context": context,
    }
    weights = {
        "Amount deviation": 0.25,
        "Behaviour anomaly": 0.25,
        "Recipient unfamiliarity": 0.20,
        "Unusual timing": 0.15,
        "Payment context": 0.15,
    }
    score = sum(weights[key] * _unit(value) for key, value in values.items()) * 100
    parts = {key: round(_unit(value) * 100, 1) for key, value in values.items()}
    return round(score, 1), parts


def action(risk, ids=None):
    decision_score = max(float(risk), float(ids if ids is not None else risk))
    if decision_score >= 75:
        return "PROTECT", (
            "HIGH-RISK PAYMENT — pause and independently verify the recipient "
            "and payment purpose."
        )
    if decision_score >= 40:
        return "VERIFY", (
            "SUSPICIOUS PAYMENT — independently verify the recipient and "
            "payment purpose before proceeding."
        )
    return "ALLOW", (
        "LOW-RISK PAYMENT — no strong warning detected; review payment details carefully."
    )
