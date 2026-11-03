from src.scoring import intent_drift_score, action


def test_low_risk_payment():
    decision, message = action(20, 15)
    assert decision == "ALLOW"


def test_medium_risk_payment():
    decision, message = action(35, 50)
    assert decision == "VERIFY"


def test_high_risk_payment():
    decision, message = action(80, 60)
    assert decision == "PROTECT"


def test_intent_drift_score():
    score, parts = intent_drift_score(
        0.5, 0.5, 1.0, 0.0, 0.5
    )

    assert score == 52.5
    assert parts["Recipient unfamiliarity"] == 100.0