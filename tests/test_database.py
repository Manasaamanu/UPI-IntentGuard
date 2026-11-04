import pytest
from src.database import save, history


def test_save_valid_assessment():
    save("TEST_USER", "TEST_RECIPIENT", 500, 20, 15, "ALLOW", 10)
    records = history()
    assert len(records) > 0
    assert records.iloc[0]["user_ref"] == "TEST_USER"


def test_save_rejects_negative_amount():
    with pytest.raises(ValueError):
        save("TEST_USER", "TEST_RECIPIENT", -500, 20, 15, "ALLOW", 10)


def test_save_rejects_invalid_risk():
    with pytest.raises(ValueError):
        save("TEST_USER", "TEST_RECIPIENT", 500, 120, 15, "ALLOW", 10)


def test_rejects_nan_amount():
    with pytest.raises(ValueError):
        save("USER1", "RECIPIENT1", float("nan"), 20, 15, "ALLOW", 10)


def test_rejects_infinite_risk():
    with pytest.raises(ValueError):
        save("USER1", "RECIPIENT1", 500, float("inf"), 15, "ALLOW", 10)


def test_rejects_empty_user_reference():
    with pytest.raises(ValueError):
        save("", "RECIPIENT1", 500, 20, 15, "ALLOW", 10)


def test_rejects_empty_recipient_reference():
    with pytest.raises(ValueError):
        save("USER1", "", 500, 20, 15, "ALLOW", 10)