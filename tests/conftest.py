import pytest


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Keep tests independent of any API_KEY set in a developer's .env file."""
    monkeypatch.delenv("API_KEY", raising=False)
