from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


class FakePipeline:
    def query(self, question):
        return {
            "answer": "12 million dollars",
            "sources": [{"content": "snippet", "source": "a.pdf", "page": 1}],
            "source_count": 1,
        }


class BrokenPipeline:
    def query(self, question):
        raise RuntimeError("boom")


def _use(pipeline):
    api.app.dependency_overrides[api.pipeline_dep] = lambda: pipeline


def teardown_function():
    api.app.dependency_overrides.clear()
    api.get_pipeline.cache_clear()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_query_ok():
    _use(FakePipeline())
    r = client.post("/query", json={"question": "How much is the budget?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] == "12 million dollars"
    assert body["source_count"] == 1
    assert body["sources"][0]["source"] == "a.pdf"
    assert body["latency_ms"] >= 0


def test_query_rejects_empty_question():
    _use(FakePipeline())
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_rejects_too_long_question():
    _use(FakePipeline())
    assert client.post("/query", json={"question": "x" * 501}).status_code == 422


def test_query_pipeline_failure_returns_500():
    _use(BrokenPipeline())
    r = client.post("/query", json={"question": "anything here"})
    assert r.status_code == 500


def test_missing_vectorstore_returns_503(monkeypatch):
    monkeypatch.setattr(api, "VECTORSTORE_PATH", "does/not/exist")
    api.get_pipeline.cache_clear()
    r = client.post("/query", json={"question": "anything here"})
    assert r.status_code == 503
    assert "ingest.py" in r.json()["detail"]


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    _use(FakePipeline())
    payload = {"question": "How much is the budget?"}
    assert client.post("/query", json=payload).status_code == 401
    assert client.post("/query", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/query", json=payload, headers={"X-API-Key": "secret"}).status_code == 200


def test_api_key_not_required_when_unset():
    _use(FakePipeline())
    r = client.post("/query", json={"question": "How much is the budget?"})
    assert r.status_code == 200


def test_health_is_public_even_with_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.get("/health").status_code == 200
