"""API tests. The real model is swapped for a deterministic fake so tests run without a GPU."""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.detector import HFDetector
from app.main import create_app


class FakeDetector(HFDetector):
    """Runs the real windowing/ensembling/aggregation code, but with fixed model scores."""

    def __init__(self, settings, score=0.9, second_model_score=None):
        super().__init__(settings)
        self.score = score
        self.second = second_model_score
        self.seen_chunks: list[str] = []

    def start_loading(self):
        self._models = ["fake"]  # anything non-empty
        self._device = "cpu"
        self._state = "ready"

    def _score_chunks(self, chunks):
        from app.detector import combine

        self.seen_chunks = chunks
        per_model = {"model-a": [self.score] * len(chunks)}
        if self.second is not None:
            per_model["model-b"] = [self.second] * len(chunks)
        combined = [combine(list(col), self.settings.ensemble) for col in zip(*per_model.values())]
        return combined, per_model


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def _make(score=0.9, second_model_score=None, **overrides):
        settings = Settings(db_path=tmp_path / f"db{len(clients)}.sqlite", **overrides)
        detector = FakeDetector(settings, score, second_model_score)
        client = TestClient(create_app(settings, detector))
        client.__enter__()
        client.detector = detector
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def save(client, doc, text, title="Doc", **extra):
    payload = {"title": title, "content": json.dumps({"ops": [{"insert": text}]}),
               "plain_text": text, "version": doc["version"], **extra}
    return client.put(f"/api/documents/{doc['id']}", json=payload)


# ---------------------------------------------------------------- documents
def test_health(client):
    body = client.get("/api/health").json()
    assert body["model"]["state"] == "ready"
    assert body["min_words"] == 300 and body["human_threshold"] == 0.2


def test_document_crud_roundtrip(client):
    doc = client.post("/api/documents", json={}).json()
    assert doc["title"] == "Untitled document" and doc["version"] == 1

    r = save(client, doc, "hello brave new world", title="My essay")
    assert r.status_code == 200
    assert r.json()["version"] == 2 and r.json()["word_count"] == 4

    got = client.get(f"/api/documents/{doc['id']}").json()
    assert got["title"] == "My essay" and got["plain_text"] == "hello brave new world"
    assert got["latest_scan"] is None

    listed = client.get("/api/documents").json()
    assert [d["id"] for d in listed] == [doc["id"]]
    assert "content" not in listed[0]  # list stays lightweight

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204
    assert client.get(f"/api/documents/{doc['id']}").status_code == 404
    assert client.delete(f"/api/documents/{doc['id']}").status_code == 404


def test_stale_version_is_rejected_then_forced(client):
    doc = client.post("/api/documents", json={}).json()
    assert save(client, doc, "device A").status_code == 200  # now version 2
    conflict = save(client, doc, "device B (stale)")  # still based on version 1
    assert conflict.status_code == 409
    assert conflict.json()["current"]["plain_text"] == "device A"

    forced = save(client, doc, "device B wins", force=True)
    assert forced.status_code == 200
    assert client.get(f"/api/documents/{doc['id']}").json()["plain_text"] == "device B wins"


def test_save_unknown_document_404(client):
    r = client.put("/api/documents/999", json={"title": "x", "content": "", "plain_text": "", "version": 1})
    assert r.status_code == 404


def test_oversized_document_rejected(make_client):
    client = make_client(max_content_bytes=100)
    doc = client.post("/api/documents", json={}).json()
    assert save(client, doc, "x" * 200).status_code == 413


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Scan for AI" in r.text
    assert client.get("/static/js/app.js").status_code == 200


# ---------------------------------------------------------------- scanning
def test_scan_requires_300_words(client):
    r = client.post("/api/scan", json={"text": words(299)})
    assert r.status_code == 422
    assert r.json()["code"] == "too_short" and r.json()["word_count"] == 299


def test_scan_ai_text_shows_percentage_and_chunks(make_client):
    client = make_client(score=0.83)
    r = client.post("/api/scan", json={"text": words(1000)})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "ai" and body["ai_probability"] == pytest.approx(0.83)
    assert body["chunk_count"] == 3 == len(body["chunks"])  # 500-word windows at 0, 350, 500
    assert all(len(c.split()) == 500 for c in client.detector.seen_chunks)
    assert body["model_scores"] == {"model-a": pytest.approx(0.83)}


def test_custom_window_size(make_client):
    client = make_client(score=0.9, chunk_words=200, chunk_overlap=50)
    body = client.post("/api/scan", json={"text": words(500)}).json()
    assert body["chunk_count"] == 3  # windows at 0, 150, 300
    assert all(len(c.split()) == 200 for c in client.detector.seen_chunks)


def test_windows_keep_paragraph_breaks(client):
    text = "\n\n".join(words(120) for _ in range(4))
    client.post("/api/scan", json={"text": text})
    assert "\n\n" in client.detector.seen_chunks[0]


def test_scan_below_threshold_withholds_score(make_client):
    client = make_client(score=0.12)
    body = client.post("/api/scan", json={"text": words(400)}).json()
    assert body["label"] == "Human-Written"
    assert body["ai_probability"] is None and body["chunks"] is None and body["model_scores"] is None
    assert "0.12" not in json.dumps(body)


def test_min_ensemble_requires_both_models_to_agree(make_client):
    # model A is sure it's AI, model B says human -> "min" rule clears the text
    client = make_client(score=0.97, second_model_score=0.05)
    body = client.post("/api/scan", json={"text": words(400)}).json()
    assert body["verdict"] == "human" and "0.97" not in json.dumps(body)


def test_mean_ensemble(make_client):
    client = make_client(score=0.9, second_model_score=0.5, ensemble="mean")
    body = client.post("/api/scan", json={"text": words(400)}).json()
    assert body["ai_probability"] == pytest.approx(0.7)
    assert body["model_scores"] == {"model-a": pytest.approx(0.9), "model-b": pytest.approx(0.5)}


def test_scan_is_persisted_and_marked_stale_after_edit(client):
    doc = client.post("/api/documents", json={}).json()
    text = words(320)
    doc = {**doc, **save(client, doc, text).json()}
    client.post("/api/scan", json={"text": text, "document_id": doc["id"]})

    latest = client.get(f"/api/documents/{doc['id']}").json()["latest_scan"]
    assert latest["result"]["verdict"] == "ai" and latest["is_stale"] is False

    save(client, doc, text + " extra")
    latest = client.get(f"/api/documents/{doc['id']}").json()["latest_scan"]
    assert latest["is_stale"] is True


def test_scan_while_model_loading_returns_503(client):
    client.detector._state = "loading"
    r = client.post("/api/scan", json={"text": words(400)})
    assert r.status_code == 503 and r.json()["code"] == "model_not_ready"


# ---------------------------------------------------------------- LAN token gate
def test_access_token_gate(make_client):
    client = make_client(access_token="s3cret")
    assert client.get("/api/documents").status_code == 401
    assert client.get("/").status_code == 401
    assert client.get("/api/documents", headers={"x-access-token": "nope"}).status_code == 401

    r = client.get("/?token=s3cret", follow_redirects=False)
    assert r.status_code == 303 and "token" not in r.headers["location"]
    client.cookies.set("scribescan_token", "s3cret")
    assert client.get("/api/documents").status_code == 200
