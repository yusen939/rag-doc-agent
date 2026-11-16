from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_endpoint(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["documents"] == 0
    assert payload["vector_backend"] == "local"
