from fastapi.testclient import TestClient
from app.api import app
client=TestClient(app)
def test_health(): assert client.get("/health").json()["status"]=="ok"
def test_all_country_prediction():
    r=client.get("/predict"); assert r.status_code==200; assert r.json()["country"]=="all"
def test_specific_country_prediction():
    r=client.get("/predict?country=Singapore"); assert r.status_code==200; assert r.json()["country"]=="Singapore"
