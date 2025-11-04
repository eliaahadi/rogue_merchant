import requests

API = "http://127.0.0.1:8000"

def suggest_price(payload: dict) -> dict:
    r = requests.post(f"{API}/suggest_price", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

def log_event(event: dict) -> None:
    r = requests.post(f"{API}/log_event", json=event, timeout=10)
    r.raise_for_status()

def train() -> dict:
    r = requests.post(f"{API}/train", timeout=60)
    r.raise_for_status()
    return r.json()