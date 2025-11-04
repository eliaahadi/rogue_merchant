from fastapi import FastAPI
from fastapi.responses import JSONResponse
from joblib import load
import json, time
import numpy as np
import pandas as pd
from typing import Dict

from backend.schema import OfferEvent, SuggestPriceRequest, TrainResponse, SuggestPriceResponse
from backend.config import (
    EVENTS_PATH, PREPROCESSOR_PATH, CLASSIFIER_PATH, CLUSTERS_PATH,
    METRICS_PATH, PRICE_MIN, PRICE_MAX, PRICE_STEP, TARGET_BUY_PROB
)

app = FastAPI(title="Rogue Merchant ML API", version="0.1.0")

def load_artifacts():
    try:
        pre = load(PREPROCESSOR_PATH)
        clf = load(CLASSIFIER_PATH)
        km  = load(CLUSTERS_PATH)
        return pre, clf, km
    except Exception:
        return None, None, None

def features_df(req: SuggestPriceRequest, price: float, km_model) -> pd.DataFrame:
    base = {
        "rarity": req.rarity,
        "offered_price": price,
        "base_price": req.base_price,
        "player_dps": req.player_dps,
        "player_health_pct": req.player_health_pct,
        "gold": req.gold,
        "rooms_cleared": req.rooms_cleared,
        "deaths": req.deaths,
    }
    # derive player_segment from kmeans if available
    if km_model is not None and hasattr(km_model, "scaler_"):
        x = np.array([[req.gold, req.rooms_cleared, req.deaths, req.player_dps]])
        seg = int(km_model.predict(km_model.scaler_.transform(x))[0])
    else:
        seg = 0
    base["player_segment"] = seg
    return pd.DataFrame([base])

@app.get("/health")
def health():
    pre, clf, km = load_artifacts()
    return {"ok": True, "model_loaded": pre is not None}

@app.post("/log_event")
def log_event(event: OfferEvent):
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_PATH, "a") as f:
        f.write(json.dumps(event.model_dump()) + "\n")
    return {"ok": True}

@app.post("/train", response_model=TrainResponse)
def train():
    from backend.pipeline import train_models
    res = train_models()
    return res

@app.post("/suggest_price", response_model=SuggestPriceResponse)
def suggest_price(req: SuggestPriceRequest):
    pre, clf, km = load_artifacts()
    target = req.target_buy_prob if req.target_buy_prob is not None else TARGET_BUY_PROB

    if pre is None or clf is None:
        # fall back: base price clamped
        price = float(np.clip(req.base_price, PRICE_MIN, PRICE_MAX))
        return SuggestPriceResponse(
            recommended_price=price,
            predicted_buy_prob=0.0,
            target_buy_prob=target,
            price_grid=0,
            model_loaded=False
        )

    # evaluate a price grid and pick price closest to target
    grid = np.arange(PRICE_MIN, PRICE_MAX + 1e-9, PRICE_STEP)
    rows = []
    for p in grid:
        X = features_df(req, p, km)
        Xt = pre.transform(X[["rarity","player_segment","offered_price","base_price","player_dps","player_health_pct","gold","rooms_cleared","deaths"]])
        prob = float(clf.predict_proba(Xt)[:,1][0])
        rows.append((p, prob))
    arr = np.array(rows)
    idx = np.argmin(np.abs(arr[:,1] - target))
    chosen_price, chosen_prob = float(arr[idx,0]), float(arr[idx,1])

    return SuggestPriceResponse(
        recommended_price=chosen_price,
        predicted_buy_prob=chosen_prob,
        target_buy_prob=target,
        price_grid=len(grid),
        model_loaded=True
    )