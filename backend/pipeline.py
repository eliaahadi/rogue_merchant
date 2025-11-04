import json
from time import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import dump, load

from backend.config import (
    EVENTS_PATH, PREPROCESSOR_PATH, CLASSIFIER_PATH, CLUSTERS_PATH,
    METRICS_PATH, PROCESSED
)

FEATURES_CAT = ["rarity"]
FEATURES_NUM = ["offered_price","base_price","player_dps","player_health_pct","gold","rooms_cleared","deaths"]
LABEL = "bought"

def read_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        return pd.DataFrame()
    rows = []
    with open(EVENTS_PATH, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    return df

def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # keep only valid offers
    df = df[df["event_type"] == "offer"].copy()
    # basic cleaning
    df["rarity"] = df["rarity"].astype("category")
    df = df.drop_duplicates(subset=["timestamp","run_id","item_id","offered_price"])
    df = df.dropna(subset=FEATURES_CAT + FEATURES_NUM + [LABEL])
    # clip numerics
    df["player_health_pct"] = df["player_health_pct"].clip(0, 100)
    df["deaths"] = df["deaths"].clip(0, 100)
    return df

def fit_player_segments(df: pd.DataFrame, k: int = 3) -> KMeans:
    seg_features = df[["gold","rooms_cleared","deaths","player_dps"]].copy()
    seg_features = seg_features.fillna(seg_features.median())
    scaler = StandardScaler()
    X = scaler.fit_transform(seg_features)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    # persist scaler inside model for convenience
    km.scaler_ = scaler
    return km

def train_models():
    df = read_events()
    df = build_dataset(df)
    if df.empty or df[LABEL].nunique() < 2:
        print("Not enough data to train.")
        return {"ok": False, "n_rows": 0, "metrics": {}}

    # Player segments
    km = fit_player_segments(df, k=3)
    seg_X = km.scaler_.transform(df[["gold","rooms_cleared","deaths","player_dps"]])
    segments = km.predict(seg_X)
    df["player_segment"] = pd.Series(segments, dtype="category")

    # Preprocessor
    cat_cols = FEATURES_CAT + ["player_segment"]
    num_cols = FEATURES_NUM

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    clf = LogisticRegression(max_iter=200, n_jobs=None, solver="lbfgs")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X = df[cat_cols + num_cols]
    y = df[LABEL].astype(int).values

    pipe.fit(X, y)

    # Metrics
    y_scores = pipe.predict_proba(X)[:,1]
    roc = roc_auc_score(y, y_scores)
    pr  = average_precision_score(y, y_scores)
    metrics = {"roc_auc": float(roc), "avg_precision": float(pr), "n_rows": int(len(df))}

    # Save artifacts
    dump(pipe.named_steps["pre"], PREPROCESSOR_PATH)
    dump(pipe.named_steps["clf"], CLASSIFIER_PATH)
    dump(km, CLUSTERS_PATH)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2))

    print("Trained classifier with ROC-AUC:", roc)
    return {"ok": True, "n_rows": int(len(df)), "metrics": metrics}

if __name__ == "__main__":
    out = train_models()
    print(out)
