from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
MODELS = ROOT / "models"

RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# Files
EVENTS_PATH = RAW / "events.jsonl"
METRICS_PATH = PROCESSED / "metrics.json"
PREPROCESSOR_PATH = MODELS / "preprocessor.joblib"
CLASSIFIER_PATH = MODELS / "buy_classifier.joblib"
CLUSTERS_PATH = MODELS / "player_clusters.joblib"

# Serving config
PRICE_MIN = 5.0
PRICE_MAX = 200.0
PRICE_STEP = 1.0
TARGET_BUY_PROB = 0.35  # aim for ~35% buy-through per item