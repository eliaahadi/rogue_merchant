from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict

Rarity = Literal["common", "rare", "epic", "legendary"]

class OfferEvent(BaseModel):
    event_type: Literal["offer"] = "offer"
    timestamp: float
    run_id: str
    item_id: str
    rarity: Rarity
    base_price: float
    offered_price: float
    bought: int = Field(ge=0, le=1)
    # minimal player state snapshot
    player_dps: float
    player_health_pct: float
    gold: float
    rooms_cleared: int
    deaths: int

class SuggestPriceRequest(BaseModel):
    run_id: str
    item_id: str
    rarity: Rarity
    base_price: float
    player_dps: float
    player_health_pct: float
    gold: float
    rooms_cleared: int
    deaths: int
    target_buy_prob: Optional[float] = None  # overrides default

class TrainResponse(BaseModel):
    ok: bool
    n_rows: int
    metrics: Dict[str, float]

class SuggestPriceResponse(BaseModel):
    recommended_price: float
    predicted_buy_prob: float
    target_buy_prob: float
    price_grid: int
    model_loaded: bool
    model_config = {"protected_namespaces": ()}
