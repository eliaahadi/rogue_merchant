import argparse, random, time, uuid
from economy_client import suggest_price, log_event, train

RARITIES = ["common","rare","epic","legendary"]
BASE_PRICES = {"common":20,"rare":50,"epic":100,"legendary":160}

def player_profile():
    # simple synthetic players with different price sensitivity
    archetype = random.choice(["hoarder","balanced","risk_taker"])
    if archetype == "hoarder":
        return {"dps": 15+random.random()*5, "health": 90, "gold": 300, "elasticity": 0.025}
    if archetype == "risk_taker":
        return {"dps": 35+random.random()*10, "health": 50, "gold": 120, "elasticity": 0.05}
    return {"dps": 25, "health": 70, "gold": 200, "elasticity": 0.035}

def buy_probability(price, base_price, rarity, player):
    # logistic-ish response around base price shifted by rarity and elasticity
    rarity_bias = {"common":0.0,"rare":0.5,"epic":0.8,"legendary":1.1}[rarity]
    x = (base_price - price) * player["elasticity"] + rarity_bias
    # squash
    return 1/(1+pow(2.71828, -x))

def simulate_episode(run_id: str, rooms=10):
    p = player_profile()
    rooms_cleared, deaths = 0, 0

    for room in range(rooms):
        rarity = random.choices(RARITIES, weights=[0.5,0.3,0.15,0.05])[0]
        base_price = BASE_PRICES[rarity]
        payload = {
            "run_id": run_id,
            "item_id": f"item-{random.randint(1,12)}",
            "rarity": rarity,
            "base_price": base_price,
            "player_dps": p["dps"],
            "player_health_pct": p["health"],
            "gold": p["gold"],
            "rooms_cleared": rooms_cleared,
            "deaths": deaths
        }
        # get ML price
        res = suggest_price(payload)
        offered = res["recommended_price"]
        prob = buy_probability(offered, base_price, rarity, p)
        bought = 1 if random.random() < prob and p["gold"] >= offered else 0
        if bought:
            p["gold"] -= offered

        event = {
            "event_type": "offer",
            "timestamp": time.time(),
            "run_id": run_id,
            "item_id": payload["item_id"],
            "rarity": rarity,
            "base_price": base_price,
            "offered_price": offered,
            "bought": bought,
            "player_dps": p["dps"],
            "player_health_pct": p["health"],
            "gold": p["gold"],
            "rooms_cleared": rooms_cleared,
            "deaths": deaths
        }
        log_event(event)
        # simple gameplay effects
        rooms_cleared += 1
        if random.random() < 0.05:
            deaths += 1
            p["health"] = max(10, p["health"] - 20)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    # try a quick train first (ok if not enough data)
    try:
        print("Initial train:", train())
    except Exception as e:
        print("Train skipped:", e)

    for _ in range(args.episodes):
        run_id = str(uuid.uuid4())[:8]
        simulate_episode(run_id, rooms=12)

    print("Final train:", train())
    print("Done.")