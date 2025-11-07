# Rogue Merchant — ML-driven Dynamic Shop (Python)

A tiny portfolio project that proves end-to-end ML skills in a game context:
- Sim generates **offer events** like a roguelite shop
- FastAPI service **trains & serves** a buy-probability model + player segments
- A pricing endpoint chooses a price that targets a buy-through rate
- Streamlit **dashboard** shows economy health and model metrics

## Quickstart
```bash
make install
make run-api          # in one terminal
make simulate         # in another terminal, generates events and auto-trains
make dashboard        # Streamlit economy board

