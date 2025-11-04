import json
import pandas as pd
import streamlit as st
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
EVENTS = DATA / "raw" / "events.jsonl"
METRICS = DATA / "processed" / "metrics.json"

st.title("Rogue Merchant Economy Board")

if EVENTS.exists():
    df = pd.read_json(EVENTS, lines=True)
    st.metric("Total offers", len(df))
    st.metric("Buy-through rate", f"{100*df['bought'].mean():.1f}%")
    st.bar_chart(df["rarity"].value_counts())
    st.line_chart(df.groupby(pd.to_datetime(df["timestamp"], unit="s").dt.floor("min"))["bought"].mean())
    st.subheader("Price vs Buy")
    st.scatter_chart(df[["offered_price","bought"]])
    st.subheader("Sample rows")
    st.dataframe(df.tail(20))
else:
    st.info("No events yet. Run the simulator.")

if METRICS.exists():
    st.subheader("Model metrics")
    st.json(json.loads(METRICS.read_text()))
else:
    st.warning("Train a model to see metrics.")