PHONY_TARGETS=install run-api train simulate dashboard
.PHONY: $(PHONY_TARGETS)

PY=python3

install:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

run-api:
	. .venv/bin/activate && uvicorn backend.app:app --reload --port 8000

train:
	. .venv/bin/activate && $(PY) backend/pipeline.py

simulate:
	. .venv/bin/activate && $(PY) sim/game.py --episodes 10

dashboard:
	. .venv/bin/activate && streamlit run dashboard/dashboard.py
