PY := .venv/bin/python
SEED := 42
SEEDS := 42 1 2

.PHONY: venv answer val microcat keys analyze baseline text geo pool ranker verify all

venv:
	uv venv --python 3.12 .venv
	VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# Короткий путь: только то, что нужно для answer.csv. Около 4 минут.
answer:
	$(PY) scripts/build_val.py --seed $(SEED)
	$(PY) scripts/train_microcat.py --seed $(SEED)
	$(PY) scripts/build_pool.py --seed $(SEED) --pool 1200 --desc-chars 0 --geo-mode union
	$(PY) scripts/predict_ranker.py --train-seed $(SEED) --pool 1200 --out results/answer.csv

# Дальше — шаги подбора и проверки, по всем сидам.
val:
	for s in $(SEEDS); do $(PY) scripts/build_val.py --seed $$s; done

keys:
	$(PY) scripts/discover_param_keys.py

analyze:
	$(PY) scripts/analyze_data.py --seed $(SEED)

microcat: val
	for s in $(SEEDS); do $(PY) scripts/train_microcat.py --seed $$s; done

baseline: val
	$(PY) scripts/eval_bm25_baseline.py --seed $(SEED)

text: val
	$(PY) scripts/eval_desc.py --seed $(SEED) --radius 30 --desc-chars 0

geo: val
	$(PY) scripts/eval_geo.py --seed $(SEED) --desc-chars 0

pool: microcat
	for s in $(SEEDS); do $(PY) scripts/build_pool.py --seed $$s --pool 1200 --desc-chars 0 --geo-mode union; done

ranker: pool
	for s in $(SEEDS); do $(PY) scripts/train_ranker.py --seed $$s --pool 1200 --geo-mode union --neg-per-event 200; done

verify: val
	$(PY) scripts/verify_seeds.py --seeds $(SEEDS)

all: baseline text geo ranker answer
