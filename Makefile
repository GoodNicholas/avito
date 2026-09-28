PY := .venv/bin/python
SEEDS := 42 1 2

.PHONY: venv val microcat baseline text geo pool ranker verify answer all

venv:
	uv venv --python 3.12 .venv
	VIRTUAL_ENV=.venv uv pip install -r requirements.txt

val:
	for s in $(SEEDS); do $(PY) scripts/build_val.py --seed $$s; done

keys:
	$(PY) scripts/discover_param_keys.py

microcat: val
	for s in $(SEEDS); do $(PY) scripts/train_microcat.py --seed $$s; done

baseline: val
	$(PY) scripts/eval_bm25_baseline.py --seed 42

text: val
	$(PY) scripts/eval_desc.py --seed 42 --radius 30 --desc-chars 0

geo: val
	$(PY) scripts/eval_geo.py --seed 42 --desc-chars 0

pool: microcat
	for s in $(SEEDS); do $(PY) scripts/build_pool.py --seed $$s --pool 1200 --desc-chars 0 --geo-mode union; done

ranker: pool
	for s in $(SEEDS); do $(PY) scripts/train_ranker.py --seed $$s --pool 1200 --geo-mode union --neg-per-event 200; done

verify: val
	$(PY) scripts/verify_seeds.py --seeds $(SEEDS)

answer: pool
	$(PY) scripts/predict_ranker.py --train-seed 42 --pool 1200 --out results/answer.csv

all: baseline text geo ranker answer
