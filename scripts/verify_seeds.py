"""
Проверка выбранного конфига на нескольких сидах валидационного сплита (ТЗ §2).

ЗАЧЕМ. Все решения выше принимались на сиде 42. Сид управляет тем, какие запросы попали в
unseen-часть и какие события вырезаны из обучения, — то есть влияет и на сложность валидации,
и на состав корпуса. Улучшение, которое не переносится на другие сиды, — переподгонка под сплит.
ТЗ: «сделай 2–3 разных сида и смотри среднее; улучшение < 0.005 считай шумом».

Считает ровно те варианты, между которыми мы выбираем, и не тратит время на остальную сетку.

Запуск: python scripts/verify_seeds.py --seeds 42 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D
from src.bm25 import BM25FBuilder
from src.geo import haversine_km, location_centers
from src.metrics import topk_from_scores
from src.params import parse_many
from src.paths import RESULTS, val_dir
from src.text import Lemmatizer

KS = (50, 200, 1000)
BATCH = 200
B_BY_FIELD = {"title": 0.3, "kind": 0.3, "service": 0.75, "place": 0.5, "other": 0.75, "desc": 0.9}
WEIGHTS = {"title": 3, "kind": 2, "service": 2, "place": 0.5, "other": 0.3, "desc": 1}

# Ровно те варианты, из которых выбираем финальный. baseline_tz — для сверки с ТЗ §2.
CONFIGS = {
    "baseline_tz_r80":   dict(radius=80.0, mu=0.0),
    "hard_r30":          dict(radius=30.0, mu=0.0),
    "hard_r50":          dict(radius=50.0, mu=0.0),
    "hard_r50_soft1":    dict(radius=50.0, mu=1.0),
    "hard_r50_soft0.5":  dict(radius=50.0, mu=0.5),
    "soft_mu1":          dict(radius=None, mu=1.0),
}


def run_seed(seed: int) -> pd.DataFrame:
    t0 = time.time()
    out = val_dir(seed)
    vq = pd.read_parquet(out / "val_queries.parquet")
    vc = pd.read_parquet(out / "val_corpus.parquet")
    fit_mask = np.load(out / "fit_mask.npy")
    n_q = len(vq)

    lm = Lemmatizer()
    parsed = parse_many(vc.item_infm_params_text.fillna("").tolist())
    fields = {
        "title": lm.many(vc.item_title_raw.fillna("").tolist()),
        "kind": lm.many(parsed["kind"]), "service": lm.many(parsed["service"]),
        "place": lm.many(parsed["place"]), "other": lm.many(parsed["other"]),
        "desc": lm.many(vc.item_description_raw.fillna("").tolist()),
    }
    qs = lm.many(vq.search_query.tolist())
    lm.save()
    builder = BM25FBuilder().fit(fields, b=B_BY_FIELD)
    W = builder.combine(WEIGHTS)
    Q = builder.transform_queries(qs)
    del fields, parsed, builder

    g = D.read_train(columns=["search_location_id", "item_latitude", "item_longitude"])[fit_mask]
    qc = location_centers(g).reindex(vq.search_location_id)
    qlat, qlon = np.radians(qc.lat.to_numpy()), np.radians(qc.lon.to_numpy())
    ilat, ilon = np.radians(vc.item_latitude.to_numpy()), np.radians(vc.item_longitude.to_numpy())
    ids = vc.item_id.to_numpy()
    positives = [set(p) for p in vq.pos]
    seen = vq.seen.to_numpy()

    hits = {c: {k: np.zeros(n_q) for k in KS} for c in CONFIGS}
    for s in range(0, n_q, BATCH):
        e = min(s + BATCH, n_q)
        S = (Q[s:e] @ W).toarray()
        for j in range(e - s):
            i = s + j
            d = (np.full(len(vc), 1e4, dtype=np.float32) if np.isnan(qlat[i])
                 else haversine_km(qlat[i], qlon[i], ilat, ilon).astype(np.float32))
            ld = np.log1p(d)
            for c, p in CONFIGS.items():
                x = S[j].copy()
                if p["radius"] is not None:
                    x += 1000.0 * (d < p["radius"])
                if p["mu"]:
                    x -= p["mu"] * ld
                top = topk_from_scores(x, ids, max(KS))
                for k in KS:
                    hits[c][k][i] = len(positives[i] & set(top[:k])) / len(positives[i])
    rows = []
    for c in CONFIGS:
        r = {"seed": seed, "config": c}
        for k in KS:
            r[f"R@{k}"] = float(hits[c][k].mean())
        r["R@50_seen"] = float(hits[c][50][seen].mean())
        r["R@50_unseen"] = float(hits[c][50][~seen].mean())
        rows.append(r)
    print(f"сид {seed} готов за {time.time()-t0:.0f} с")
    return pd.DataFrame(rows)


def main(seeds: list[int]) -> None:
    parts = [run_seed(s) for s in seeds]
    all_res = pd.concat(parts, ignore_index=True)
    exp = RESULTS / "verify_seeds"
    exp.mkdir(parents=True, exist_ok=True)
    all_res.to_csv(exp / "per_seed.csv", index=False)

    agg = (all_res.groupby("config")[["R@50", "R@200", "R@1000", "R@50_seen", "R@50_unseen"]]
           .agg(["mean", "std"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.sort_values("R@50_mean", ascending=False)
    agg.to_csv(exp / "aggregated.csv")
    with (exp / "meta.json").open("w") as fh:
        json.dump({"seeds": seeds, "weights": WEIGHTS, "b_by_field": B_BY_FIELD,
                   "configs": {k: {kk: vv for kk, vv in v.items()} for k, v in CONFIGS.items()}},
                  fh, ensure_ascii=False, indent=2)
    print("\n=== по сидам ===")
    print(all_res.pivot(index="config", columns="seed", values="R@50").round(4).to_string())
    print("\n=== среднее по сидам ===")
    print(agg[["R@50_mean", "R@50_std", "R@1000_mean", "R@50_seen_mean", "R@50_unseen_mean"]]
          .round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    main(ap.parse_args().seeds)
