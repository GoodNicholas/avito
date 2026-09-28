"""
Эксперимент: гео-сигнал — жёсткий приоритет радиуса против мягкой функции расстояния.

ПОВОД. Baseline ТЗ использует грубый приём: score + 1000*(dist < R). Это двухуровневый
порядок — «сначала всё в радиусе, потом всё остальное». У него два недостатка:
  1) внутри радиуса расстояние вообще не учитывается, хотя медиана до позитива 3.9 км,
     а 75-й процентиль 10 км — то есть близкое почти всегда лучше далёкого;
  2) на границе радиуса разрыв: объявление в 29 км обгоняет объявление в 31 км с втрое
     большим BM25.
Проверяем мягкие варианты: bm25 − mu*log(1+dist), бонус за свою локацию, комбинации.
4.6% позитивов лежат дальше 80 км (онлайн-услуги) — мягкая функция даёт им шанс, жёсткий
радиус нет.

Запуск: python scripts/eval_geo.py --seed 42 --desc-chars 400
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


def main(seed: int, desc_chars: int) -> None:
    t0 = time.time()
    out = val_dir(seed)
    vq = pd.read_parquet(out / "val_queries.parquet")
    vc = pd.read_parquet(out / "val_corpus.parquet")
    fit_mask = np.load(out / "fit_mask.npy")
    n_q, n_d = len(vq), len(vc)

    lm = Lemmatizer()
    parsed = parse_many(vc.item_infm_params_text.fillna("").tolist())
    desc_raw = vc.item_description_raw.fillna("")
    fields = {
        "title": lm.many(vc.item_title_raw.fillna("").tolist()),
        "kind": lm.many(parsed["kind"]), "service": lm.many(parsed["service"]),
        "place": lm.many(parsed["place"]), "other": lm.many(parsed["other"]),
        "desc": lm.many((desc_raw if desc_chars == 0 else desc_raw.str[:desc_chars]).tolist()),
    }
    qs = lm.many(vq.search_query.tolist())
    lm.save()
    builder = BM25FBuilder().fit(fields, b=B_BY_FIELD)
    W = builder.combine(WEIGHTS)
    Q = builder.transform_queries(qs)
    del fields, parsed, builder
    print(f"индекс готов, {time.time()-t0:.0f} с")

    # --- расстояния: матрица считается один раз, дальше варианты бесплатны ---
    g = D.read_train(columns=["search_location_id", "item_latitude", "item_longitude"])[fit_mask]
    qc = location_centers(g).reindex(vq.search_location_id)
    qlat, qlon = np.radians(qc.lat.to_numpy()), np.radians(qc.lon.to_numpy())
    ilat, ilon = np.radians(vc.item_latitude.to_numpy()), np.radians(vc.item_longitude.to_numpy())
    dist = np.empty((n_q, n_d), dtype=np.float32)
    for i in range(n_q):
        dist[i] = 1e4 if np.isnan(qlat[i]) else haversine_km(qlat[i], qlon[i], ilat, ilon)
    log_dist = np.log1p(dist)
    same_loc = (vc.item_location_id.to_numpy()[None, :] == vq.search_location_id.to_numpy()[:, None])
    print(f"матрица расстояний готова, {time.time()-t0:.0f} с ({dist.nbytes/1e9:.1f} ГБ)")

    ids = vc.item_id.to_numpy()
    positives = [set(p) for p in vq.pos]
    seen = vq.seen.to_numpy()
    rows = []

    # Каждый вариант — функция (bm25_batch, срез) -> итоговый скор.
    def hard(r):      return lambda S, sl: S + 1000.0 * (dist[sl] < r)
    def hard_soft(r, mu): return lambda S, sl: S + 1000.0 * (dist[sl] < r) - mu * log_dist[sl]
    def soft(mu):     return lambda S, sl: S - mu * log_dist[sl]
    def soft_loc(mu, a): return lambda S, sl: S - mu * log_dist[sl] + a * same_loc[sl]

    variants = {
        "hard_r20": hard(20), "hard_r30": hard(30), "hard_r50": hard(50), "hard_r80": hard(80),
        "hard_r30_soft0.5": hard_soft(30, 0.5), "hard_r30_soft2": hard_soft(30, 2.0),
        "hard_r50_soft1": hard_soft(50, 1.0),
        "soft_mu1": soft(1.0), "soft_mu3": soft(3.0), "soft_mu6": soft(6.0), "soft_mu10": soft(10.0),
        "soft_mu3_loc2": soft_loc(3.0, 2.0), "soft_mu6_loc3": soft_loc(6.0, 3.0),
    }

    hits = {v: {k: np.zeros(n_q) for k in KS} for v in variants}
    for s in range(0, n_q, BATCH):
        e = min(s + BATCH, n_q)
        sl = slice(s, e)
        S = (Q[sl] @ W).toarray()
        for name, fn in variants.items():
            X = fn(S, sl)
            for j in range(e - s):
                i = s + j
                top = topk_from_scores(X[j], ids, max(KS))
                for k in KS:
                    hits[name][k][i] = len(positives[i] & set(top[:k])) / len(positives[i])
        if s % 1000 == 0:
            print(f"  {s}/{n_q}, {time.time()-t0:.0f} с")

    for name in variants:
        r = {"config": name}
        for k in KS:
            r[f"R@{k}"] = float(hits[name][k].mean())
        r["R@50_seen"] = float(hits[name][50][seen].mean())
        r["R@50_unseen"] = float(hits[name][50][~seen].mean())
        rows.append(r)
    res = pd.DataFrame(rows).set_index("config").sort_values("R@50", ascending=False)
    exp = RESULTS / f"geo_seed{seed}_d{desc_chars}"
    exp.mkdir(parents=True, exist_ok=True)
    res.to_csv(exp / "summary.csv")
    with (exp / "meta.json").open("w") as fh:
        json.dump({"seed": seed, "desc_chars": desc_chars, "weights": WEIGHTS,
                   "b_by_field": B_BY_FIELD, "seconds": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"\n{res.round(4).to_string()}\n\nготово за {time.time()-t0:.0f} с -> {exp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--desc-chars", type=int, default=400)
    a = ap.parse_args()
    main(a.seed, a.desc_chars)
