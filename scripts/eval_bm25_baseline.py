"""
Шаг 2 (ТЗ §6.1). Воспроизведение baseline-таблицы из ТЗ §2 на локальном окружении.
Цифры должны совпасть с заявленными ±0.005, иначе валидация построена иначе и всё
дальнейшее сравнение бессмысленно.

Считаем BM25 по слитному полю (заголовок + параметры, леммы) и пять гео-режимов:
  global          — без гео, чистый BM25;
  same_loc_first  — объявления своей локации поиска идут вперёд (score + 1000);
  r30 / r80 / r200 — вперёд идут объявления в радиусе R км от центра локации запроса.
Бонус 1000 заведомо больше любого BM25-скора, поэтому это именно ПРИОРИТЕТ (жёсткий
двухуровневый порядок), а не мягкая добавка: внутри группы порядок остаётся по BM25.
Жёсткий ФИЛЬТР по локации запрещён (у 17.4% запросов бенчмарка в своей локации 0 объявлений),
а приоритет безопасен: остальные объявления просто уходят ниже, но остаются в пуле.

Запуск: python scripts/eval_bm25_baseline.py --seed 42
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
from src.bm25 import single_field_index
from src.geo import haversine_km, location_centers
from src.metrics import topk_from_scores
from src.paths import RESULTS, val_dir
from src.text import Lemmatizer

KS = (50, 200, 1000)
VARIANTS = {"global": None, "same_loc_first": "loc", "r30km": 30.0, "r80km": 80.0, "r200km": 200.0}
BATCH = 250          # запросов за раз: 250 x 192k float32 = ~190 МБ на матрицу скоров


def main(seed: int) -> None:
    t0 = time.time()
    out = val_dir(seed)
    vq = pd.read_parquet(out / "val_queries.parquet")
    vc = pd.read_parquet(out / "val_corpus.parquet")
    fit_mask = np.load(out / "fit_mask.npy")

    # --- 1. Лемматизация корпуса и запросов -------------------------------
    lm = Lemmatizer()
    docs = (vc.item_title_raw.fillna("") + " " + vc.item_infm_params_text.fillna("")).tolist()
    docs = lm.many(docs)
    qs = lm.many(vq.search_query.tolist())
    lm.save()
    print(f"лемматизация: {time.time()-t0:.0f} с, уникальных токенов {lm.vocab_size}")

    # --- 2. BM25-индекс ----------------------------------------------------
    ix = single_field_index(docs, k1=1.2, b=0.75)
    Q = ix.transform_queries(qs)
    print(f"индекс: {ix.W.shape[0]} терминов x {ix.W.shape[1]} документов, {time.time()-t0:.0f} с")

    # --- 3. Центры локаций поиска — ТОЛЬКО по обучающей части -------------
    g = D.read_train(columns=["search_location_id", "item_latitude", "item_longitude"])[fit_mask]
    cent = location_centers(g)
    qc = cent.reindex(vq.search_location_id)
    print(f"валидационных запросов без центра локации: {qc.lat.isna().mean():.4f}")

    qlat, qlon = np.radians(qc.lat.to_numpy()), np.radians(qc.lon.to_numpy())
    ilat = np.radians(vc.item_latitude.to_numpy())
    ilon = np.radians(vc.item_longitude.to_numpy())
    iloc = vc.item_location_id.to_numpy()
    ids = vc.item_id.to_numpy()
    qloc = vq.search_location_id.to_numpy()
    positives = [set(p) for p in vq.pos]

    # --- 4. Прогон ---------------------------------------------------------
    hits = {v: {k: np.zeros(len(vq)) for k in KS} for v in VARIANTS}
    for s in range(0, len(vq), BATCH):
        S = ix.score_batch(Q, s, min(s + BATCH, len(vq)))
        for j in range(S.shape[0]):
            i = s + j
            sc = S[j]
            pos = positives[i]
            d = None if np.isnan(qlat[i]) else haversine_km(qlat[i], qlon[i], ilat, ilon)
            for v, mode in VARIANTS.items():
                x = sc
                if mode == "loc":
                    x = sc + 1000.0 * (iloc == qloc[i])
                elif mode is not None and d is not None:
                    x = sc + 1000.0 * (d < mode)
                top = topk_from_scores(x, ids, max(KS))
                for k in KS:
                    hits[v][k][i] = len(pos & set(top[:k])) / len(pos)
        if s % 1000 == 0:
            print(f"  {s}/{len(vq)} событий, {time.time()-t0:.0f} с")

    # --- 5. Отчёт ----------------------------------------------------------
    seen = vq.seen.to_numpy()
    rows = []
    for v in VARIANTS:
        r = {"variant": v}
        for k in KS:
            r[f"R@{k}"] = float(hits[v][k].mean())
        r["R@50_seen"] = float(hits[v][50][seen].mean())
        r["R@50_unseen"] = float(hits[v][50][~seen].mean())
        rows.append(r)
    res = pd.DataFrame(rows).set_index("variant")
    print("\n" + res.round(4).to_string())

    exp = RESULTS / f"baseline_bm25_seed{seed}"
    exp.mkdir(parents=True, exist_ok=True)
    res.to_csv(exp / "summary.csv")
    with (exp / "meta.json").open("w") as fh:
        json.dump({"seed": seed, "k1": 1.2, "b": 0.75, "corpus": int(len(vc)),
                   "events": int(len(vq)), "seconds": round(time.time() - t0, 1)}, fh, indent=2)
    np.savez_compressed(exp / "per_event.npz",
                        **{f"{v}_R{k}": hits[v][k] for v in VARIANTS for k in KS}, seen=seen)
    print(f"\nготово за {time.time()-t0:.0f} с -> {exp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
