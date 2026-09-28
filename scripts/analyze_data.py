"""
Анализ данных, из которого следуют решения по архитектуре, и разбор промахов.

Считает две вещи:

  A. Свойства данных, определяющие устройство решения: насколько полезна история по item_id,
     сколько запросов бенчмарка новые, насколько локальны услуги, хватает ли заголовка,
     насколько предсказуема микрокатегория. Всё считается по своим данным: брать такие числа на веру
     нельзя.

  B. Таксономию промахов точки отсчёта (BM25 по заголовку и сырым параметрам + приоритет радиуса
     80 км): каждый позитив, не попавший в top-50, относим к одной причине. Причины упорядочены
     по тому, каким каналом их можно закрыть, и сумма даёт весь разрыв до 1.0.

Запуск: python scripts/analyze_data.py --seed 42
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
from src.paths import RESULTS, val_dir
from src.text import Lemmatizer, norm_query

BATCH = 200


def part_a() -> dict:
    """Свойства данных. Каждая строка — довод в пользу конкретного решения."""
    out = {}
    tr = D.read_train(columns=["search_query", "search_location_id", "item_id",
                               "item_location_id", "item_microcat_id",
                               "item_latitude", "item_longitude"])
    bi = D.read_bench_items(columns=["item_id", "item_title_raw", "item_location_id"])
    bq = D.read_bench_queries()

    # 1. Есть ли смысл в истории «запрос -> конкретное объявление»
    out["корпус бенчмарка, встречавшийся в train"] = float(bi.item_id.isin(set(tr.item_id)).mean())

    # 2. Сколько запросов бенчмарка вообще знакомы обучению
    train_qn = {norm_query(q) for q in tr.search_query.fillna("").unique()}
    out["запросы бенчмарка, встречавшиеся в train"] = float(
        np.mean([norm_query(q) in train_qn for q in bq.search_query]))

    # 3. Насколько локальны услуги
    out["пары train с совпадением локации поиска и объявления"] = float(
        (tr.search_location_id == tr.item_location_id).mean())

    # 4. Можно ли фильтровать по локации жёстко
    loc_counts = bi.item_location_id.value_counts()
    out["запросы бенчмарка без объявлений в своей локации"] = float(
        (~bq.search_location_id.isin(loc_counts.index)).mean())

    # 5. Хватает ли заголовка, чтобы отличить объявление
    tc = bi.item_title_raw.fillna("").value_counts()
    out["объявления корпуса с неуникальным заголовком"] = float(
        bi.item_title_raw.fillna("").map(tc).gt(1).mean())

    # 6. Насколько запрос определяет микрокатегорию
    g = tr.groupby(tr.search_query.fillna("").map(norm_query), observed=True).item_microcat_id
    dom = g.apply(lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else np.nan)
    sizes = g.size()
    out["выборы на доминирующую microcat запроса"] = float(
        (dom * sizes).sum() / sizes.sum())

    # 7. Как далеко бывает позитив от центра локации запроса
    cent = location_centers(tr[["search_location_id", "item_latitude", "item_longitude"]])
    c = cent.reindex(tr.search_location_id)
    d = haversine_km(np.radians(c.lat.to_numpy()), np.radians(c.lon.to_numpy()),
                     np.radians(tr.item_latitude.to_numpy()),
                     np.radians(tr.item_longitude.to_numpy()))
    for q in (50, 75, 90, 95, 99):
        out[f"расстояние до позитива, процентиль {q}, км"] = float(np.nanpercentile(d, q))
    return out


def part_b(seed: int) -> dict:
    """Таксономия промахов точки отсчёта: почему позитив не попал в top-50."""
    o = val_dir(seed)
    vq = pd.read_parquet(o / "val_queries.parquet")
    vc = pd.read_parquet(o / "val_corpus.parquet")
    fit = np.load(o / "fit_mask.npy")

    lm = Lemmatizer()
    docs = lm.many((vc.item_title_raw.fillna("") + " " + vc.item_infm_params_text.fillna("")).tolist())
    qs = lm.many(vq.search_query.tolist())
    lm.save()
    ix = single_field_index(docs)
    Q = ix.transform_queries(qs)

    g = D.read_train(columns=["search_location_id", "item_latitude", "item_longitude"])[fit]
    qc = location_centers(g).reindex(vq.search_location_id)
    qlat, qlon = np.radians(qc.lat.to_numpy()), np.radians(qc.lon.to_numpy())
    ilat, ilon = np.radians(vc.item_latitude.to_numpy()), np.radians(vc.item_longitude.to_numpy())
    id2i = {x: i for i, x in enumerate(vc.item_id)}

    n_pos = 0
    cause = {"нет общих лемм с запросом": 0, "дальше 80 км": 0,
             "в пуле top-1000, но ниже 50": 0, "вне пула top-1000": 0}
    in_top50 = 0
    for s in range(0, len(vq), BATCH):
        e = min(s + BATCH, len(vq))
        S = (Q[s:e] @ ix.W).toarray()
        for j in range(e - s):
            i = s + j
            d = (np.full(len(vc), 1e4, dtype=np.float32) if np.isnan(qlat[i])
                 else haversine_km(qlat[i], qlon[i], ilat, ilon).astype(np.float32))
            x = S[j] + 1000.0 * (d < 80)
            for p in vq.pos.iloc[i]:
                k = id2i.get(p)
                if k is None:
                    continue
                n_pos += 1
                rank = int((x > x[k]).sum())
                if rank < 50:
                    in_top50 += 1
                elif S[j][k] == 0:
                    cause["нет общих лемм с запросом"] += 1
                elif d[k] >= 80:
                    cause["дальше 80 км"] += 1
                elif rank < 1000:
                    cause["в пуле top-1000, но ниже 50"] += 1
                else:
                    cause["вне пула top-1000"] += 1
    return {"позитивов всего": n_pos,
            "попали в top-50": in_top50 / n_pos,
            **{k: v / n_pos for k, v in cause.items()}}


def main(seed: int) -> None:
    t0 = time.time()
    a = part_a()
    print("=== A. Свойства данных ===")
    for k, v in a.items():
        print(f"  {k:60s} {v:.4f}" if v < 1 else f"  {k:60s} {v:.1f}")
    b = part_b(seed)
    print("\n=== B. Куда деваются позитивы у точки отсчёта (BM25 + радиус 80 км) ===")
    for k, v in b.items():
        print(f"  {k:40s} {v}" if isinstance(v, int) else f"  {k:40s} {v:.4f}")
    exp = RESULTS / "data_analysis"
    exp.mkdir(parents=True, exist_ok=True)
    with (exp / "report.json").open("w") as fh:
        json.dump({"data_facts": a, "loss_taxonomy": b, "seed": seed}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\nготово за {time.time()-t0:.0f} с -> {exp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
