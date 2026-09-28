"""
Шаг 1. Локальная валидация, имитирующая бенчмарк. Единственный судья качества (ТЗ §2).

СОБЫТИЕ = набор строк train с одинаковыми (нормализованный запрос, локация поиска, фильтры).
Позитивы события = все item_id, выбранные в этих строках. Так восстанавливается структура
бенчмарка, где одна строка = один поисковый запрос, а выбранных объявлений может быть несколько.

Главное требование — воспроизвести долю «новых» запросов: в бенчмарке 37.5% запросов
встречались в train, 62.5% нет. Поэтому два типа валидационных событий:
  unseen (62.5%) — запрос удаляется из обучающей части ЦЕЛИКОМ (все его события);
  seen   (37.5%) — удаляется одно событие, остальные события того же запроса остаются в обучении.
Без этого разделения любая «история по запросу» покажет на валидации рост, которого не будет
на бенчмарке.

КОРПУС валидации = корпус бенчмарка (189 212) + позитивы валидации (их метаданные есть только
в train). Получается ~192k объявлений, из которых ~9.9% встречались в обучении — как на
бенчмарке (9.6%). Это защищает от иллюзии «все нужные объявления есть в train».

fit_mask.npy — булева маска строк train, РАЗРЕШЁННЫХ для обучения. Все статистики, словари,
классификаторы и модели строятся только по ней. Нарушение = утечка.

Запуск: python scripts/build_val.py --seed 42
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src import data as D
from src.paths import val_dir
from src.text import norm_query

N_VAL = 3000          # столько валидационных событий (ТЗ §2)
P_SEEN = 0.375        # доля seen-событий, как в бенчмарке


def main(seed: int) -> None:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    out = val_dir(seed)

    # --- 1. Ключевые колонки. Строки как category: 497k строк * 4 колонки иначе съедают
    #        гигабайты, а уникальных значений в разы меньше.
    key = D.read_train(columns=["search_query", "search_location_id",
                                "search_infm_params_text", "item_id"])
    for c in ("search_query", "search_infm_params_text", "item_id"):
        key[c] = key[c].fillna("").astype("category")

    # Нормализуем только УНИКАЛЬНЫЕ запросы (их ~десятки тысяч), затем разворачиваем по строкам.
    qmap = {q: norm_query(q) for q in key.search_query.cat.categories}
    key["qn"] = key.search_query.map(qmap).astype("category")
    key["ev"] = key.groupby(["qn", "search_location_id", "search_infm_params_text"],
                            observed=True).ngroup()

    ev_q = key.drop_duplicates("ev")[["ev", "qn"]]
    n_ev = ev_q.qn.value_counts()          # сколько событий у каждого нормализованного запроса

    # --- 2. Выбор unseen-запросов: любой запрос, все его события уйдут из обучения.
    n_unseen = int(N_VAL * (1 - P_SEEN))
    q_unseen = rng.choice(n_ev.index.to_numpy(), n_unseen, replace=False)
    ev_unseen = ev_q[ev_q.qn.isin(q_unseen)].groupby("qn", observed=True).ev.first()

    # --- 3. Выбор seen-запросов: нужны запросы с >=2 событиями, иначе после удаления
    #        единственного события запрос перестанет быть «виденным» в обучении.
    n_seen = N_VAL - n_unseen
    q_multi = np.setdiff1d(n_ev[n_ev >= 2].index.to_numpy(), q_unseen)
    q_seen = rng.choice(q_multi, n_seen, replace=False)
    ev_seen = ev_q[ev_q.qn.isin(q_seen)].groupby("qn", observed=True).ev.first()

    val_ev = set(ev_unseen) | set(ev_seen)

    # --- 4. Маска обучения: выкидываем все строки unseen-запросов и сами валидационные события.
    drop = key.qn.isin(q_unseen).to_numpy() | key.ev.isin(val_ev).to_numpy()
    fit_mask = ~drop
    np.save(out / "fit_mask.npy", fit_mask)

    # --- 5. Таблица валидационных событий.
    val = key[key.ev.isin(val_ev)].copy()
    for c in ("search_query", "qn", "search_infm_params_text", "item_id"):
        val[c] = val[c].astype(str)
    vq = (val.groupby("ev")
             .agg(search_query=("search_query", "first"),
                  qn=("qn", "first"),
                  search_location_id=("search_location_id", "first"),
                  search_infm_params_text=("search_infm_params_text", "first"),
                  pos=("item_id", lambda s: sorted(set(s))))
             .reset_index())
    vq["seen"] = vq.ev.isin(set(ev_seen))
    vq.to_parquet(out / "val_queries.parquet")

    pos_ids = {x for p in vq.pos for x in p}
    fit_items = set(key.item_id[fit_mask].astype(str))
    stats = {
        "seed": seed,
        "fit_rows": int(fit_mask.sum()),
        "dropped_rows": int(drop.sum()),
        "val_events": int(len(vq)),
        "seen_share": round(float(vq.seen.mean()), 4),
        "mean_positives": round(float(vq.pos.str.len().mean()), 3),
        "positives_unique": len(pos_ids),
    }
    del key, val
    gc.collect()

    # --- 6. Корпус валидации: бенчмарк + позитивы валидации из train.
    bi = D.read_bench_items(columns=D.ITEM_COLS)
    extra = D.train_items_subset(pos_ids, columns=D.ITEM_COLS).drop_duplicates("item_id")
    vc = D.concat_items(bi, extra)
    vc.to_parquet(out / "val_corpus.parquet")

    stats["corpus_size"] = int(len(vc))
    stats["corpus_seen_in_fit_share"] = round(float(vc.item_id.isin(fit_items).mean()), 4)
    stats["positives_in_corpus"] = int(vc.item_id.isin(pos_ids).sum())
    stats["seconds"] = round(time.time() - t0, 1)

    # --- 7. Проверка на утечку (ТЗ §1.1: assert в каждом прогоне).
    assert stats["positives_in_corpus"] == len(pos_ids), "часть позитивов не попала в корпус"
    assert abs(stats["seen_share"] - P_SEEN) < 0.01, "доля seen уехала от 0.375"
    assert stats["val_events"] == N_VAL, "число валидационных событий не 3000"

    with (out / "val_stats.json").open("w") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args().seed)
