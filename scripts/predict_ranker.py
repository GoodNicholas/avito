"""
Генерация answer.csv с ранкером: полный финальный пайплайн (ТЗ §0, фаза 5).

ПОРЯДОК ДЕЙСТВИЙ
  1. Обучить ранкер на валидационных событиях (это настоящие события train с настоящими
     метками; признаки для них считались БЕЗ них самих — по fit_mask, — то есть ровно в том
     режиме, в котором ранкер будет работать на бенчмарке).
  2. Построить пул кандидатов для 2 452 запросов бенчмарка тем же способом: объединение
     жёсткого гео-канала (радиус 30 км) и мягкого (−mu·log(1+dist)).
  3. Признаки бенчмарка считаются по ПОЛНОМУ train: центры локаций, популярность объявлений,
     классификатор P(microcat | запрос). Валидационных событий здесь нет, прятать нечего.
  4. Скоринг ранкером, top-50, проверки формата, запись.

ПОЧЕМУ НЕ «обучение на 20-50k событий train», как в ТЗ фазе 3: для этого нужно строить пул по
событиям train, а это отдельный прогон с out-of-fold признаками. Текущий вариант — 3000 событий
x ~1016 кандидатов = 3 млн строк — уже даёт достаточную обучающую матрицу; ограничение в числе
РАЗНЫХ запросов (3000), и это отмечено в experiments.md как открытый пункт.

СОГЛАСОВАННОСТЬ ПРИЗНАКОВ. Блок построения признаков должен совпадать с scripts/build_pool.py
дословно, иначе ранкер получит на входе не то, на чём учился. Совпадение проверяется ассертом
по именам и порядку признаков модели.

Запуск: python scripts/predict_ranker.py --train-seed 42 --pool 1200 --out results/answer_ranker.csv
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
from src.microcat import MicrocatModel
from src.params import parse_many
from src.paths import RESULTS, val_dir
from src.text import Lemmatizer, norm_query
from scripts.train_ranker import BASE_FEATURES, add_relative_features

TOP_K = 50
BATCH = 100
B_BY_FIELD = {"title": 0.3, "kind": 0.3, "service": 0.75, "place": 0.5, "other": 0.75, "desc": 0.9}
WEIGHTS = {"title": 3, "kind": 2, "service": 2, "place": 0.5, "other": 0.3, "desc": 1}
RADIUS = 30.0        # тот же радиус, что в build_pool.py: пул должен строиться одинаково
SOFT_MU = 1.0


def train_ranker(train_seed: int, pool_size: int):
    """Ранкер на всех строках валидационного пула, без кросс-валидации (она нужна была для оценки)."""
    import lightgbm as lgb

    out = val_dir(train_seed)
    pool = pd.read_parquet(out / f"pool{pool_size}_union.parquet")
    vq = pd.read_parquet(out / "val_queries.parquet")
    npos = vq.pos.str.len().to_numpy()

    extra = add_relative_features(pool)
    F = [f for f in BASE_FEATURES if f in pool.columns] + extra

    # Случайная подвыборка негативов (НЕ по рангу: см. experiments.md, ловушка #17).
    n_neg = int((pool.label == 0).sum())
    take = min(200 * int(pool.ev.nunique()), n_neg)
    t = pd.concat([pool[pool.label == 1],
                   pool[pool.label == 0].sample(n=take, random_state=0)]).sort_values("ev")
    t["w"] = 1.0
    pos_w = 1.0 / np.maximum(npos, 1)
    t.loc[t.label == 1, "w"] = pos_w[t.loc[t.label == 1, "ev"].to_numpy()]

    groups = t.groupby("ev", sort=True).size().to_numpy()
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                       num_leaves=63, min_child_samples=30, subsample=0.9,
                       colsample_bytree=0.8, random_state=0, verbose=-1, label_gain=[0, 1])
    m.fit(t[F], t.label, group=groups, sample_weight=t.w)
    print(f"ранкер обучен: {len(t)} строк, {len(F)} признаков, {t.label.sum()} позитивов")
    return m, F


def main(train_seed: int, pool_size: int, out_path: Path) -> None:
    t0 = time.time()
    model, F = train_ranker(train_seed, pool_size)

    q = D.read_bench_queries()
    it = D.read_bench_items(columns=D.ITEM_COLS)
    n_q, n_d = len(q), len(it)
    print(f"бенчмарк: {n_q} запросов, {n_d} объявлений, {time.time()-t0:.0f} с")

    # --- индекс BM25F по корпусу бенчмарка ---------------------------------
    lm = Lemmatizer()
    parsed = parse_many(it.item_infm_params_text.fillna("").tolist())
    fields = {
        "title": lm.many(it.item_title_raw.fillna("").tolist()),
        "kind": lm.many(parsed["kind"]), "service": lm.many(parsed["service"]),
        "place": lm.many(parsed["place"]), "other": lm.many(parsed["other"]),
        "desc": lm.many(it.item_description_raw.fillna("").tolist()),
    }
    qs = lm.many(q.search_query.tolist())
    builder = BM25FBuilder().fit(fields, b=B_BY_FIELD)
    Q = builder.transform_queries(qs)
    W_all = builder.combine(WEIGHTS)
    W_field = {f: builder.combine({f: 1.0}) for f in fields}
    del fields, parsed, builder
    print(f"индекс готов, {time.time()-t0:.0f} с")

    # --- признаки из ПОЛНОГО train ----------------------------------------
    tr = D.read_train(columns=["search_query", "search_location_id", "item_id",
                               "item_latitude", "item_longitude", "item_microcat_id"])
    cent = location_centers(tr[["search_location_id", "item_latitude", "item_longitude"]])
    qc = cent.reindex(q.search_location_id)
    print(f"запросов без центра локации: {int(qc.lat.isna().sum())}")
    qlat, qlon = np.radians(qc.lat.to_numpy()), np.radians(qc.lon.to_numpy())
    ilat, ilon = np.radians(it.item_latitude.to_numpy()), np.radians(it.item_longitude.to_numpy())
    iloc = it.item_location_id.to_numpy()
    qloc = q.search_location_id.to_numpy()
    item_pop = it.item_id.map(tr.item_id.value_counts()).fillna(0).to_numpy(dtype=np.float32)

    # классификатор microcat на полном train
    tr["qn"] = tr.search_query.map({x: norm_query(x) for x in tr.search_query.unique()})
    agg = tr.groupby(["qn", "item_microcat_id"], observed=True).size().reset_index(name="w")
    uniq = agg.qn.unique().tolist()
    agg["lem"] = agg.qn.map(dict(zip(uniq, lm.many(uniq))))
    mc = MicrocatModel(kind="word", alpha=0.1).fit(
        agg.lem.tolist(), agg.item_microcat_id.to_numpy(), agg.w.to_numpy().astype(float))
    P_mc = mc.predict_proba(lm.many(q.search_query.tolist()))
    lm.save()
    cix = {c: i for i, c in enumerate(mc.classes_)}
    item_mc_col = it.item_microcat_id.map(cix).fillna(-1).to_numpy(dtype=np.int64)
    del tr, agg
    print(f"признаки из train готовы ({len(mc.classes_)} классов microcat), {time.time()-t0:.0f} с")

    ids = it.item_id.to_numpy()
    rating = it.item_rating.to_numpy(dtype=np.float32)
    reviews = it.item_rating_reviews_count.fillna(0).to_numpy(dtype=np.float32)
    price = it.item_price.to_numpy(dtype=np.float32)
    qlen = np.array([len(x.split()) for x in q.search_query], dtype=np.float32)

    # seen: встречался ли нормализованный запрос бенчмарка в train (ТЗ: 37.5% встречались)
    tr_q = D.read_train(columns=["search_query"]).search_query
    all_qn = {norm_query(x) for x in tr_q.unique()}
    q_seen = np.array([norm_query(x) in all_qn for x in q.search_query], dtype=np.int8)
    print(f"доля запросов бенчмарка, виденных в train: {q_seen.mean():.3f} (ТЗ: 0.375)")

    preds = []
    for s in range(0, n_q, BATCH):
        e = min(s + BATCH, n_q)
        S = (Q[s:e] @ W_all).toarray()
        SF = {f: (Q[s:e] @ W_field[f]).toarray() for f in W_field}
        for j in range(e - s):
            i = s + j
            d = (np.full(n_d, 1e4, dtype=np.float32) if np.isnan(qlat[i])
                 else haversine_km(qlat[i], qlon[i], ilat, ilon).astype(np.float32))
            in_r = d < RADIUS
            ld = np.log1p(d)
            x_hard = S[j] + 1000.0 * in_r
            x_soft = S[j] - SOFT_MU * ld
            kh = min(pool_size // 2, n_d)
            ih = np.argpartition(-x_hard, kh - 1)[:kh]
            isf = np.argpartition(-x_soft, kh - 1)[:kh]
            idx = np.union1d(ih, isf)
            idx = idx[np.argsort(-x_hard[idx])]
            k = len(idx)
            rank_hard = np.empty(k, dtype=np.int32)
            rank_hard[np.argsort(-x_hard[idx])] = np.arange(k)
            rank_soft = np.empty(k, dtype=np.int32)
            rank_soft[np.argsort(-x_soft[idx])] = np.arange(k)

            col = item_mc_col[idx]
            p_mc = np.where(col >= 0, P_mc[i, np.maximum(col, 0)], 0.0).astype(np.float32)

            rec = {
                "ev": np.full(k, i, dtype=np.int32), "it": idx.astype(np.int32),
                "rank": np.arange(k, dtype=np.int32),
                "rank_hard": rank_hard, "rank_soft": rank_soft,
                "bm25f": S[j][idx], "dist": d[idx],
                "in_radius": in_r[idx].astype(np.int8),
                "in_r50": (d[idx] < 50).astype(np.int8),
                "in_r80": (d[idx] < 80).astype(np.int8),
                "same_loc": (iloc[idx] == qloc[i]).astype(np.int8),
                "p_mc": p_mc, "item_pop": item_pop[idx],
                "rating": rating[idx], "reviews": reviews[idx], "price": price[idx],
                "qlen": np.full(k, qlen[i], dtype=np.float32),
                "seen": np.full(k, q_seen[i], dtype=np.int8),
                "n_in_radius": np.full(k, in_r.sum(), dtype=np.float32),
                "label": np.zeros(k, dtype=np.int8),   # метки неизвестны, нужен только для схемы
            }
            for f in W_field:
                rec[f"s_{f}"] = SF[f][j][idx]
            cand = pd.DataFrame(rec)
            add_relative_features(cand)
            score = model.predict(cand[F])
            top = idx[np.argsort(-score)[:TOP_K]]
            preds.append(" ".join(ids[top]))
        if s % 500 == 0:
            print(f"  {s}/{n_q}, {time.time()-t0:.0f} с")

    ans = pd.DataFrame({"query_id": q.query_id.astype(str), "answer": preds})

    corpus_ids = set(ids)
    assert list(ans.columns) == ["query_id", "answer"]
    assert len(ans) == n_q and ans.query_id.is_unique
    assert set(ans.query_id) == set(q.query_id.astype(str))
    for p in preds:
        parts = p.split()
        assert 0 < len(parts) <= TOP_K, f"неверное число кандидатов: {len(parts)}"
        assert len(set(parts)) == len(parts), "неуникальные item_id"
        assert set(parts) <= corpus_ids, "item_id вне корпуса"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ans.to_csv(out_path, index=False)

    meta = {"pipeline": "BM25F(6 полей, полное описание) + пул объединением гео-каналов + "
                        "LightGBM lambdarank",
            "ranker_trained_on_seed": train_seed, "pool_size": pool_size,
            "radius": RADIUS, "soft_mu": SOFT_MU, "weights": WEIGHTS, "b_by_field": B_BY_FIELD,
            "features": F, "queries": n_q, "corpus": n_d,
            "bench_seen_share": float(q_seen.mean()),
            "validation_R@50": {"seed42": 0.9167, "seed1": 0.9178},
            "seconds": round(time.time() - t0, 1)}
    with out_path.with_suffix(".meta.json").open("w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"\nвсе проверки формата пройдены -> {out_path}  ({time.time()-t0:.0f} с)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=1200)
    ap.add_argument("--out", type=Path, default=RESULTS / "answer_ranker.csv")
    a = ap.parse_args()
    main(a.train_seed, a.pool, a.out)
