"""
Ранкер поверх пула кандидатов.

ЗАЧЕМ. 12.9% позитивов попадают в пул, но стоят ниже 50-го места — это крупнейшая статья
бюджета потерь. Порядок внутри пула задаётся сейчас грубым правилом «в радиусе -> по BM25F».
Ранкер учится взвешивать сигналы: текст по каждому полю, расстояние, P(microcat), популярность.

ОЦЕНКА БЕЗ ОБМАНА. Ранкер обучается и проверяется кросс-валидацией ПО СОБЫТИЯМ (фолд = ev % K),
поэтому событие никогда не оценивается моделью, видевшей его. Это честная оценка прироста
ранкера, но НЕ финальный пайплайн: финальная модель обучается на событиях train с out-of-fold
признаками, иначе «популярность» и «история» переоценятся. Здесь мы решаем,
стоит ли вообще эта конструкция усилий.

ПРИЗНАКИ ВНУТРИ СОБЫТИЯ. Абсолютный BM25F бесполезен сам по себе: у длинного запроса скоры
выше. Ранкеру нужны ОТНОСИТЕЛЬНЫЕ величины — score/max по событию, ранг, доля от суммы.

ВЕСА. Метрика усредняется по событиям, а не по парам, поэтому событие с 3 позитивами не должно
тянуть обучение сильнее события с 1: вес позитива = 1/число позитивов события.

Запуск: python scripts/train_ranker.py --seed 42 --pool 1000 --folds 3
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

from src.paths import RESULTS, val_dir

BASE_FEATURES = [
    "bm25f", "rank", "rank_hard", "rank_soft", "dist", "in_radius", "in_r50", "in_r80",
    "same_loc", "p_mc", "item_pop",
    "rating", "reviews", "price", "qlen", "seen", "n_in_radius",
    "s_title", "s_kind", "s_service", "s_place", "s_other", "s_desc",
]


def add_relative_features(pool: pd.DataFrame) -> list[str]:
    """Относительные признаки внутри события: без них абсолютные скоры плохо сравнимы между запросами."""
    g = pool.groupby("ev", sort=False)
    pool["log_dist"] = np.log1p(pool.dist)
    pool["log_pop"] = np.log1p(pool.item_pop)
    pool["log_reviews"] = np.log1p(pool.reviews)
    pool["bm25f_rel"] = pool.bm25f / g.bm25f.transform("max").clip(lower=1e-6)
    pool["p_mc_rel"] = pool.p_mc / g.p_mc.transform("max").clip(lower=1e-9)
    pool["p_mc_rank"] = g.p_mc.rank(ascending=False, method="min")
    pool["title_rel"] = pool.s_title / g.s_title.transform("max").clip(lower=1e-6)
    pool["desc_rel"] = pool.s_desc / g.s_desc.transform("max").clip(lower=1e-6)
    pool["dist_rel"] = pool.dist / g.dist.transform("median").clip(lower=1e-3)
    extra = ["log_dist", "log_pop", "log_reviews", "bm25f_rel", "p_mc_rel",
             "p_mc_rank", "title_rel", "desc_rel", "dist_rel"]
    return extra


def recall_at_k(pool: pd.DataFrame, score_col: str, npos: np.ndarray, n_ev: int, k: int = 50) -> np.ndarray:
    """Recall@k по событиям при заданном столбце скоров. npos — ИСТИННОЕ число позитивов события
    (включая те, что не попали в пул), иначе recall будет завышен."""
    t = (pool.sort_values(["ev", score_col], ascending=[True, False])
             .groupby("ev", sort=True).head(k))
    hit = t.groupby("ev").label.sum().reindex(range(n_ev), fill_value=0).to_numpy()
    return hit / npos



def main(seed: int, pool_size: int, folds: int, neg_per_event: int, geo_mode: str,
         drop: set[str] | None = None, tag: str = "", corpus: str = "bench_plus_pos") -> None:
    t0 = time.time()
    out = val_dir(seed)
    vq = pd.read_parquet(out / "val_queries.parquet")
    pool = pd.read_parquet(out / f"pool{pool_size}_{geo_mode}.parquet")
    n_ev = len(vq)
    npos = vq.pos.str.len().to_numpy()
    seen = vq.seen.to_numpy()

    extra = add_relative_features(pool)
    F = [f for f in BASE_FEATURES if f in pool.columns] + extra
    print(f"пул {len(pool)} строк, {len(F)} признаков, {time.time()-t0:.0f} с")

    # --- точка отсчёта: текущий порядок пула (в радиусе -> по BM25F) --------
    pool["base_score"] = -pool["rank"].astype(np.float32)
    r_base = recall_at_k(pool, "base_score", npos, n_ev)
    print(f"порядок пула без ранкера:  R@50 = {r_base.mean():.4f} "
          f"(seen {r_base[seen].mean():.4f} / unseen {r_base[~seen].mean():.4f})")
    ceiling = (pool.groupby("ev").label.sum().reindex(range(n_ev), fill_value=0).to_numpy() / npos)
    print(f"потолок пула R@{pool_size} = {ceiling.mean():.4f}")

    # --- веса: событие с несколькими позитивами не должно доминировать -----
    pool["w"] = 1.0
    pos_w = 1.0 / np.maximum(npos, 1)
    pool.loc[pool.label == 1, "w"] = pos_w[pool.loc[pool.label == 1, "ev"].to_numpy()]

    fold = pool.ev.to_numpy() % folds
    rows = []

    import lightgbm as lgb

    for model_name in ("lambdarank", "binary"):
        oof = np.zeros(len(pool), dtype=np.float32)
        for f in range(folds):
            tr_mask = fold != f
            tr = pool[tr_mask]
            if neg_per_event <= 0:
                t = tr.sort_values("ev")          # все строки пула, без подвыборки
            else:
                # ВАЖНО: негативы прореживаем СЛУЧАЙНО по всему пулу, а НЕ «берём топ-N».
                # Если негативы взять только из топа пула, а позитивы оставить все, возникает
                # ложная связь: позитив на ранге 700 в обучении есть, а негатива на ранге 700
                # нет — и модель выучивает «большой ранг = позитив», то есть ОБРАТНЫЙ порядок.
                # Измерено: такая подвыборка дала R@50 = 0.037 против 0.830 у базового порядка
                # пула. Случайная подвыборка сохраняет распределение рангов у негативов.
                n_neg = int((tr.label == 0).sum())
                take = min(neg_per_event * int(pool.ev.nunique()), n_neg)
                neg = tr[tr.label == 0].sample(n=take, random_state=0)
                t = pd.concat([tr[tr.label == 1], neg]).sort_values("ev")
            if model_name == "lambdarank":
                groups = t.groupby("ev", sort=True).size().to_numpy()
                m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                                   num_leaves=63, min_child_samples=30, subsample=0.9,
                                   colsample_bytree=0.8, random_state=0, verbose=-1,
                                   label_gain=[0, 1])
                m.fit(t[F], t.label, group=groups, sample_weight=t.w)
            else:
                m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                       min_child_samples=30, subsample=0.9, colsample_bytree=0.8,
                                       random_state=0, verbose=-1)
                m.fit(t[F], t.label, sample_weight=t.w)
            te = pool[~tr_mask]
            pred = (m.predict(te[F]) if model_name == "lambdarank"
                    else m.predict_proba(te[F])[:, 1])
            oof[~tr_mask] = pred
            print(f"  {model_name} фолд {f}: обучено на {len(t)} строках, {time.time()-t0:.0f} с")
        pool[f"score_{model_name}"] = oof
        r = recall_at_k(pool, f"score_{model_name}", npos, n_ev)
        rows.append({"model": model_name, "R@50": float(r.mean()),
                     "R@50_seen": float(r[seen].mean()), "R@50_unseen": float(r[~seen].mean()),
                     "R@10": float(recall_at_k(pool, f"score_{model_name}", npos, n_ev, 10).mean()),
                     "R@200": float(recall_at_k(pool, f"score_{model_name}", npos, n_ev, 200).mean())})
        print(f"  -> {model_name}: R@50 = {r.mean():.4f} "
              f"(seen {r[seen].mean():.4f} / unseen {r[~seen].mean():.4f})")
        if model_name == "lambdarank":
            imp = pd.Series(m.feature_importances_, index=F).sort_values(ascending=False)
            print("\nважность признаков (последний фолд, top-15):")
            print(imp.head(15).to_string())

    rows.append({"model": "pool_order_baseline", "R@50": float(r_base.mean()),
                 "R@50_seen": float(r_base[seen].mean()), "R@50_unseen": float(r_base[~seen].mean()),
                 "R@10": float(recall_at_k(pool, "base_score", npos, n_ev, 10).mean()),
                 "R@200": float(recall_at_k(pool, "base_score", npos, n_ev, 200).mean())})
    rows.append({"model": f"pool_ceiling_R@{pool_size}", "R@50": float(ceiling.mean()),
                 "R@50_seen": float(ceiling[seen].mean()),
                 "R@50_unseen": float(ceiling[~seen].mean()), "R@10": np.nan, "R@200": np.nan})

    res = pd.DataFrame(rows).set_index("model")
    exp = RESULTS / f"ranker_seed{seed}_p{pool_size}_{geo_mode}"
    exp.mkdir(parents=True, exist_ok=True)
    res.to_csv(exp / "summary.csv")
    with (exp / "meta.json").open("w") as fh:
        json.dump({"seed": seed, "pool_size": pool_size, "geo_mode": geo_mode, "folds": folds,
                   "neg_per_event": neg_per_event, "features": F,
                   "seconds": round(time.time() - t0, 1)}, fh, ensure_ascii=False, indent=2)
    print(f"\n{res.round(4).to_string()}\n\nготово за {time.time()-t0:.0f} с -> {exp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--neg-per-event", type=int, default=0,
                    help="0 = все строки пула (безопасно); >0 = случайная подвыборка "
                         "негативов, в среднем N на событие")
    ap.add_argument("--geo-mode", choices=["hard", "soft", "union"], default="union")
    a = ap.parse_args()
    main(a.seed, a.pool, a.folds, a.neg_per_event, a.geo_mode)
