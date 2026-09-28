"""
Сборка пула кандидатов top-N на каждое валидационное событие + признаки для ранкера.

ЗАЧЕМ. По таксономии промахов (scripts/analyze_data.py) самая большая дыра — 13.0% позитивов
лежат В ПУЛЕ, но ниже
50-го места (медианный ранг 128). Их вытаскивает не новый канал, а переупорядочивание, то есть
ранкер. Здесь готовится его обучающая матрица.

ПРИЗНАКИ. Кроме суммарного BM25F считаем скор КАЖДОГО поля отдельно: ранкеру важно различать
«совпало с заголовком» и «совпало где-то в описании» — это разная надёжность. Плюс гео,
P(microcat|запрос), популярность объявления в обучающей части и статистики события.

Популярность считается ТОЛЬКО по fit_mask: иначе объявление-позитив валидационного события
получит +1 к популярности от самого себя (утечка, из-за которой «история» кажется сильной).

Запуск: python scripts/build_pool.py --seed 42 --pool 1000 --desc-chars 1600
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
from src.params import parse_many
from src.paths import val_dir
from src.text import Lemmatizer

B_BY_FIELD = {"title": 0.3, "kind": 0.3, "service": 0.75, "place": 0.5, "other": 0.75, "desc": 0.9}
WEIGHTS = {"title": 3, "kind": 2, "service": 2, "place": 0.5, "other": 0.3, "desc": 1}
RADIUS = 30.0
SOFT_MU = 1.0        # score - mu*log(1+dist): мягкое гео даёт лучший ПОТОЛОК пула (R@1000 0.957)
BATCH = 100          # меньше батч: держим 7 матриц скоров одновременно (суммарная + 6 полей)



def main(seed: int, pool_size: int, desc_chars: int, geo_mode: str) -> None:
    t0 = time.time()
    out = val_dir(seed)
    vq = pd.read_parquet(out / "val_queries.parquet")
    vc = pd.read_parquet(out / "val_corpus.parquet")
    fit_mask = np.load(out / "fit_mask.npy")
    n_q, n_d = len(vq), len(vc)

    # --- тексты и индекс ---------------------------------------------------
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
    Q = builder.transform_queries(qs)
    W_all = builder.combine(WEIGHTS)
    # Отдельные матрицы полей: скор каждого поля — самостоятельный признак ранкера.
    W_field = {f: builder.combine({f: 1.0}) for f in fields}
    del fields, parsed, builder
    print(f"индекс и матрицы полей готовы, {time.time()-t0:.0f} с")

    # --- гео ---------------------------------------------------------------
    g = D.read_train(columns=["search_location_id", "item_latitude", "item_longitude"])[fit_mask]
    qc = location_centers(g).reindex(vq.search_location_id)
    qlat, qlon = np.radians(qc.lat.to_numpy()), np.radians(qc.lon.to_numpy())
    ilat, ilon = np.radians(vc.item_latitude.to_numpy()), np.radians(vc.item_longitude.to_numpy())
    iloc = vc.item_location_id.to_numpy()
    qloc = vq.search_location_id.to_numpy()

    # --- популярность объявления в обучающей части (сглаженная) -----------
    pop = (D.read_train(columns=["item_id"])[fit_mask].item_id.value_counts())
    item_pop = vc.item_id.map(pop).fillna(0).to_numpy(dtype=np.float32)

    # --- P(microcat | запрос) ---------------------------------------------
    mc_path = out / "mc_proba.npz"
    if mc_path.exists():
        z = np.load(mc_path, allow_pickle=True)
        P_mc, mc_classes = z["proba"], z["classes"]
        cix = {c: i for i, c in enumerate(mc_classes)}
        item_mc_col = vc.item_microcat_id.map(cix).fillna(-1).to_numpy(dtype=np.int64)
        print("P(microcat|запрос) подключён")
    else:
        P_mc, item_mc_col = None, None
        print("ВНИМАНИЕ: mc_proba.npz нет — признак p_mc будет нулевым. "
              "Сначала запусти scripts/train_microcat.py")

    ids = vc.item_id.to_numpy()
    id2ix = {x: i for i, x in enumerate(ids)}
    rating = vc.item_rating.to_numpy(dtype=np.float32)
    reviews = vc.item_rating_reviews_count.fillna(0).to_numpy(dtype=np.float32)
    price = vc.item_price.to_numpy(dtype=np.float32)
    item_mc = vc.item_microcat_id.to_numpy()

    qlen = np.array([len(q.split()) for q in vq.search_query], dtype=np.float32)
    seen_flag = vq.seen.to_numpy()
    pos_ix = [np.array([id2ix[p] for p in plist if p in id2ix], dtype=np.int64) for plist in vq.pos]

    chunks, pos_info = [], []
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
            x_hard = S[j] + 1000.0 * in_r          # канал 1: сначала радиус, внутри по BM25F
            x_soft = S[j] - SOFT_MU * ld           # канал 2: мягкий штраф за расстояние

            if geo_mode == "hard":
                k = min(pool_size, n_d)
                idx = np.argpartition(-x_hard, k - 1)[:k]
                idx = idx[np.argsort(-x_hard[idx])]
            elif geo_mode == "soft":
                k = min(pool_size, n_d)
                idx = np.argpartition(-x_soft, k - 1)[:k]
                idx = idx[np.argsort(-x_soft[idx])]
            else:                                   # union: половина пула из каждого канала
                kh = min(pool_size // 2, n_d)
                ih = np.argpartition(-x_hard, kh - 1)[:kh]
                isf = np.argpartition(-x_soft, kh - 1)[:kh]
                idx = np.union1d(ih, isf)
                # Порядок в пуле — по жёсткому каналу, он лучше для прямого top-50.
                idx = idx[np.argsort(-x_hard[idx])]
            k = len(idx)

            # Ранги внутри пула по каждому каналу: относительный порядок — то, что нужно ранкеру.
            rank_hard = np.empty(k, dtype=np.int32)
            rank_hard[np.argsort(-x_hard[idx])] = np.arange(k)
            rank_soft = np.empty(k, dtype=np.int32)
            rank_soft[np.argsort(-x_soft[idx])] = np.arange(k)

            if P_mc is not None:
                col = item_mc_col[idx]
                p_mc = np.where(col >= 0, P_mc[i, np.maximum(col, 0)], 0.0).astype(np.float32)
            else:
                p_mc = np.zeros(k, dtype=np.float32)

            rec = {
                "ev": np.full(k, i, dtype=np.int32),
                "it": idx.astype(np.int32),
                "rank": np.arange(k, dtype=np.int32),
                "rank_hard": rank_hard,
                "rank_soft": rank_soft,
                "bm25f": S[j][idx],
                "dist": d[idx],
                "in_radius": in_r[idx].astype(np.int8),
                "in_r50": (d[idx] < 50).astype(np.int8),
                "in_r80": (d[idx] < 80).astype(np.int8),
                "same_loc": (iloc[idx] == qloc[i]).astype(np.int8),
                "p_mc": p_mc,
                "item_pop": item_pop[idx],
                "rating": rating[idx],
                "reviews": reviews[idx],
                "price": price[idx],
                "qlen": np.full(k, qlen[i], dtype=np.float32),
                "seen": np.full(k, seen_flag[i], dtype=np.int8),
                "n_in_radius": np.full(k, in_r.sum(), dtype=np.float32),
                "label": np.isin(idx, pos_ix[i]).astype(np.int8),
            }
            for f in W_field:
                rec[f"s_{f}"] = SF[f][j][idx]
            chunks.append(pd.DataFrame(rec))

            # Где оказался КАЖДЫЙ позитив — даже если он не попал в пул. Нужно для бюджета потерь.
            for p in pos_ix[i]:
                # rank — позиция позитива в ПОЛНОМ корпусе по жёсткому каналу (не в пуле):
                # так видно, насколько далеко он был, даже если в пул не попал.
                pos_info.append((i, int(p), float(S[j][p]), float(d[p]),
                                 int((x_hard > x_hard[p]).sum()), int(S[j][p] > 0)))
        if s % 500 == 0:
            print(f"  {s}/{n_q} событий, {time.time()-t0:.0f} с")

    pool = pd.concat(chunks, ignore_index=True)
    tag = f"pool{pool_size}_{geo_mode}"
    pool.to_parquet(out / f"{tag}.parquet", index=False)
    pi = pd.DataFrame(pos_info, columns=["ev", "it", "bm25f", "dist", "rank", "has_overlap"])
    pi.to_parquet(out / f"pos_info_{tag}.parquet", index=False)

    npos = vq.pos.str.len().to_numpy()
    in_pool = pool.groupby("ev").label.sum().reindex(range(n_q), fill_value=0).to_numpy()
    ceiling = float((in_pool / npos).mean())
    stats = {
        "seed": seed, "pool_size": pool_size, "desc_chars": desc_chars,
        "radius": RADIUS, "geo_mode": geo_mode, "soft_mu": SOFT_MU,
        "pool_rows_per_event": float(len(pool) / n_q),
        "pool_rows": int(len(pool)), "ceiling_R@pool": ceiling,
        "positives_total": int(npos.sum()),
        "positives_in_pool": int(in_pool.sum()),
        "median_rank_of_missed": float(pi[pi["rank"] >= 50]["rank"].median()),
        "missed_no_overlap_share": float((pi[pi["rank"] >= 50].has_overlap == 0).mean()),
        "seconds": round(time.time() - t0, 1),
    }
    with (out / f"{tag}_stats.json").open("w") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--desc-chars", type=int, default=0)
    ap.add_argument("--geo-mode", choices=["hard", "soft", "union"], default="union",
                    help="hard=приоритет радиуса, soft=мягкий штраф за расстояние, "
                         "union=объединение обоих каналов")
    a = ap.parse_args()
    main(a.seed, a.pool, a.desc_chars, a.geo_mode)
