"""
Обучение и проверка классификатора P(microcat | запрос) на обучающей части (fit_mask).

Сверяемся с ориентиром ТЗ §3.3: MultinomialNB по леммам даёт top-1 0.70, top-5 0.91.
Если сильно расходится — ошибка в маске или в агрегации, а не «другая модель».

Сохраняет матрицу вероятностей для валидационных запросов в cache/val_seed<N>/mc_proba.npz,
чтобы её переиспользовали и канал генерации, и признаки ранкера.

Запуск: python scripts/train_microcat.py --seed 42
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
from src.microcat import MicrocatModel
from src.paths import RESULTS, val_dir
from src.text import Lemmatizer, norm_query



def main(seed: int) -> None:
    t0 = time.time()
    out = val_dir(seed)
    vq = pd.read_parquet(out / "val_queries.parquet")
    fit_mask = np.load(out / "fit_mask.npy")

    # --- обучающие пары: (нормализованный запрос, microcat) с весом = число выборов -----
    f = D.read_train(columns=["search_query", "item_microcat_id"])[fit_mask]
    f["qn"] = f.search_query.map({q: norm_query(q) for q in f.search_query.unique()})
    agg = f.groupby(["qn", "item_microcat_id"], observed=True).size().reset_index(name="w")
    print(f"обучающих пар (запрос, microcat): {len(agg)}, уникальных запросов "
          f"{agg.qn.nunique()}, классов {agg.item_microcat_id.nunique()}, {time.time()-t0:.0f} с")

    lm = Lemmatizer()
    uniq = agg.qn.unique().tolist()
    lem_map = dict(zip(uniq, lm.many(uniq)))
    agg["lem"] = agg.qn.map(lem_map)
    vq_lem = lm.many(vq.search_query.tolist())
    lm.save()

    # --- истинные microcat валидационных событий (по позитивам) -----------------------
    vc = pd.read_parquet(out / "val_corpus.parquet", columns=["item_id", "item_microcat_id"])
    mc_of = dict(zip(vc.item_id, vc.item_microcat_id))
    true_sets = [{mc_of[x] for x in p if x in mc_of} for p in vq.pos]
    seen = vq.seen.to_numpy()

    rows = []
    best = None
    for kind, alpha in [("word", 0.1), ("word", 0.01), ("char", 0.01), ("both", 0.01)]:
        m = MicrocatModel(kind=kind, alpha=alpha).fit(
            agg.lem.tolist(), agg.item_microcat_id.to_numpy(), agg.w.to_numpy().astype(float))
        P = m.predict_proba(vq_lem)
        acc = m.topk_accuracy(P, true_sets)
        top5 = np.argsort(-P, axis=1)[:, :5]
        hit5 = np.array([len({m.classes_[j] for j in row} & s) > 0
                         for row, s in zip(top5, true_sets)])
        r = {"model": f"{kind}_a{alpha}", **acc,
             "top5_seen": float(hit5[seen].mean()), "top5_unseen": float(hit5[~seen].mean())}
        rows.append(r)
        print("  " + json.dumps(r, ensure_ascii=False))
        if best is None or acc["top5"] > best[1]:
            best = (m, acc["top5"], P, f"{kind}_a{alpha}")

    m, _, P, name = best
    np.savez_compressed(out / "mc_proba.npz", proba=P.astype(np.float32), classes=m.classes_)
    res = pd.DataFrame(rows).set_index("model")
    exp = RESULTS / f"microcat_seed{seed}"
    exp.mkdir(parents=True, exist_ok=True)
    res.to_csv(exp / "summary.csv")
    print(f"\n{res.round(4).to_string()}\n\nлучшая модель: {name}; "
          f"сохранено в {out/'mc_proba.npz'}; {time.time()-t0:.0f} с")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.seed)
