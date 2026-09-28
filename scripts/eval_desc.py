"""
Эксперимент: описание объявления как поле BM25F + изоляция вклада чистки параметров.

ПОВОД. В фазе 1 ТЗ описание упомянуто мельком («описание — первые N символов»), а baseline
из §2 его вообще не использует. Первый прогон BM25F показал, что именно описание даёт
основной прирост (+0.075 R@50), тогда как чистка шаблонов — минус. Значит надо разобраться:
  1) какая длина описания оптимальна (обрезка защищает от «воды» и контактов в конце текста,
     но режет полезное);
  2) какой вес описания в BM25F;
  3) действительно ли чистка параметров вредит, или она вредит только БЕЗ описания.

Вариант params=raw держит параметры одним сырым полем (как baseline), params=clean — разбор
на kind/service/place/other. Всё остальное идентично, поэтому разница = вклад чистки.

Запуск: python scripts/eval_desc.py --seed 42 --radius 30 --desc-chars 400
        (--desc-chars 0 = полное описание)
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
BATCH = 250
B_BY_FIELD = {"title": 0.3, "kind": 0.3, "service": 0.75, "place": 0.5,
              "other": 0.75, "desc": 0.9, "params_raw": 0.75}
DESC_WEIGHTS = (0.5, 1.0, 2.0, 3.0)


def main(seed: int, radius: float, desc_chars: int) -> None:
    t0 = time.time()
    out = val_dir(seed)
    vq = pd.read_parquet(out / "val_queries.parquet")
    vc = pd.read_parquet(out / "val_corpus.parquet")
    fit_mask = np.load(out / "fit_mask.npy")
    n_q = len(vq)

    lm = Lemmatizer()
    title = lm.many(vc.item_title_raw.fillna("").tolist())
    desc_raw = vc.item_description_raw.fillna("")
    desc = lm.many((desc_raw if desc_chars == 0 else desc_raw.str[:desc_chars]).tolist())
    params_raw = lm.many(vc.item_infm_params_text.fillna("").tolist())
    parsed = parse_many(vc.item_infm_params_text.fillna("").tolist())
    clean = {f: lm.many(parsed[f]) for f in ("kind", "service", "place", "other")}
    qs = lm.many(vq.search_query.tolist())
    lm.save()
    print(f"тексты готовы, {time.time()-t0:.0f} с")

    # --- гео-маска один раз ------------------------------------------------
    g = D.read_train(columns=["search_location_id", "item_latitude", "item_longitude"])[fit_mask]
    qc = location_centers(g).reindex(vq.search_location_id)
    qlat, qlon = np.radians(qc.lat.to_numpy()), np.radians(qc.lon.to_numpy())
    ilat, ilon = np.radians(vc.item_latitude.to_numpy()), np.radians(vc.item_longitude.to_numpy())
    in_radius = np.zeros((n_q, len(vc)), dtype=bool)
    for i in range(n_q):
        if not np.isnan(qlat[i]):
            in_radius[i] = haversine_km(qlat[i], qlon[i], ilat, ilon) < radius

    ids = vc.item_id.to_numpy()
    positives = [set(p) for p in vq.pos]
    seen = vq.seen.to_numpy()
    rows = []

    def evaluate(W, Q) -> dict[int, np.ndarray]:
        hits = {k: np.zeros(n_q) for k in KS}
        for s in range(0, n_q, BATCH):
            e = min(s + BATCH, n_q)
            S = (Q[s:e] @ W).toarray()
            S += 1000.0 * in_radius[s:e]
            for j in range(e - s):
                i = s + j
                top = topk_from_scores(S[j], ids, max(KS))
                for k in KS:
                    hits[k][i] = len(positives[i] & set(top[:k])) / len(positives[i])
        return hits

    def record(name: str, hits) -> None:
        r = {"config": name, "desc_chars": desc_chars}
        for k in KS:
            r[f"R@{k}"] = float(hits[k].mean())
        r["R@50_seen"] = float(hits[50][seen].mean())
        r["R@50_unseen"] = float(hits[50][~seen].mean())
        rows.append(r)
        print(f"  {name:26s} R@50={r['R@50']:.4f}  R@200={r['R@200']:.4f}  "
              f"R@1000={r['R@1000']:.4f}  (seen {r['R@50_seen']:.4f} / unseen {r['R@50_unseen']:.4f})")

    # --- А. сырые параметры + описание ------------------------------------
    print(f"\n--- params=raw, описание {desc_chars or 'полное'} симв ---")
    br = BM25FBuilder().fit({"title": title, "params_raw": params_raw, "desc": desc}, b=B_BY_FIELD)
    Qr = br.transform_queries(qs)
    record("raw_nodesc", evaluate(br.combine({"title": 3, "params_raw": 1}), Qr))
    for dw in DESC_WEIGHTS:
        record(f"raw_desc{dw:g}", evaluate(br.combine({"title": 3, "params_raw": 1, "desc": dw}), Qr))
    del br, Qr

    # --- Б. очищенные поля + описание -------------------------------------
    print(f"\n--- params=clean, описание {desc_chars or 'полное'} симв ---")
    bc = BM25FBuilder().fit({"title": title, **clean, "desc": desc}, b=B_BY_FIELD)
    Qc = bc.transform_queries(qs)
    base = {"title": 3, "kind": 2, "service": 2, "place": 0.5, "other": 0.3}
    record("clean_nodesc", evaluate(bc.combine(base), Qc))
    for dw in DESC_WEIGHTS:
        record(f"clean_desc{dw:g}", evaluate(bc.combine({**base, "desc": dw}), Qc))
    # только заголовок + описание: нужны ли параметры вообще при наличии описания
    record("title_desc_only", evaluate(bc.combine({"title": 3, "desc": 1}), Qc))

    res = pd.DataFrame(rows).set_index("config")
    exp = RESULTS / f"desc_seed{seed}_r{int(radius)}_d{desc_chars}"
    exp.mkdir(parents=True, exist_ok=True)
    res.to_csv(exp / "summary.csv")
    with (exp / "meta.json").open("w") as fh:
        json.dump({"seed": seed, "radius": radius, "desc_chars": desc_chars,
                   "b_by_field": B_BY_FIELD, "seconds": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"\n{res.round(4).to_string()}\n\nготово за {time.time()-t0:.0f} с -> {exp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--desc-chars", type=int, default=400)
    a = ap.parse_args()
    main(a.seed, a.radius, a.desc_chars)
