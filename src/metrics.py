"""
Recall@K — единственная метрика задачи.

Recall@K события = |topK ∩ позитивы| / |позитивы|; итог — СРЕДНЕЕ ПО СОБЫТИЯМ (не по парам).
Это важно: событие с 3 позитивами весит столько же, сколько событие с 1, поэтому при
обучении ранкера позитивы редких «многопозитивных» событий нельзя брать с весом 1
(требование ТЗ §3, фаза 3).
"""
from __future__ import annotations

import numpy as np


def recall_at_k(top_ids: np.ndarray, positives: set, k: int) -> float:
    """Recall@K одного события. top_ids отсортирован по убыванию скора."""
    if not positives:
        return np.nan
    return len(positives & set(top_ids[:k])) / len(positives)


def topk_from_scores(scores: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    """
    top-K id по убыванию скора. argpartition (O(n)) вместо полной сортировки (O(n log n)):
    на 192k документов × 3000 событий это разница в разы.
    """
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    return ids[idx[np.argsort(-scores[idx])]]


def summarize(hits: dict[int, list[float]], seen_mask: np.ndarray) -> dict:
    """
    Сводка по набору событий: R@K для каждого K плюс разбивка R@50 на seen/unseen.
    seen = нормализованный запрос встречался в обучающей части (37.5% бенчмарка).
    """
    out = {}
    for k, vals in hits.items():
        v = np.asarray(vals, dtype=float)
        out[f"R@{k}"] = float(np.nanmean(v))
        if k == 50:
            out["R@50_seen"] = float(np.nanmean(v[seen_mask]))
            out["R@50_unseen"] = float(np.nanmean(v[~seen_mask]))
    return out
