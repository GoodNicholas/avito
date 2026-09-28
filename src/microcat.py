"""
Классификатор P(microcat | запрос): вероятность, что запрос относится к данной микрокатегории.

ЗАЧЕМ. 86.3% выборов приходятся на доминирующую microcat запроса (ТЗ §1), а внутри-событийная
AUC этого признака 0.928 — сильнее BM25 (0.904). Причина понятна: запрос «ремонт стиральных
машин» и объявление «Мастер по стиралкам» могут не иметь общих лемм, но лежат в одной
микрокатегории. Поэтому microcat — и признак для ранкера, и отдельный канал генерации
кандидатов (взять объявления топовых microcat в гео-окрестности, даже без текстового совпадения).

ЧЕМ. MultinomialNB по леммам запроса. Выбор осознанный, а не от лени:
  - обучающих пар сотни тысяч, классов — тысячи, признаки разреженные -> NB обучается секунды
    и не переобучается на редких классах;
  - нужна КАЛИБРОВАННАЯ вероятность по всем классам сразу (для ранкера нужен P у конкретного
    кандидата), а не только top-1;
  - ТЗ фиксирует ориентир: top-1 0.70, top-5 0.91 — есть с чем сверяться.
Дополнительно — вариант на символьных n-граммах (3-5): он устойчив к опечаткам и к составным
словам, которых в запросах много («стиралка», «посудомойка»).

УТЕЧКА. Обучение только по строкам с fit_mask=True. Агрегируем до уникальных пар
(запрос, microcat) с весом = число выборов: иначе частотные запросы («маникюр») задавят хвост,
а нам нужна вероятность для ЛЮБОГО запроса, включая редкий.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class MicrocatModel:
    """Обёртка: fit по (леммы запроса, microcat, вес) -> predict_proba по леммам запроса."""

    def __init__(self, kind: str = "word", alpha: float = 0.1):
        """
        kind='word'  — леммы, CountVectorizer (быстро, ориентир ТЗ);
        kind='char'  — символьные 3-5-граммы, tf-idf (устойчив к опечаткам);
        kind='both'  — конкатенация обоих представлений.
        """
        self.kind = kind
        self.alpha = alpha
        self._vecs: list = []
        self.clf: MultinomialNB | None = None
        self.classes_: np.ndarray | None = None

    def _make_vecs(self):
        vecs = []
        if self.kind in ("word", "both"):
            vecs.append(CountVectorizer(token_pattern=r"\S+", dtype=np.float32))
        if self.kind in ("char", "both"):
            vecs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                        min_df=2, sublinear_tf=True, dtype=np.float32))
        return vecs

    def _transform(self, texts: list[str], fit: bool) -> sp.csr_matrix:
        mats = []
        for v in self._vecs:
            mats.append(v.fit_transform(texts) if fit else v.transform(texts))
        return mats[0] if len(mats) == 1 else sp.hstack(mats).tocsr()

    def fit(self, query_lemmas: list[str], microcat: np.ndarray,
            weight: np.ndarray | None = None) -> "MicrocatModel":
        self._vecs = self._make_vecs()
        X = self._transform(query_lemmas, fit=True)
        self.clf = MultinomialNB(alpha=self.alpha)
        self.clf.fit(X, microcat, sample_weight=weight)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, query_lemmas: list[str]) -> np.ndarray:
        """Матрица (n_queries x n_classes). Порядок классов — self.classes_."""
        return self.clf.predict_proba(self._transform(query_lemmas, fit=False))

    def topk_accuracy(self, proba: np.ndarray, true_sets: list[set], ks=(1, 3, 5, 10, 20)) -> dict:
        """
        Доля событий, где ХОТЬ ОДНА истинная microcat попала в top-k предсказаний.
        У события может быть несколько позитивов в разных microcat — отсюда множества.
        """
        order = np.argsort(-proba, axis=1)
        res = {}
        for k in ks:
            top = [set(self.classes_[o[:k]]) for o in order]
            res[f"top{k}"] = float(np.mean([len(t & s) > 0 for t, s in zip(top, true_sets)]))
        return res

    def proba_of(self, proba: np.ndarray, microcat_of_item: np.ndarray) -> np.ndarray:
        """
        Вектор P(microcat объявления | запрос) для набора (событие, объявление).
        microcat_of_item — microcat каждого кандидата; используется в признаках ранкера.
        Возвращает столбец индексов классов; microcat, не виденная при обучении, даёт 0.
        """
        cix = {c: i for i, c in enumerate(self.classes_)}
        return np.array([cix.get(m, -1) for m in microcat_of_item])
