"""
BM25 и BM25F над разреженными матрицами.

BM25 (одно поле): вклад термина t в документ d
    idf(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))
скор документа = сумма вкладов терминов запроса (запрос — МНОЖЕСТВО лемм, без повторов).

BM25F (несколько полей, Robertson et al. 2004) — это НЕ сумма BM25 по полям. Частоты
сначала складываются с весами полей и нормировкой длины каждого поля:
    tf~(t,d) = sum_f  w_f * tf(t,d,f) / (1 - b_f + b_f*|d_f|/avgdl_f)
и только потом применяется насыщение:
    score = sum_t idf(t) * tf~(t,d)*(k1+1) / (k1 + tf~(t,d))
Разница принципиальна: при суммировании готовых BM25-скоров документ, где термин есть в
трёх полях, получает втрое больший вклад, хотя насыщение должно это гасить. Заголовок в
наших данных короткий и точный, параметры — длинные и шумные, поэтому нужны разные b.

idf считается по вхождению термина в документ ЦЕЛИКОМ (в любое поле), а не по полям.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class Field:
    """Одно поле документа: имя, вес в BM25F и собственный коэффициент нормировки длины b."""
    name: str
    weight: float
    b: float = 0.75


class BM25FIndex:
    """
    Индекс BM25F. Хранит матрицу весов W размера (V x N) и словарь, чтобы запросы
    векторизовались тем же словарём. Скор пачки запросов = Q @ W (n_q x N).
    """

    def __init__(self, fields: list[Field], k1: float = 1.2,
                 token_pattern: str = r"\S+", analyzer: str = "word",
                 ngram_range: tuple[int, int] = (1, 1), min_df: int = 1):
        self.fields = fields
        self.k1 = k1
        self._vec_kwargs = dict(analyzer=analyzer, ngram_range=ngram_range,
                                min_df=min_df, dtype=np.float32)
        if analyzer == "word":
            self._vec_kwargs["token_pattern"] = token_pattern
        self.vocabulary_: dict[str, int] | None = None
        self.W: sp.csr_matrix | None = None
        self.n_docs = 0

    # --- построение ---------------------------------------------------------

    def fit(self, field_texts: dict[str, list[str]]) -> "BM25FIndex":
        """
        field_texts: имя поля -> список текстов (уже лемматизированных), длина = числу документов.
        Все поля должны идти в одном порядке документов.
        """
        names = [f.name for f in self.fields]
        missing = [n for n in names if n not in field_texts]
        if missing:
            raise KeyError(f"нет текстов для полей: {missing}")
        n_docs = len(field_texts[names[0]])
        for n in names:
            if len(field_texts[n]) != n_docs:
                raise ValueError(f"поле {n}: {len(field_texts[n])} текстов, ожидалось {n_docs}")
        self.n_docs = n_docs

        # Единый словарь по конкатенации всех полей: термин должен иметь один индекс везде.
        cv = CountVectorizer(**self._vec_kwargs)
        joined = [" ".join(field_texts[n][i] for n in names) for i in range(n_docs)]
        cv.fit(joined)
        self.vocabulary_ = cv.vocabulary_
        V = len(self.vocabulary_)

        # tf~ — взвешенная сумма нормированных по длине частот полей.
        tf_tilde = sp.csr_matrix((n_docs, V), dtype=np.float32)
        # Наличие термина в документе (для idf) — логическое ИЛИ по полям.
        present = sp.csr_matrix((n_docs, V), dtype=bool)
        counter = CountVectorizer(vocabulary=self.vocabulary_, **self._vec_kwargs)
        for f in self.fields:
            C = counter.transform(field_texts[f.name]).tocsr().astype(np.float32)
            present = present + (C > 0)
            dl = np.asarray(C.sum(axis=1)).ravel()
            avgdl = dl.mean() if dl.mean() > 0 else 1.0
            # Нормировка длины поля: делим каждую строку на (1-b + b*|d_f|/avgdl_f).
            denom = 1.0 - f.b + f.b * dl / avgdl
            denom[denom <= 0] = 1.0
            C = sp.diags(1.0 / denom, dtype=np.float32) @ C
            tf_tilde = tf_tilde + f.weight * C
        tf_tilde = tf_tilde.tocsr()
        tf_tilde.eliminate_zeros()

        df = np.asarray(present.tocsc().sum(axis=0)).ravel().astype(np.float64)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        # Насыщение применяем к ненулевым элементам: при tf~=0 вклад всё равно нулевой.
        w = tf_tilde.copy()
        w.data = w.data * (self.k1 + 1.0) / (self.k1 + w.data)
        w = w @ sp.diags(idf, dtype=np.float32)          # N x V
        self.W = w.T.tocsr()                              # V x N — так удобнее умножать запросы
        return self

    # --- запросы -----------------------------------------------------------

    def transform_queries(self, queries: list[str]) -> sp.csr_matrix:
        """Запросы -> бинарная матрица (n_q x V). Бинарная: повтор слова в запросе не усиливает."""
        if self.vocabulary_ is None:
            raise RuntimeError("индекс не построен")
        vec = CountVectorizer(vocabulary=self.vocabulary_, binary=True, **self._vec_kwargs)
        return vec.transform(queries)

    def score_batch(self, Q: sp.csr_matrix, lo: int, hi: int) -> np.ndarray:
        """Плотная матрица скоров (hi-lo) x n_docs для срезa запросов [lo:hi)."""
        return (Q[lo:hi] @ self.W).toarray()


def single_field_index(docs: list[str], k1: float = 1.2, b: float = 0.75) -> BM25FIndex:
    """Классический BM25 по одному слитному полю — точка отсчёта, см. scripts/eval_bm25_baseline.py."""
    return BM25FIndex([Field("all", weight=1.0, b=b)], k1=k1).fit({"all": docs})


class BM25FBuilder:
    """
    BM25F с ОТДЕЛЁННЫМ подбором весов полей.

    Зачем отдельный класс: подбор весов — это десятки конфигураций, а самое дорогое в
    построении индекса (лемматизация, словарь, подсчёт частот по полям, нормировка длин)
    от весов НЕ зависит. Здесь тяжёлая часть делается один раз в fit(), а combine(weights)
    собирает матрицу весов из готовых нормированных частот за секунды.

    Математика та же, что в BM25FIndex: tf~ = sum_f w_f * tf_f/(1-b_f+b_f*|d_f|/avgdl_f),
    затем насыщение idf * tf~(k1+1)/(k1+tf~).
    """

    def __init__(self, k1: float = 1.2, token_pattern: str = r"\S+",
                 analyzer: str = "word", ngram_range: tuple[int, int] = (1, 1), min_df: int = 1):
        self.k1 = k1
        self._vec_kwargs = dict(analyzer=analyzer, ngram_range=ngram_range,
                                min_df=min_df, dtype=np.float32)
        if analyzer == "word":
            self._vec_kwargs["token_pattern"] = token_pattern
        self.vocabulary_: dict[str, int] | None = None
        self.norm_counts: dict[str, sp.csr_matrix] = {}   # поле -> tf/(нормировка длины)
        self.idf: np.ndarray | None = None
        self.n_docs = 0
        self.field_b: dict[str, float] = {}

    def fit(self, field_texts: dict[str, list[str]], b: dict[str, float]) -> "BM25FBuilder":
        """
        field_texts: поле -> тексты (лемматизированные), одинаковый порядок документов.
        b: поле -> коэффициент нормировки длины. Разный b по полям обязателен: заголовок
           короткий и точный (b мал -> длина почти не наказывается), описание длинное
           и рыхлое (b велик).
        """
        names = list(field_texts)
        self.n_docs = len(field_texts[names[0]])
        self.field_b = dict(b)

        cv = CountVectorizer(**self._vec_kwargs)
        joined = [" ".join(field_texts[n][i] for n in names) for i in range(self.n_docs)]
        cv.fit(joined)
        self.vocabulary_ = cv.vocabulary_
        del joined

        counter = CountVectorizer(vocabulary=self.vocabulary_, **self._vec_kwargs)
        present = sp.csr_matrix((self.n_docs, len(self.vocabulary_)), dtype=bool)
        for n in names:
            C = counter.transform(field_texts[n]).tocsr().astype(np.float32)
            present = present + (C > 0)
            dl = np.asarray(C.sum(axis=1)).ravel()
            avgdl = dl.mean() if dl.mean() > 0 else 1.0
            bf = b.get(n, 0.75)
            denom = 1.0 - bf + bf * dl / avgdl
            denom[denom <= 0] = 1.0
            self.norm_counts[n] = (sp.diags(1.0 / denom, dtype=np.float32) @ C).tocsr()

        df = np.asarray(present.tocsc().sum(axis=0)).ravel().astype(np.float64)
        self.idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        return self

    def combine(self, weights: dict[str, float]) -> sp.csr_matrix:
        """Матрица весов W (V x N) для заданных весов полей. Поле с весом 0 пропускается."""
        tf = None
        for n, w in weights.items():
            if w == 0 or n not in self.norm_counts:
                continue
            part = self.norm_counts[n] * np.float32(w)
            tf = part if tf is None else tf + part
        if tf is None:
            raise ValueError("все веса нулевые")
        tf = tf.tocsr()
        tf.data = tf.data * (self.k1 + 1.0) / (self.k1 + tf.data)
        return (tf @ sp.diags(self.idf, dtype=np.float32)).T.tocsr()

    def transform_queries(self, queries: list[str]) -> sp.csr_matrix:
        vec = CountVectorizer(vocabulary=self.vocabulary_, binary=True, **self._vec_kwargs)
        return vec.transform(queries)
