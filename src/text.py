"""
Нормализация и лемматизация русского текста запросов и объявлений.

Зачем: BM25 сопоставляет запрос и объявление по совпадению токенов, а в русском одно и то же
слово в запросе и в заголовке стоит в разных формах («ремонт стиральных машин» против
«Ремонт стиральной машины»). Без лемматизации теряется большая часть совпадений.

pymorphy3 разбирает ~20k слов/с — на 192k объявлений это минуты. Поэтому кэш на уровне
УНИКАЛЬНЫХ токенов (их десятки тысяч, а не миллионы вхождений) плюс сохранение кэша на диск,
чтобы повторные прогоны были мгновенными. Требование ТЗ §5.
"""
from __future__ import annotations
import pickle
import re
from typing import Iterable

from .paths import CACHE

# Токен = последовательность русских/латинских букв и цифр. Пунктуация — разделитель.
_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_CACHE_FILE = CACHE / "lemma_cache.pkl"


class Lemmatizer:
    """Лемматизатор с кэшем «токен -> нормальная форма», переживающим перезапуск процесса."""

    def __init__(self, persist: bool = True):
        import pymorphy3

        self._morph = pymorphy3.MorphAnalyzer()
        self._persist = persist
        self._cache: dict[str, str] = {}
        if persist and _CACHE_FILE.exists():
            with _CACHE_FILE.open("rb") as fh:
                self._cache = pickle.load(fh)
        self._dirty = False

    def token(self, word: str) -> str:
        """Нормальная форма одного токена."""
        lemma = self._cache.get(word)
        if lemma is None:
            lemma = self._morph.parse(word)[0].normal_form
            self._cache[word] = lemma
            self._dirty = True
        return lemma

    def __call__(self, s) -> str:
        """Строка -> строка лемм через пробел. 'ё' сводим к 'е' (в данных пишут и так, и так)."""
        low = str(s).lower().replace("ё", "е")
        return " ".join(self.token(w) for w in _TOKEN_RE.findall(low))

    def many(self, texts: Iterable) -> list[str]:
        """Лемматизация последовательности текстов (удобно для .map по Series)."""
        return [self(t) for t in texts]

    def save(self) -> None:
        if self._persist and self._dirty:
            with _CACHE_FILE.open("wb") as fh:
                pickle.dump(self._cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
            self._dirty = False

    @property
    def vocab_size(self) -> int:
        return len(self._cache)


def norm_query(s) -> str:
    """
    Нормализация запроса для СКЛЕЙКИ событий (не для поиска): нижний регистр, ё->е,
    пунктуация -> пробел, схлопывание пробелов. Так «Ремонт  стиральных машин!» и
    «ремонт стиральных машин» становятся одним запросом. Ровно эта функция определяет,
    какие строки train считаются одним поисковым событием, и она же даёт оценку
    «37.5% запросов бенчмарка встречались в train».
    """
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower().replace("ё", "е")).split())
