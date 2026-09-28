"""
Сборка полей документа и их лемматизация, с кэшем на диске.

ЗАЧЕМ: BM25F нужен не один текст объявления, а несколько полей с разными весами. Разбор
параметров (src/params.py) и лемматизация 192k объявлений занимают минуты, а при подборе
весов индекс пересобирается десятки раз — поэтому результат кэшируется в cache/ по хешу
(набор полей + версия парсера + размер корпуса).

Поля:
  title   — item_title_raw, самый точный сигнал (но у 45.6% корпуса заголовок не уникален,
            замер в scripts/analyze_data.py, поэтому одного title мало);
  kind    — «Вид услуги» / «Тип услуги» и родственные: совпадают с фильтрами запроса в 98.2%/95.4%;
  service — конкретные услуги из прайса: ближе всего к формулировке запроса;
  place   — гео-текст («Место оказания услуг», «Куда выезжаете»);
  other   — значения неизвестных ключей, низкий вес;
  desc    — начало item_description_raw: полезно против «нет общих лемм» (5.4% потерь, см. «Анализ ошибок» в README),
            но описания рыхлые и длинные, поэтому обрезаются и получают малый вес.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd

from .params import parse_many
from .paths import CACHE
from .text import Lemmatizer

DESC_CHARS = 400          # первые N символов описания (дальше обычно контакты и самореклама)
PARSER_VERSION = 3        # менять при правке src/params.py, иначе подхватится старый кэш
FIELD_NAMES = ("title", "kind", "service", "place", "other", "desc")


def _key(tag: str, n_docs: int) -> Path:
    h = hashlib.md5(f"{tag}|{n_docs}|{PARSER_VERSION}|{DESC_CHARS}".encode()).hexdigest()[:12]
    return CACHE / f"fields_{tag}_{h}.pkl"


def build_fields(items: pd.DataFrame, tag: str, use_cache: bool = True) -> dict[str, list[str]]:
    """
    items — датафрейм объявлений с колонками item_title_raw, item_infm_params_text,
    item_description_raw. Возвращает dict поле -> список ЛЕММАТИЗИРОВАННЫХ текстов.
    tag — метка корпуса для кэша ('val42', 'bench', ...).
    """
    path = _key(tag, len(items))
    if use_cache and path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)

    parsed = parse_many(items.item_infm_params_text.fillna("").tolist())
    raw = {
        "title": items.item_title_raw.fillna("").tolist(),
        "kind": parsed["kind"],
        "service": parsed["service"],
        "place": parsed["place"],
        "other": parsed["other"],
        "desc": items.item_description_raw.fillna("").str[:DESC_CHARS].tolist(),
    }

    lm = Lemmatizer()
    fields = {name: lm.many(texts) for name, texts in raw.items()}
    lm.save()

    with path.open("wb") as fh:
        pickle.dump(fields, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return fields


def lemmatize_queries(queries: list[str]) -> list[str]:
    lm = Lemmatizer()
    out = lm.many(queries)
    lm.save()
    return out
