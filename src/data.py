"""
Чтение исходных parquet.

Зачем отдельный слой: (1) координаты и цена лежат как decimal128 — pandas превращает их в
объекты Decimal, с которыми numpy не считает, поэтому приводим к float в одном месте;
(2) train.parquet — 490 МБ и 19 колонок, почти всегда нужны 3-4 из них, читаем только их;
(3) item_id в benchmark_queries имеет тип large_string, а в train — string, при склейке
таблиц pyarrow нужен явный каст схемы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .paths import BENCH_ITEMS, BENCH_QUERIES, TRAIN

# Колонки-десятичные, которые всегда приводим к float64
DECIMAL_COLS = ("item_price", "item_latitude", "item_longitude")

# Метаданные объявления, нужные для построения корпуса и признаков
ITEM_COLS = [
    "item_id", "item_title_raw", "item_infm_params_text", "item_description_raw",
    "item_microcat_id", "item_category_id", "item_location_id",
    "item_latitude", "item_longitude", "item_price", "item_rating",
    "item_rating_reviews_count", "item_is_phone_hidden", "item_is_message_forbidden",
]


def _to_float(df: pd.DataFrame) -> pd.DataFrame:
    """decimal128 -> float64 для колонок, по которым потом считается арифметика."""
    for c in DECIMAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("float64")
    return df


def read_train(columns: list[str] | None = None) -> pd.DataFrame:
    """Строки train (497 673 пары «запрос -> выбранное объявление»)."""
    return _to_float(pq.read_table(TRAIN, columns=columns).to_pandas())


def read_bench_queries() -> pd.DataFrame:
    """2 452 запроса бенчмарка. query_id приводим к обычному str."""
    df = pq.read_table(BENCH_QUERIES).to_pandas()
    for c in ("query_id", "search_query", "search_infm_params_text"):
        df[c] = df[c].astype(str)
    return df


def read_bench_items(columns: list[str] | None = None) -> pd.DataFrame:
    """Корпус бенчмарка: 189 212 объявлений."""
    return _to_float(pq.read_table(BENCH_ITEMS, columns=columns).to_pandas())


def train_items_subset(item_ids: set[str], columns: list[str] | None = None) -> pd.DataFrame:
    """
    Метаданные объявлений из train по списку id. Нужно для валидационного корпуса:
    позитивы валидации живут только в train, в benchmark_items их нет.
    Фильтруем на уровне pyarrow (не pandas), чтобы не поднимать 490 МБ в память целиком.
    """
    cols = columns or ITEM_COLS
    t = pq.read_table(TRAIN, columns=cols)
    keep = pc.is_in(t["item_id"], value_set=pa.array(list(item_ids), type=t.schema.field("item_id").type))
    return _to_float(t.filter(keep).to_pandas())


def concat_items(bench: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """
    Корпус валидации = корпус бенчмарка + позитивы валидации из train.
    Дубликаты item_id убираем, приоритет у строки из бенчмарка (она встречается первой).
    """
    both = pd.concat([bench, extra[bench.columns]], ignore_index=True)
    return both.drop_duplicates("item_id").reset_index(drop=True)
