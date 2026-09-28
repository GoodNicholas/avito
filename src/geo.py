"""
Гео: центры локаций поиска и расстояния.

Зачем: в данных нет координат самого поискового запроса — есть только search_location_id.
Координаты центра локации оцениваем как МЕДИАНУ координат выбранных в этой локации объявлений
по обучающей части train (медиана, не среднее: устойчива к «выездным» объявлениям с
координатами в другом городе). Все локации бенчмарка встречаются в train (факт из ТЗ §3.6).

Гео — сильнейший сигнал после текста: 83.1% пар train имеют локацию поиска = локации
объявления, медианное расстояние до позитива 3.9 км. Но жёсткий фильтр по location_id
запрещён: у 17.4% запросов бенчмарка в своей локации нет ни одного объявления.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0


def location_centers(train_geo: pd.DataFrame) -> pd.DataFrame:
    """
    search_location_id -> (lat, lon) медиана координат выбранных объявлений.
    На вход — уже отфильтрованные по fit_mask строки train с колонками
    search_location_id, item_latitude, item_longitude.
    """
    g = train_geo.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
    return g.rename(columns={"item_latitude": "lat", "item_longitude": "lon"})


def haversine_km(qlat_rad: float, qlon_rad: float,
                 ilat_rad: np.ndarray, ilon_rad: np.ndarray) -> np.ndarray:
    """
    Расстояние от одной точки запроса до всех объявлений корпуса, км.
    Аргументы в РАДИАНАХ (перевод делается один раз снаружи — это горячий цикл).
    """
    a = (np.sin((ilat_rad - qlat_rad) / 2) ** 2
         + np.cos(qlat_rad) * np.cos(ilat_rad) * np.sin((ilon_rad - qlon_rad) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
