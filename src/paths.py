"""
Единая точка правды о путях проекта.

Все пути собраны здесь, чтобы перенос на другую машину 
(Colab, GPU-бокс) сводился к правке одного файла.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"          # симлинки на исходные parquet
CACHE = ROOT / "cache"        # тяжёлые артефакты: леммы, индексы, эмбеддинги
RESULTS = ROOT / "results"    # метрики экспериментов и сабмиты
CONFIGS = ROOT / "configs"

TRAIN = DATA / "train.parquet"
BENCH_ITEMS = DATA / "benchmark_items.parquet"
BENCH_QUERIES = DATA / "benchmark_queries.parquet"

for _d in (CACHE, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


def val_dir(seed: int) -> Path:
    """Артефакты валидационного сплита: у каждого сида свой подкаталог."""
    d = CACHE / f"val_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d
