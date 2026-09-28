"""
Добыча инвентаря ключей из item_infm_params_text через ВЕТВЯЩУЮСЯ ВАРИАТИВНОСТЬ.

Проблема: параметры лежат плоской строкой «Ключ Значение Ключ Значение …» без разделителей,
а чтобы чистить мусор и выделять ценные поля, нужно знать границы значений — то есть список
ключей-разделителей.

Наивный подсчёт частотных фраз не работает: самые частотные фразы склеивают конец одного
ключа с началом следующего («Начальная цена Тип стоимости за»). Нужен признак ГРАНИЦЫ.

Признак: после КЛЮЧА стоит значение, а значений у ключа много разных. После обрывка
«Начальная цена Тип стоимости за» стоит почти всегда одно слово («услугу») — вариативности нет.
Это классическая accessor variety / branching entropy.

Правила отбора:
  1. Кандидат: фраза 1..5 слов с заглавной, частота >= MIN_FREQ, число РАЗНЫХ следующих
     слов >= MIN_BRANCH (после ключа идёт много разных значений).
  2. Фраза отбрасывается, если ВНУТРИ неё (не с начала) начинается другой кандидат —
     значит, она перескочила границу ключ|значение.
  3. Из вложенных префиксов («Вид» ⊂ «Вид услуги») отбрасывается короткий, если он почти
     всегда встречается только как часть длинного (доля >= PREFIX_ABSORB). «Тип» остаётся,
     потому что расходится на «Тип услуги» и «Тип стоимости за услугу».

Итог — cache/param_keys.json. Разделение на «мусор» и «ценное» делается ВРУЧНУЮ в
src/params.py: ценность поля для поиска — предметное решение, статистика его не выведет.

Запуск: python scripts/discover_param_keys.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import CACHE, val_dir

SAMPLE = 60000
MIN_FREQ = 150          # реже — не влияет на BM25 и не стоит риска ложного ключа
MIN_BRANCH = 15         # столько разных слов должно идти после ключа
MAX_KEY_WORDS = 5
PREFIX_ABSORB = 0.85    # короткий префикс поглощён длинным, если встречается только в нём

CAP = re.compile(r"^[А-ЯЁA-Z][а-яёa-z]")


def main() -> None:
    texts = (pd.read_parquet(val_dir(42) / "val_corpus.parquet",
                             columns=["item_infm_params_text"])
             .item_infm_params_text.fillna("")
             .sample(SAMPLE, random_state=0).tolist())
    texts = [t.split() for t in texts if t.strip()]

    # --- 1. частота фраз и множество следующих слов ------------------------
    freq = collections.Counter()
    nxt: dict[str, set[str]] = collections.defaultdict(set)
    for w in texts:
        L = len(w)
        for i in range(L):
            if not CAP.match(w[i]):
                continue
            for n in range(1, min(MAX_KEY_WORDS, L - i) + 1):
                p = " ".join(w[i:i + n])
                freq[p] += 1
                if i + n < L:
                    nxt[p].add(w[i + n])

    cand = {p for p, c in freq.items() if c >= MIN_FREQ and len(nxt[p]) >= MIN_BRANCH}
    print(f"кандидатов после частоты+вариативности: {len(cand)}")

    # --- 2. выкинуть фразы, внутри которых начинается другой кандидат -----
    def crosses_boundary(p: str) -> bool:
        w = p.split()
        return any(" ".join(w[i:j]) in cand
                   for i in range(1, len(w)) for j in range(i + 1, len(w) + 1))

    cand = {p for p in cand if not crosses_boundary(p)}
    print(f"после снятия склеек ключ|значение: {len(cand)}")

    # --- 3. поглощение коротких префиксов ---------------------------------
    keys = set(cand)
    for p in sorted(cand, key=lambda x: len(x.split())):
        w = p.split()
        for longer in cand:
            lw = longer.split()
            if len(lw) > len(w) and lw[:len(w)] == w and freq[longer] / freq[p] >= PREFIX_ABSORB:
                keys.discard(p)
                break
    print(f"после поглощения префиксов: {len(keys)} ключей")

    # --- 4. проверка покрытия: сколько текста разбирается на (ключ, значение) ---
    ordered = sorted(keys, key=lambda k: (-len(k.split()), -len(k)))
    rx = re.compile(r"(?<![\w])(" + "|".join(re.escape(k) for k in ordered) + r")(?![\w])")
    key_hits = collections.Counter()
    val_words, prefix_unparsed, n_pairs = [], [], []
    for w in texts[:20000]:
        t = " ".join(w)
        spans = [(m.start(), m.end(), m.group(1)) for m in rx.finditer(t)]
        prefix_unparsed.append(spans[0][0] if spans else len(t))
        n_pairs.append(len(spans))
        for idx, (a, b, k) in enumerate(spans):
            key_hits[k] += 1
            end = spans[idx + 1][0] if idx + 1 < len(spans) else len(t)
            val_words.append(len(t[b:end].split()))

    vw = pd.Series(val_words)
    print(f"\nпар (ключ,значение) на объявление: медиана {pd.Series(n_pairs).median():.0f}")
    print(f"длина значения в словах: медиана {vw.median():.0f}, p90 {vw.quantile(.9):.0f}, "
          f"доля значений >6 слов {(vw > 6).mean():.3f}")
    print(f"неразобранный префикс до первого ключа: медиана "
          f"{pd.Series(prefix_unparsed).median():.0f} символов")

    with (CACHE / "param_keys.json").open("w") as fh:
        json.dump({"n_keys": len(keys), "keys": [[k, key_hits[k]] for k in
                   sorted(keys, key=lambda x: -key_hits[x])]}, fh, ensure_ascii=False, indent=1)
    print(f"\n=== ключи по частоте (всего {len(keys)}) ===")
    for k in sorted(keys, key=lambda x: -key_hits[x]):
        print(f"{key_hits[k]:8d}  {k}")


if __name__ == "__main__":
    main()
