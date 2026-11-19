#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 6. Линейный, бинарный и интерполяционный поиск — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-06 и правилами использования генеративного ИИ
(docs/ai-verification.md).

Каждая функция поиска работает с отсортированным по неубыванию массивом
и возвращает пару (индекс найденного элемента или -1, число сравнений ключей).

Запуск: python lab06-search-starter.py --variant N
"""
from __future__ import annotations

import argparse
import bisect
import random
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Алгоритмы поиска (реализуются вручную, без in/index/bisect в основной части)
# ---------------------------------------------------------------------------


def linear_search(a: list[int], key: int) -> tuple[int, int]:
    """Линейный поиск первого вхождения key.

    Возвращает (индекс первого вхождения или -1, число сравнений ключей).

    Ожидаемая сложность: Θ(n) в худшем и среднем случае, O(1) в лучшем.
    Алгоритм сравнивает key с элементами по одному подряд, пока не найдёт
    совпадение или не дойдёт до конца массива. Если ключа нет или он последний - потребуется
    ровно n сравнений; если ключ равновероятно распределён по позициям
    массива - в среднем n/2 сравнений, всё равно Θ(n) по порядку роста.
    Лучший случай - ключ на первой позиции - 1 сравнение, O(1), но это
    не характеризует алгоритм в целом.
    """
    if len(a) == 0:
        return (-1, 0)
    count = 0
    for i in range(len(a)):
        count +=1
        if a[i] == key:
            return (i, count)
    return (-1, count)


def binary_search(a: list[int], key: int) -> tuple[int, int]:
    """Бинарный поиск key в отсортированном массиве.

    Возвращает (индекс любого вхождения или -1, число сравнений ключей).
    Аккуратно поддерживайте инвариант обеих границ lo и hi: искомый элемент,
    если он есть, всегда лежит в текущем отрезке; цикл обязан завершаться.

    Ожидаемая сложность: O(log n) в худшем и среднем случае, O(1) в лучшем.
    Каждая итерация сравнивает key с элементом в середине текущего отрезка
    и отбрасывает ровно половину оставшихся элементов - ту, где искомого
    ключа точно нет (это и есть инвариант: если ключ в массиве есть, он
    всегда внутри [lo, hi]). После k итераций в отрезке остаётся не больше
    n / 2^k элементов; цикл заканчивается, когда отрезок пуст или найден
    ключ, откуда k > log2(n) - число
    итераций (и сравнений) ограничено ⌈log2(n)⌉ + 1 независимо от того,
    есть ключ или нет и в какой позиции он лежит - в отличие от линейного
    поиска, здесь нет разницы между худшим и средним случаем по порядку
    роста. Лучший случай - ключ ровно в середине отрезка - 1 сравнение.
    """
    lo, hi = 0, len(a) -1
    count =0
    while lo <= hi:
        mid = (lo + hi) //2
        count +=1
        if key == a[mid]:
            return (mid, count)
        if key < a[mid]:
            hi = mid -1
        if key > a[mid]:
            lo = mid + 1
    return (-1, count)
    # TODO: реализовать через инвариант границ lo <= hi (или lo < hi — но
    # TODO: тогда согласовать вычисление mid и сужение отрезка!)


def interpolation_search(a: list[int], key: int) -> tuple[int, int]:
    """Интерполяционный поиск key в отсортированном массиве.

    Возвращает (индекс любого вхождения или -1, число сравнений ключей).
    Позиция зонда: mid = lo + (key - a[lo]) * (hi - lo) // (a[hi] - a[lo]).
    Обязательно защититься от деления на ноль при a[hi] == a[lo] и от выхода
    key за пределы [a[lo], a[hi]].

    Средняя сложность на равномерных данных: O(log log n); в худшем случае
    (сильно неравномерное распределение значений) деградирует до O(n).
    Зонд устроен как линейная интерполяция: предполагается, что ключи
    распределены в отрезке [a[lo], a[hi]] равномерно, и mid ставится ровно
    в ту точку, где key должен был бы оказаться при таком распределении -
    в отличие от бинарного поиска, который всегда берёт середину отрезка
    индексов, не глядя на сами значения. Если предположение о равномерности
    верно, зонд почти сразу попадает рядом с искомым ключом, и после
    каждого шага остаётся отрезок длины порядка sqrt(текущей длины); серия n -> sqrt(n) -> sqrt(sqrt(n))
    -> ... доходит до 1 за O(log log n) шагов.
    Худший случай - когда распределение сильно неравномерно (например,
    экспоненциальное: много близких мелких значений и редкие большие) -
    формула зонда систематически промахивается мимо настоящей позиции,
    отрезок на каждом шаге сужается всего на несколько элементов, а не
    в разы, и алгоритм вырождается до O(n).
    """
    # TODO: реализовать с зондом по формуле интерполяции и счётчиком сравнений
    if len(a) == 0:
        return (-1, 0)
    count =0
    lo = 0
    hi = len(a) - 1
    while a[lo] < key and a[hi] > key and lo <= hi:
        count +=1
        if a[lo] == a[hi]:
            break
        mid = lo + (key - a[lo]) * (hi - lo) // (a[hi] - a[lo])
        if a[mid] > key:
            hi = mid - 1
        elif a[mid] < key:
            lo = mid + 1
        else:
            return (mid, count)
    count +=1
    if a[lo] == key:
        return (lo, count)
    count +=1
    if a[hi] == key:
        return (hi, count)
    return (-1, count)
    


# ---------------------------------------------------------------------------
# 2. Данные по варианту: два распределения, присутствующие/отсутствующие ключи
# ---------------------------------------------------------------------------


def make_dataset(n: int, distribution: str, rng: random.Random) -> list[int]:
    """Отсортированный массив n ключей: 'uniform' — равномерное распределение,
    'exponential' — неравномерное (экспоненциальное, значения сгущаются у нуля)."""
    if distribution == "uniform":
        return sorted(rng.randint(0, 10 * n) for _ in range(n))
    if distribution == "exponential":
        return sorted(int(rng.expovariate(1.0 / 1000.0)) for _ in range(n))
    raise ValueError(f"неизвестное распределение: {distribution!r}")


def pick_keys(a: list[int], rng: random.Random, count: int,
              present: bool) -> list[int]:
    """count ключей для серии поисков: присутствующие берутся из массива,
    отсутствующие подбираются вне множества его значений."""
    if present:
        return [a[rng.randrange(len(a))] for _ in range(count)]
    have = set(a)
    hi = (a[-1] if a else 0) + 2
    keys: list[int] = []
    while len(keys) < count:
        k = rng.randint(-hi, hi)
        if k not in have:
            keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# 3. Репрезентативные тесты и инварианты (шаги 1–3 методики верификации)
# ---------------------------------------------------------------------------

ALGORITHMS = [
    ("linear", linear_search),
    ("binary", binary_search),
    ("interpolation", interpolation_search),
]


def self_check() -> None:
    """Граничные случаи + сверка с эталоном (оператор in, list.index, bisect)."""
    for name, fn in ALGORITHMS:
        # Пустой массив: элемент не найден, сравнений нет.
        assert fn([], 1) == (-1, 0), f"{name}: пустой массив"
        # Один элемент: найден / не найден.
        assert fn([7], 7)[0] == 0, f"{name}: один элемент, ключ найден"
        assert fn([7], 3)[0] == -1, f"{name}: один элемент, ключ отсутствует"
        # Первый, последний и отсутствующие ключи (внутри и вне диапазона).
        a = [1, 3, 3, 5, 8, 13, 21]
        assert fn(a, 1)[0] == 0, f"{name}: первый элемент"
        assert fn(a, 21)[0] == len(a) - 1, f"{name}: последний элемент"
        assert fn(a, 4)[0] == -1, f"{name}: отсутствующий ключ внутри диапазона"
        assert fn(a, 100)[0] == -1, f"{name}: ключ вне диапазона"

    # Сверка с эталоном стандартной библиотеки на случайных массивах.
    rng = random.Random(0)
    for _ in range(300):
        a = sorted(rng.randint(0, 60) for _ in range(rng.randint(1, 40)))
        key = rng.randint(-5, 65)
        # Эталон присутствия ключа — bisect_left (и, эквивалентно, оператор in).
        j = bisect.bisect_left(a, key)
        present = j < len(a) and a[j] == key
        assert present == (key in a)
        for name, fn in ALGORITHMS:
            idx, cmp_count = fn(a, key)
            assert cmp_count >= 1, f"{name}: счётчик сравнений не ведётся"
            if present:
                # При дубликатах бинарный/интерполяционный могут вернуть
                # любое вхождение — проверяем значение по индексу.
                assert idx != -1 and a[idx] == key, f"{name}: ключ не найден"
            else:
                assert idx == -1, f"{name}: найден отсутствующий ключ"
        # Линейный поиск обязан возвращать первое вхождение — эталон list.index.
        if present:
            assert linear_search(a, key)[0] == a.index(key)
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 4. Бенчмарк: сравнения и время от n (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5      # число повторов серии замеров (берётся медиана)
N_KEYS = 200     # число ключей в одной серии поисков


def bench(fn, a: list[int], keys: list[int]) -> float:
    """Медиана времени одной серии поисков по REPEATS запускам, с прогревом."""
    for k in keys:  # прогрев — не учитывается
        fn(a, k)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for k in keys:
            fn(a, k)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def mean_comparisons(fn, a: list[int], keys: list[int]) -> float:
    """Среднее число сравнений ключей на один поиск."""
    return statistics.mean(fn(a, k)[1] for k in keys)


DISTRIBUTIONS = ("uniform", "exponential")


def run_benchmarks(seed: int) -> dict:
    """Печатает полную таблицу (присутствующие и отсутствующие ключи) и
    возвращает результаты для присутствующих ключей — для графиков
    (distribution -> algorithm -> список (n, среднее число сравнений, время))."""
    rng = random.Random(seed)
    results = {dist: {name: [] for name, _ in ALGORITHMS} for dist in DISTRIBUTIONS}
    for distribution in DISTRIBUTIONS:
        print(f"\nРаспределение: {distribution}")
        for n in SIZES:
            a = make_dataset(n, distribution, rng)
            for present in (True, False):
                keys = pick_keys(a, rng, N_KEYS, present)
                label = "есть" if present else "нет"
                for name, fn in ALGORITHMS:
                    if name == "linear" and n > 30_000:
                        continue  # линейный поиск: большие n занимают минуты
                    cmp_mean = mean_comparisons(fn, a, keys)
                    t = bench(fn, a, keys)
                    print(f"  n={n:>7}  ключ {label:<4}  {name:<13}"
                          f"  сравнений={cmp_mean:>10.1f}  t={t:.6f} c")
                    if present:
                        results[distribution][name].append((n, cmp_mean, t))
    return results


def plot_results(results: dict, out_dir: Path) -> None:
    """Графики «сравнения от n» и «время от n» для присутствующих ключей,
    отдельно для каждого распределения (докажем/опровергнем O(log n) и
    O(log log n) по реальным замерам — КИМ-06, п.5)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен — графики пропущены "
              "(pip install -r requirements.txt)")
        return

    fig, axes = plt.subplots(2, len(DISTRIBUTIONS), figsize=(7 * len(DISTRIBUTIONS), 9))
    for col, dist in enumerate(DISTRIBUTIONS):
        ax_cmp, ax_time = axes[0, col], axes[1, col]
        for name, points in results[dist].items():
            if not points:
                continue
            ns = [p[0] for p in points]
            cmps = [p[1] for p in points]
            ts = [p[2] for p in points]
            ax_cmp.plot(ns, cmps, marker="o", label=name)
            ax_time.plot(ns, ts, marker="o", label=name)
        for ax, ylabel, title in (
            (ax_cmp, "среднее число сравнений", f"{dist}: сравнения от n"),
            (ax_time, f"время {N_KEYS} поисков, с", f"{dist}: время от n"),
        ):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("размер массива n")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, which="both", linewidth=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "lab06_search.png"
    fig.savefig(path, dpi=150)
    print(f"\nГрафик сохранён: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    ap.add_argument("--out", type=Path, default=Path.cwd(),
                    help="каталог для графика (по умолчанию текущий)")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    results = run_benchmarks(seed)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()
