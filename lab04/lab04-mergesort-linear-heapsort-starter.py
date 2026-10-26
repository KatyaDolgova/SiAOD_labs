#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 4. MergeSort, Counting, Radix и HeapSort: оценка производительности — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-04 и правилами использования генеративного ИИ
(docs/ai-verification.md). QuickSort перенесите из собственной ЛР 3.

Запуск: python lab04-mergesort-linear-heapsort-starter.py --variant N
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Сортировки (реализуются вручную; sorted() используется только как эталон)
# ---------------------------------------------------------------------------


def merge(left: list, right: list, key) -> list:
    """Стабильное слияние двух отсортированных списков в новый список.

    Стабильность: при равенстве ключей (key(left[i]) == key(right[j]))
    условие <= забирает элемент из left первым - поэтому элемент левой
    половины, который во входном массиве всегда стоял раньше элементов
    правой половины, никогда не окажется позади равного ему элемента
    из right. Ожидаемая сложность: O(len(left) + len(right)) - каждый
    элемент обоих списков посещается ровно один раз.
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(a: list, key=lambda x: x) -> list:
    """MergeSort: возвращает новый отсортированный список.

    Ожидаемая сложность: T(n log n) в худшем, среднем и лучшем случае одновременно:
    деление всегда ровно пополам по длине, поэтому глубина рекурсии гарантированно log n независимо
    от входных данных. (количество уровней) × (работа на уровень)
    """
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    left = merge_sort(a[:mid], key)
    right = merge_sort(a[mid:], key)
    return merge(left, right, key)


def counting_sort(a: list[int], key_max: int) -> list[int]:
    """Устойчивая сортировка подсчётом: ключи - целые из диапазона [0, key_max].

    Схема: массив счётчиков -> префиксные суммы -> расстановка с конца.
    Время: O(n + k), память: O(n + k), где k = key_max + 1 - проход по
    входу для подсчёта (O(n)), проход по счётчикам для префиксных сумм
    (O(k)), проход по входу для расстановки (O(n)). Ограничение диапазона
    ключей: алгоритм теряет смысл, когда k сопоставим с n^2 или больше -
    например, при k = 10^9 массив счётчиков такого размера физически не
    помещается в память
    """
    k = key_max + 1
    count = [0] * k
    for x in a:
        count[x] += 1
    for i in range(1, k):
        count[i] += count[i - 1]
    result = [0] * len(a)
    for x in reversed(a):
        count[x] -= 1
        result[count[x]] = x
    return result


def _counting_sort_by_digit(a: list[int], exp: int) -> list[int]:
    """Устойчивая сортировка по одной десятичной цифре (a[i] // exp) % 10."""
    n = len(a)
    count = [0] * 10
    for x in a:
        digit = (x // exp) % 10
        count[digit] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    result = [0] * n
    for x in reversed(a):
        digit = (x // exp) % 10
        count[digit] -= 1
        result[count[digit]] = x
    return result


def radix_sort_lsd(a: list[int]) -> list[int]:
    """RadixSort LSD по десятичным разрядам, неотрицательные целые.

    Поверх устойчивой поразрядной counting-сортировки: от младшего разряда
    к старшему. Устойчивость внутреннего прохода: после
    сортировки по разряду i результат должен сохранить относительный
    порядок, установленный сортировкой по разрядам.
    Ожидаемая сложность: O(d * (n + k)), где d - число разрядов,
    k = 10 (база счётчика на каждом проходе) -
    то есть фактически O(d * n).
    """
    if not a:
        return []
    result = list(a)
    max_val = max(result)
    exp = 1
    while max_val // exp > 0:
        result = _counting_sort_by_digit(result, exp)
        exp *= 10
    return result


def sift_down(a: list[int], i: int, n: int) -> None:
    """Просейка вниз элемента i в куче a[0:n] (max-heap), на месте.

    Пока у узла i есть потомок больше его самого, меняем местами с
    большим из двух потомков и
    продолжаем спуск с новой позиции - иначе после обмена свойство
    кучи может остаться нарушенным в поддереве того потомка, с которым
    не поменялись. Глубина одной просейки - O(log n) (высота кучи).
    """
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        if left < n and a[left] > a[largest]:
            largest = left
        if right < n and a[right] > a[largest]:
            largest = right
        if largest == i:
            break
        a[i], a[largest] = a[largest], a[i]
        i = largest


def heap_sort(a: list[int]) -> None:
    """HeapSort in-place: сортирует список a на месте, без вспомогательного массива.

    Схема: построение max-кучи просейками вниз снизу вверх (с последнего
    родителя до корня) - O(n), хотя каждая просейка
    стоит до O(log n), узлов у основания кучи большинство и их просейка
    почти мгновенна (высота поддерева мала), а дорогих просеек у корня
    мало - сумма по уровням даёт ряд, который сходится
    к O(n). Затем n-1 раз: обмен корня (максимума) с последним
    элементом невыделенной части кучи, уменьшение границы кучи на 1 и
    просейка нового корня, O(log n) на извлечение, итого O(n log n) на
    всю фазу извлечения. Суммарно: O(n) + O(n log n) = O(n log n).
    """
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end)


def quick_sort(a: list[int], rng: random.Random | None = None) -> list[int]:
    """QuickSort с рандомизацией опорного элемента — перенесено из ЛР 3 (КИМ-03).

    Ожидаемая сложность: в среднем O(n log n) (случайный pivot в среднем
    делит массив на сбалансированные части), в худшем O(n^2) (pivot
    каждый раз оказывается экстремальным элементом подмассива).
    """
    if rng is None:
        rng = random.Random()
    if len(a) <= 1:
        return list(a)
    pivot = a[rng.randrange(0, len(a))]
    less = [x for x in a if x < pivot]
    equal = [x for x in a if x == pivot]
    greater = [x for x in a if x > pivot]
    return quick_sort(less, rng) + equal + quick_sort(greater, rng)


# ---------------------------------------------------------------------------
# 2. Верификация: инварианты + сверка с эталоном sorted()
#    (шаги 1–3 методики docs/ai-verification.md)
# ---------------------------------------------------------------------------


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    """Явная проверка вместо assert: assert-ы исчезают при запуске python -O."""
    if not condition:
        raise SelfCheckError(message)


def heap_sorted(a: list[int]) -> list[int]:
    """heap_sort сортирует на месте: проверяется тот список, который ему передали."""
    buf = list(a)
    heap_sort(buf)
    return buf


# Каждая функция возвращает отсортированный список; ключи — целые из [0, 999].
SORTS = {
    "merge_sort": lambda a: merge_sort(a),
    "counting_sort": lambda a: counting_sort(a, key_max=999),
    "radix_sort_lsd": lambda a: radix_sort_lsd(a),
    "heap_sort": lambda a: heap_sorted(a),
    "quick_sort": lambda a: quick_sort(a),
}


def check_sort(fn) -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],                    # пустой массив
        [7],                   # один элемент
        [0],                   # нулевой ключ
        [5, 5, 5],             # все элементы равны
        [1, 2, 3, 4, 5],       # уже отсортирован
        [5, 4, 3, 2, 1],       # обратный порядок
    ]
    for a in boundary:
        res = fn(list(a))
        expect(isinstance(res, list), f"на {a} возвращён {type(res).__name__}, а не список")
        expect(res == sorted(a), f"на {a} получено {res}, ожидалось {sorted(a)}")

    # Случайные тесты: инвариант перестановки + сверка с эталоном
    rng = random.Random(0)
    for _ in range(200):
        a = [rng.randint(0, 999) for _ in range(rng.randint(0, 60))]
        res = fn(list(a))
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: получено {res}")
        expect(res == sorted(a), f"расходится с sorted() на {a}: получено {res}")


def check_merge_stability() -> None:
    """Стабильность слияния на парах (ключ, порядковый номер): эталон sorted() стабилен,
    поэтому при равных ключах порядок номеров обязан совпасть."""
    rng = random.Random(0)
    pairs = [(rng.randint(0, 5), i) for i in range(100)]
    got = merge_sort(pairs, key=lambda p: p[0])
    expect(got == sorted(pairs, key=lambda p: p[0]),
           "merge_sort нестабилен: при равных ключах нарушен исходный порядок номеров")


def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""
    # fn=fn фиксирует функцию текущего шага цикла (иначе все лямбды увидят последнюю)
    checks = [(name, lambda fn=fn: check_sort(fn)) for name, fn in SORTS.items()]
    checks.append(("стабильность merge_sort", check_merge_stability))
    failed = 0
    print("self_check:")
    for name, check in checks:
        try:
            check()
        except NotImplementedError as err:
            where = traceback.extract_tb(err.__traceback__)[-1].name
            print(f"  {name}: не реализовано — {where} (TODO)")
            failed += 1
        except SelfCheckError as err:
            print(f"  {name}: ОШИБКА — {err}")
            failed += 1
        except Exception:  # noqa: BLE001 — падение реализации тоже результат проверки
            print(f"  {name}: ИСКЛЮЧЕНИЕ")
            traceback.print_exc(file=sys.stdout)
            failed += 1
        else:
            print(f"  {name}: OK")
    # TODO: добавить собственные проверки (свойство кучи после построения,
    #       устойчивость поразрядного прохода radix) и описать их в отчёте.
    if failed:
        raise SystemExit(f"self_check: не пройдено разделов — {failed}; "
                         f"замеры выполняются только после всех проверок.")
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 3. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
SMALL_KEY_MAX = 100        # узкий диапазон: counting уместен
WIDE_KEY_MAX = 10**9       # широкий диапазон: counting теряет смысл (обсудить в отчёте)


def make_datasets(rng: random.Random, n: int) -> dict[str, list[int]]:
    """Массивы разных распределений и диапазонов ключей для размера n."""
    almost = list(range(n))
    for _ in range(max(1, n // 100)):  # ~1 % случайных обменов
        i, j = rng.randrange(n), rng.randrange(n)
        almost[i], almost[j] = almost[j], almost[i]
    return {
        "random_wide": [rng.randint(0, WIDE_KEY_MAX) for _ in range(n)],
        "random_small": [rng.randint(0, SMALL_KEY_MAX) for _ in range(n)],
        "sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "almost_sorted": almost,
    }


def bench(fn, data: list[int]) -> float:
    """Медиана времени fn на копии data по REPEATS запускам, с прогревом.

    Каждый запуск получает свежую копию: сортировка на месте не должна
    превращать последующие запуски в сортировку уже отсортированного массива.
    """
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        buf = list(data)
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


DISTRIBUTIONS = ["random_wide", "random_small", "sorted", "reversed", "almost_sorted"]


def log_log_slope(points: list[tuple[int, float]]) -> float:
    """Наклон прямой в осях (log n, log t) — оценка показателя степени."""
    xs = [math.log10(n) for n, _ in points]
    ys = [math.log10(t) for _, t in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def run_benchmarks(seed: int) -> dict:
    rng = random.Random(seed)
    algorithms = [
        ("merge_sort", merge_sort),
        ("counting_sort", lambda a: counting_sort(a, key_max=max(a, default=0))),
        ("radix_sort_lsd", radix_sort_lsd),
        ("heap_sort", heap_sort),
        ("quick_sort (ЛР 3)", quick_sort),
        ("sorted() (эталон)", sorted),
    ]
    results = {name: {dist: [] for dist in DISTRIBUTIONS} for name, _ in algorithms}
    for n in SIZES:
        datasets = make_datasets(rng, n)
        print(f"\nn = {n}:")
        for dist_name, data in datasets.items():
            print(f"  распределение {dist_name}:")
            for algo_name, fn in algorithms:
                if algo_name == "counting_sort" and dist_name == "random_wide":
                    # счётчики размера ~10^9 не помещаются в память — граница
                    # применимости counting; зафиксировать в таблице свойств
                    print(f"    {algo_name:>18}: пропуск (диапазон ключей)")
                    continue
                t = bench(fn, data)
                results[algo_name][dist_name].append((n, t))
                print(f"    {algo_name:>18}: t = {t:.6f} c")
    print()
    for algo_name, by_dist in results.items():
        slopes = ", ".join(
            f"{dist}={log_log_slope(points):.2f}"
            for dist, points in by_dist.items() if points
        )
        print(f"{algo_name}: наклоны log-log: {slopes}")
    return results


def plot_results(results: dict, out_dir: Path) -> None:
    """Log-log графики времени от размера входа — отдельный график на алгоритм."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен — графики пропущены "
              "(pip install -r requirements.txt)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, name in zip(axes.flat, results):
        for dist, points in results[name].items():
            if not points:
                continue
            ns = [n for n, _ in points]
            ts = [t for _, t in points]
            ax.plot(ns, ts, marker="o",
                    label=f"{dist} (наклон ≈ {log_log_slope(points):.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("размер входа n")
        ax.set_ylabel("время, с")
        ax.set_title(name)
        ax.grid(True, which="both", linewidth=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "lab04_sorts.png"
    fig.savefig(path, dpi=150)
    print(f"\nГрафик сохранён: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    ap.add_argument("--out", type=Path, default=Path.cwd(),
                    help="каталог для графиков (по умолчанию текущий)")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    results = run_benchmarks(seed)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()
