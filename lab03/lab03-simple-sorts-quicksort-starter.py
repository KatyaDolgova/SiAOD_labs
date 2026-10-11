#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 3. Простые сортировки и QuickSort с рандомизацией — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-03 и правилами использования генеративного ИИ
(docs/ai-verification.md).

Запуск: python lab03-simple-sorts-quicksort-starter.py --variant N
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Простые сортировки со счётчиками (сортируют КОПИЮ входа, вход не меняют)
#    Каждая функция возвращает кортеж: (отсортированный список,
#    число сравнений ключей, число обменов/перемещений элементов).
# ---------------------------------------------------------------------------


def bubble_sort(a: list) -> tuple[list, int, int]:
    """Сортировка пузырьком.

    Ожидаемая сложность: худший и средний случай O(n^2) (обратный/случайный
    вход — полный перебор без досрочного выхода, n(n-1)/2 сравнений).
    Лучший случай O(n) на уже отсортированном входе — досрочный выход,
    если за проход не было ни одного обмена, останавливает алгоритм
    после первого же прохода (n-1 сравнений).
    """
    copy = list(a)
    comp = swaps = 0
    n = len(copy)
    for i in range(n):
        swapped = False
        for j in range(n-1-i):
            comp +=1
            if copy[j] > copy[j+1]:
                copy[j], copy[j+1] = copy[j+1], copy[j]
                swaps +=1
                swapped = True
        if not swapped:
            break
    return copy, comp, swaps

def insertion_sort(a: list) -> tuple[list, int, int]:
    """Сортировка вставками.

    Ожидаемая сложность: худший случай O(n^2) (обратный вход — каждый новый
    элемент сдвигается через весь уже отсортированный префикс). Лучший
    случай O(n) на уже отсортированном входе — внутренний while сразу
    завершается без единого сдвига, остаётся n-1 сравнений.
    """
    copy = list(a)
    comp = moves = 0
    for i  in range(1, len(copy)):
        key =copy[i]
        j = i-1
        while j >= 0 and copy[j] > key:
            comp +=1
            copy[j+1] =  copy[j]
            moves +=1
            j -=1
        if j >= 0:
            comp +=1
        copy[j+1] = key
    return copy, comp,moves


def selection_sort(a: list) -> tuple[list, int, int]:
    """Сортировка выбором.

    Ожидаемая сложность: Theta(n^2) на любом входе. Чтобы найти минимум
    среди оставшихся элементов, внутренний цикл обязан просмотреть их все
    независимо от значений — число сравнений всегда n(n-1)/2, досрочного
    выхода не существует. Единственная переменная величина — число
    обменов (не больше n-1), но она не входит в число сравнений.
    """
    copy = list(a)
    comp = swaps = 0
    n = len(copy)
    for i in range(n - 1):
        min_idx = i
        for j in range(i+1, n):
            comp +=1
            if copy[j] < copy[min_idx]:
                min_idx = j
        if min_idx != i:
            copy[i], copy[min_idx] = copy[min_idx], copy[i]
            swaps += 1
    return copy, comp, swaps


# ---------------------------------------------------------------------------
# 2. QuickSort с рандомизированным опорным элементом
# ---------------------------------------------------------------------------


def quick_sort(a: list, rng: random.Random | None = None) -> list:
    """QuickSort с выбором опорного через rng.randrange. Возвращает новый список.

    Ожидаемая сложность: в среднем O(n log n) — случайный пивот в среднем
    делит массив на сбалансированные части, глубина рекурсии ~log n, на
    каждом уровне суммарно O(n) работы на фильтрацию. В худшем случае
    O(n^2) — если пивот каждый раз оказывается экстремальным (мин/макс)
    элементом, одна из частей пуста, а другая размера n-1; рандомизация
    делает такое катастрофическое невезение на каждом уровне рекурсии
    маловероятным, а не невозможным. rng передаётся снаружи, чтобы
    запуск был воспроизводим по seed варианта.
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
# 3. Демонстрация стабильности на парах (ключ, метка)
# ---------------------------------------------------------------------------

class _ByKey:
    """Обёртка для сравнения пар только по ключу (p[0]), метка игнорируется."""
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair[0] < other.pair[0]

    def __gt__(self, other):
        return self.pair[0] > other.pair[0]

    def __eq__(self, other):
        return self.pair[0] == other.pair[0]

def stability_demo() -> None:
    """Показать, какие из четырёх сортировок стабильны.

    План: взять массив пар (ключ, метка) с повторяющимися ключами, например
    [(2, "a"), (1, "b"), (2, "c"), (1, "d")], отсортировать каждой сортировкой
    по ключу (сравнение только по p[0]!) и напечатать порядок меток при равных
    ключах. Вывод о стабильности каждой сортировки включить в отчёт.
    """
    pairs = [(2, "a"), (1, "b"), (2, "c"), (1, "d")]
    wrapped = [_ByKey(p) for p in pairs]
    algorithms = {
        "bubble_sort": lambda a: bubble_sort(a)[0],
        "insertion_sort": lambda a: insertion_sort(a)[0],
        "selection_sort": lambda a: selection_sort(a)[0],
        "quick_sort": lambda a: quick_sort(a, random.Random(0)),
    }
    print("\nДемонстрация стабильности (вход:", pairs, "):")
    for name, fn in algorithms.items():
        result = fn(list(wrapped))
        labels = [w.pair[1] for w in result]
        print(f"  {name}: порядок меток = {labels}")



# ---------------------------------------------------------------------------
# 4. Классы входов (данные по варианту, детерминированно по seed)
# ---------------------------------------------------------------------------


def make_inputs(n: int, seed: int) -> dict[str, list[int]]:
    """Три класса входов размера n: упорядоченный, случайный, обратный."""
    rng = random.Random(seed + n)  # свой поток для каждого размера
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]
    return {
        "упорядоченный": sorted(data),
        "случайный": data,
        "обратный": sorted(data, reverse=True),
    }


# ---------------------------------------------------------------------------
# 5. Верификация (шаги 1–3 методики docs/ai-verification.md)
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "bubble_sort": lambda a: bubble_sort(a)[0],
    "insertion_sort": lambda a: insertion_sort(a)[0],
    "selection_sort": lambda a: selection_sort(a)[0],
    "quick_sort": lambda a: quick_sort(a, random.Random(0)),
}


def is_sorted(a: list) -> bool:
    """Инвариант 1: неубывающий порядок элементов."""
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def self_check() -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],                    # пустой массив
        [7],                   # один элемент
        [5, 5, 5, 5],          # все элементы равны
        [1, 2, 3, 4, 5],       # уже отсортирован
        [5, 4, 3, 2, 1],       # обратный порядок
    ]
    for name, fn in ALGORITHMS.items():
        for a in boundary:
            res = fn(list(a))
            assert is_sorted(res), f"{name}: нарушен порядок на {a}"
            assert sorted(res) == sorted(a), f"{name}: не перестановка входа {a}"

    # Сверка с эталоном из стандартной библиотеки на сотнях случайных входов.
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randint(-100, 100) for _ in range(rng.randint(0, 80))]
        expected = sorted(a)
        for name, fn in ALGORITHMS.items():
            assert fn(list(a)) == expected, f"{name}: расходится с sorted() на {a}"

    # Счётчики: у сортировки выбором ровно n*(n-1)/2 сравнений на любом входе.
    _, comparisons, _ = selection_sort([3, 1, 2, 5, 4])
    assert comparisons == 10, "selection_sort: неверный счётчик сравнений"
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 6. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES_QUADRATIC = [500, 1_000, 2_000, 4_000, 8_000]     # для простых сортировок
SIZES_QUICK = [1_000, 3_000, 10_000, 30_000, 100_000]   # для QuickSort
REPEATS = 5


def bench(fn, data: list) -> float:
    """Медиана времени выполнения fn(копия data) по REPEATS запускам, с прогревом."""
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        arg = list(data)  # свежая копия: сортировка не должна получать свой результат
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def log_log_slope(points: list[tuple[int, float]]) -> float:
    """Наклон прямой в осях (log n, log t) - оценка показателя степени."""
    xs = [math.log10(n) for n, _ in points]
    ys = [math.log10(t) for _, t in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def run_benchmarks(seed: int) -> dict:
    """Замеры всех алгоритмов на трёх классах входов; результат - таблица в stdout."""
    plans = [
        ("bubble_sort", ALGORITHMS["bubble_sort"], SIZES_QUADRATIC),
        ("insertion_sort", ALGORITHMS["insertion_sort"], SIZES_QUADRATIC),
        ("selection_sort", ALGORITHMS["selection_sort"], SIZES_QUADRATIC),
        ("quick_sort", lambda a: quick_sort(a, random.Random(seed)), SIZES_QUICK),
    ]
    results: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for name, fn, sizes in plans:
        print(f"\n{name}:")
        results[name] = {"упорядоченный": [], "случайный": [], "обратный": []}
        for n in sizes:
            for cls, data in make_inputs(n, seed).items():
                t = bench(fn, data)
                results[name][cls].append((n, t))
                print(f"  n={n:>7}  вход={cls:<13} t={t:.6f} c")
        print(f"  наклоны log-log: " + ", ".join(
            f"{cls}={log_log_slope(points):.2f}" for cls, points in results[name].items()
        ))
    return results


def plot_results(results: dict, out_dir: Path) -> None:
    """Log-log графики времени от размера входа - отдельный график на алгоритм."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен - графики пропущены "
              "(pip install -r requirements.txt)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, name in zip(axes.flat, results):
        for cls, points in results[name].items():
            ns = [n for n, _ in points]
            ts = [t for _, t in points]
            ax.plot(ns, ts, marker="o",
                    label=f"{cls} (наклон ≈ {log_log_slope(points):.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("размер входа n")
        ax.set_ylabel("время, с")
        ax.set_title(name)
        ax.grid(True, which="both", linewidth=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "lab03_sorts.png"
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
    stability_demo()
    results = run_benchmarks(seed)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()
