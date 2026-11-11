#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 5. Бинарное дерево поиска: обходы, вставка и удаление — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-05 и правилами использования генеративного ИИ
(docs/ai-verification.md). Класс BST понадобится повторно в ДЗ 2
(Cases/kim-02-bst-height-experiment.md) — держите его чистым и переносимым.

Запуск: python lab05-bst-starter.py --variant N
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
import traceback
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Бинарное дерево поиска (реализуется вручную, без сторонних библиотек)
# ---------------------------------------------------------------------------


class Node:
    """Узел BST: ключ и ссылки на левое и правое поддеревья."""

    __slots__ = ("key", "left", "right")

    def __init__(self, key: int) -> None:
        self.key = key
        self.left: Node | None = None
        self.right: Node | None = None


class BST:
    """Бинарное дерево поиска без дубликатов ключей.

    Инвариант: для каждого узла все ключи левого поддерева меньше key,
    все ключи правого — больше. Ожидаемая сложность операций: search,
    insert и delete - O(h), где h - высота дерева:
    все три спускаются от корня по одному пути, на каждом шаге опускаясь
    на один уровень, так что число шагов не больше h + 1. При случайных
    ключах h ~ log2(n), но на упорядоченном
    потоке вставок дерево вырождается в цепочку и h = n - 1, то есть
    операции деградируют до O(n).
    """

    def __init__(self) -> None:
        self.root: Node | None = None

    def insert(self, key: int) -> None:
        """Вставка ключа; повторная вставка существующего ключа игнорируется.

        Итеративная версия: рекурсивная вставка имеет
        глубину h и на упорядоченном потоке ключей (дерево-цепочка) падает
        с RecursionError при пределе рекурсии CPython. Итеративный спуск
        этого ограничения не имеет.
        """
        if self.root is None:
            self.root = Node(key)
            return
        parent = None
        current = self.root
        while current is not None:
            if key == current.key:
                return  # повторная вставка существующего ключа
            parent = current
            if key < current.key:
                current = current.left
            else:
                current = current.right
        # current стал None - нашли свободную позицию; parent - её будущий родитель
        if key < parent.key:
            parent.left = Node(key)
        else:
            parent.right = Node(key)

    def search(self, key: int) -> bool:
        """Возвращает True, если ключ есть в дереве."""
        # TODO: спуск от корня со сравнением ключей
        current = self.root
        while current is not None:
            if key == current.key:
                return True
            if key < current.key:
                current = current.left
            else:
                current =current.right
        return False

    def delete(self, key: int) -> None:
        """Удаление ключа. Обработать три случая:

        1) лист — просто отцепить от родителя;
        2) один ребёнок — заменить узел его ребёнком;
        3) два ребёнка — заменить ключ преемником (минимум правого
           поддерева) и удалить преемника из правого поддерева.
        Отсутствующий ключ игнорируется.

        Итеративная версия, по аналогии с insert/search:
        сначала спуском от корня находим удаляемый узел и его родителя.
        Если у узла два ребёнка, ищем преемника - минимум правого поддерева -
        и переносим его ключ в удаляемый узел; тогда реально удалять нужно
        уже сам преемник, остаются только случаи 1 (лист) и 2 (один ребёнок).
        В обоих случаях узел заменяется на своего единственного ребёнка
        (или None) через ссылку родителя - либо self.root, если родителя нет.
        """
        parent = None
        current = self.root
        while current is not None and current.key != key:
            parent = current
            if key < current.key:
                current = current.left
            else:
                current = current.right
        if current is None:
            return  # ключа нет - по условию игнорируем

        if current.left is not None and current.right is not None:
            # случай 3: преемник - минимум правого поддерева (шаг вправо,
            # затем влево до конца); у него нет левого ребёнка
            succ_parent = current
            succ = current.right
            while succ.left is not None:
                succ_parent = succ
                succ = succ.left
            current.key = succ.key
            parent = succ_parent
            current = succ

        # случаи 1 и 2: у current не больше одного ребёнка
        child = current.left if current.left is not None else current.right
        if parent is None:
            self.root = child
        elif parent.left is current:
            parent.left = child
        else:
            parent.right = child

    # -- Обходы --------------------------------------------------------------

    def in_order(self) -> list[int]:
        """Симметричный обход (лево — узел — право): рекурсивно.
        """
        result: list[int] = []

        def visit(node: Node | None) -> None:
            if node is not None:
                visit(node.left)
                result.append(node.key)
                visit(node.right)

        visit(self.root)
        return result

    def pre_order(self) -> list[int]:
        """Прямой обход (узел — лево — право): рекурсивно."""
        result: list[int] = []
        
        def visit(node: Node | None) -> None:
            if node is not None:
                result.append(node.key)
                visit(node.left)
                visit(node.right)
        
        visit(self.root)
        return result

    def post_order(self) -> list[int]:
        """Обратный обход (лево — право — узел): рекурсивно."""
        result: list[int] = []
                
        def visit(node: Node | None) -> None:
            if node is not None:
                visit(node.left)
                visit(node.right)
                result.append(node.key)
                
        visit(self.root)
        return result

    def level_order(self) -> list[int]:
        """Обход в ширину по уровням: итеративно через очередь (deque)."""
        result: list[int] = []
        if self.root is None:
            return result

        queue: deque[Node] = deque([self.root])
        while queue:
            node = queue.popleft()
            result.append(node.key)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        return result

    # -- Характеристики дерева ----------------------------------------------

    def height(self) -> int:
        """Высота дерева: число рёбер на самом длинном пути от корня.

        Пустое дерево имеет высоту -1, дерево из одного узла — 0.
        """

        def node_height(node: Node | None) -> int:
            if node is None:
                return -1
            return 1 + max(node_height(node.left), node_height(node.right))

        return node_height(self.root)

    def size(self) -> int:
        """Число узлов в дереве."""

        def node_size(node: Node | None) -> int:
            if node is None:
                return 0
            return 1 + node_size(node.left) + node_size(node.right)

        return node_size(self.root)


# ---------------------------------------------------------------------------
# 2. Верификация: инвариант BST и сверка с эталоном (docs/ai-verification.md)
# ---------------------------------------------------------------------------


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    """Явная проверка вместо assert: assert-ы исчезают при запуске python -O."""
    if not condition:
        raise SelfCheckError(message)


def build(keys) -> BST:
    """Дерево, полученное последовательной вставкой keys в пустое BST."""
    tree = BST()
    for k in keys:
        tree.insert(k)
    return tree


def expect_search(tree: BST, key: int, present: bool, where: str) -> None:
    """search(key) обязан вернуть именно True или False."""
    got = tree.search(key)
    expect(got is present, f"{where}: search({key}) вернул {got!r}, ожидалось {present}")


TRAVERSALS = ("in_order", "pre_order", "post_order", "level_order")

# Деревья с известными обходами: идеальное дерево 4-2-6-1-3-5-7 (вставка по уровням)
# и пример лекции 5, где у узлов 10 и 14 по одному ребёнку.
KNOWN_TREES = {
    (4, 2, 6, 1, 3, 5, 7): {
        "in_order": [1, 2, 3, 4, 5, 6, 7],
        "pre_order": [4, 2, 1, 3, 6, 5, 7],
        "post_order": [1, 3, 2, 5, 7, 6, 4],
        "level_order": [4, 2, 6, 1, 3, 5, 7],
        "height": 2,
        "size": 7,
    },
    (8, 3, 10, 1, 6, 14, 4, 7, 13): {
        "in_order": [1, 3, 4, 6, 7, 8, 10, 13, 14],
        "pre_order": [8, 3, 1, 6, 4, 7, 10, 14, 13],
        "post_order": [1, 4, 7, 6, 3, 13, 14, 10, 8],
        "level_order": [8, 3, 10, 1, 6, 14, 4, 7, 13],
        "height": 3,
        "size": 9,
    },
}

# Удаление узла с двумя детьми: (ключи, удаляемый ключ, pre_order после замены
# преемником, pre_order после замены предшественником). В последних двух случаях
# у преемника (предшественника) есть свой ребёнок.
TWO_CHILDREN_CASES = [
    ((8, 3, 10, 1, 6, 14, 4, 7, 13), 3, [8, 4, 1, 6, 7, 10, 14, 13], [8, 1, 6, 4, 7, 10, 14, 13]),
    ((8, 3, 10, 1, 6, 14, 4, 7, 13), 6, [8, 3, 1, 7, 4, 10, 14, 13], [8, 3, 1, 4, 7, 10, 14, 13]),
    ((8, 3, 10, 1, 6, 14, 4, 7, 13), 8, [10, 3, 1, 6, 4, 7, 14, 13], [7, 3, 1, 6, 4, 10, 14, 13]),
    ((50, 30, 70, 60, 80, 65), 50, [60, 30, 70, 65, 80], [30, 70, 60, 65, 80]),
    ((50, 30, 70, 60, 80, 65), 70, [50, 30, 80, 60, 65], [50, 30, 65, 60, 80]),
]


def check_insert_search() -> None:
    """Поиск в пустом дереве, вставка, поиск есть/нет, повторная вставка."""
    tree = BST()
    expect_search(tree, 1, False, "пустое дерево")
    tree.insert(5)
    expect_search(tree, 5, True, "дерево из одного узла 5")
    expect_search(tree, 4, False, "дерево из одного узла 5")

    keys = [5, 2, 8, 1, 3, 9, 7]
    tree = build(keys)
    for k in range(11):
        expect_search(tree, k, k in keys, f"дерево из вставок {keys}")
    got = tree.in_order()
    expect(got == sorted(keys), f"in_order() после вставок {keys}: {got}, ожидалось {sorted(keys)}")
    tree.insert(5)
    tree.insert(1)
    got = tree.in_order()
    expect(got == sorted(keys), f"повторная вставка 5 и 1 изменила дерево: in_order() = {got}")


def check_traversals() -> None:
    """Четыре обхода: пустое дерево и деревья с известным ответом."""
    empty = BST()
    for name in TRAVERSALS:
        got = getattr(empty, name)()
        expect(got == [], f"{name}() пустого дерева: {got}, ожидалось []")
    for keys, answer in KNOWN_TREES.items():
        tree = build(keys)
        for name in TRAVERSALS:
            got = getattr(tree, name)()
            expect(got == answer[name],
                   f"{name}() дерева из вставок {list(keys)}: {got}, ожидалось {answer[name]}")


def check_height_size() -> None:
    """Высота в рёбрах (пустое дерево -1, один узел 0) и число узлов."""
    cases = [("пустое дерево", (), -1, 0), ("один узел 5", (5,), 0, 1)]
    cases += [(f"дерево из вставок {list(keys)}", keys, answer["height"], answer["size"])
              for keys, answer in KNOWN_TREES.items()]
    cases.append(("ключи 1, 2, …, 50 по возрастанию (цепочка)", tuple(range(1, 51)), 49, 50))
    for where, keys, height, size in cases:
        tree = build(keys)
        got_h, got_s = tree.height(), tree.size()
        expect(got_h == height, f"{where}: height() = {got_h}, ожидалось {height}")
        expect(got_s == size, f"{where}: size() = {got_s}, ожидалось {size}")


def check_delete() -> None:
    """Пустое дерево, лист, один ребёнок, корень с двумя детьми, отсутствующий и последний ключ."""
    tree = BST()
    tree.delete(1)  # удаление из пустого дерева не должно падать
    got = tree.in_order()
    expect(got == [], f"delete(1) из пустого дерева: in_order() = {got}")

    keys = [5, 2, 8, 1, 3]
    tree = build(keys)
    left = set(keys)
    steps = [(1, "лист 1"), (2, "узел 2 с одним ребёнком"), (5, "корень 5 с двумя детьми"),
             (4, "отсутствующий ключ 4"), (3, "ключ 3"), (8, "последний ключ 8")]
    for k, what in steps:
        tree.delete(k)
        left.discard(k)
        where = f"дерево из вставок {keys}, удаление: {what}"
        got = tree.in_order()
        expect(got == sorted(left), f"{where}: in_order() = {got}, ожидалось {sorted(left)}")
        expect_search(tree, k, False, where)
        for rest in left:
            expect_search(tree, rest, True, where)
    got_h, got_s = tree.height(), tree.size()
    expect(got_h == -1 and got_s == 0,
           f"после удаления всех ключей: height() = {got_h}, size() = {got_s}, ожидалось -1 и 0")


def check_delete_two_children() -> None:
    """Узел с двумя детьми получает ключ преемника (или предшественника): форму дерева
    после удаления однозначно показывает pre_order()."""
    for keys, k, by_successor, by_predecessor in TWO_CHILDREN_CASES:
        tree = build(keys)
        tree.delete(k)
        where = f"delete({k}) в дереве из вставок {list(keys)}"
        rest = sorted(set(keys) - {k})
        got = tree.in_order()
        expect(got == rest, f"{where}: in_order() = {got}, ожидалось {rest}")
        got = tree.pre_order()
        expect(got in (by_successor, by_predecessor),
               f"{where}: pre_order() = {got}; при замене преемником ожидалось {by_successor}, "
               f"предшественником — {by_predecessor}")


def check_invariant() -> None:
    """Инвариант BST через границы (low, high), независимо от in_order().

    Узел допустим, только если его ключ строго между границами, унаследованными
    от всех предков (лекция 5, раздел 8: локальной проверки "левый < узел < правый"
    недостаточно — условие касается всех потомков, а не только детей).
    """

    def valid(node: Node | None, low: float, high: float) -> bool:
        if node is None:
            return True
        if not (low < node.key < high):
            return False
        return valid(node.left, low, node.key) and valid(node.right, node.key, high)

    cases = [(), (5,), (5, 2, 8, 1, 3, 9, 7)] + list(KNOWN_TREES.keys())
    for keys in cases:
        tree = build(keys)
        where = f"дерево из вставок {list(keys)}"
        expect(valid(tree.root, float("-inf"), float("inf")), f"{where}: инвариант BST нарушен")


def check_against_reference() -> None:
    """Случайные команды insert/delete: после каждой in_order() совпадает с sorted(эталона)."""
    rng = random.Random(0)
    for run in range(1, 101):
        tree = BST()
        etalon: set[int] = set()
        for step in range(1, rng.randint(1, 80) + 1):
            k = rng.randint(-30, 30)
            if rng.random() < 0.6:
                tree.insert(k)
                etalon.add(k)
                op = f"insert({k})"
            else:
                tree.delete(k)
                etalon.discard(k)
                op = f"delete({k})"
            # Инвариант BST: in-order даёт строго возрастающую последовательность ключей
            got = tree.in_order()
            expect(got == sorted(etalon), f"прогон {run}, команда №{step} {op}: "
                                          f"in_order() = {got}, эталон {sorted(etalon)}")
        where = f"прогон {run}, итоговое дерево"
        for k in range(-30, 31):
            expect_search(tree, k, k in etalon, where)
        n = len(etalon)
        got = tree.size()
        expect(got == n, f"{where}: size() = {got}, эталон {n}")
        for name in TRAVERSALS[1:]:
            got = getattr(tree, name)()
            expect(sorted(got) == sorted(etalon), f"{where}: {name}() содержит не те ключи: {got}")
        got = tree.height()
        low, high = (-1, -1) if n == 0 else (n.bit_length() - 1, n - 1)
        expect(low <= got <= high,
               f"{where}: height() = {got} при {n} узлах, возможно от {low} до {high}")


def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""
    checks = (
        ("вставка и поиск", check_insert_search),
        ("обходы", check_traversals),
        ("высота и размер", check_height_size),
        ("удаление", check_delete),
        ("удаление узла с двумя детьми", check_delete_two_children),
        ("инвариант BST (границы low/high)", check_invariant),
        ("сверка с эталоном set/sorted", check_against_reference),
    )
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
    # TODO: добавить собственные проверки (например, инвариант BST после каждой
    #       операции рекурсивной проверкой с границами) и описать их в отчёте.
    if failed:
        raise SystemExit(f"self_check: не пройдено разделов — {failed}; "
                         f"замеры выполняются только после всех проверок.")
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 3. Бенчмарк: поиск в BST против поиска в списке (docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
QUERIES = 1_000  # число поисков на один замер


def bench(fn) -> float:
    """Медиана времени выполнения fn() по REPEATS запускам, с прогревом."""
    fn()  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_benchmarks(seed: int) -> list[tuple[int, int, float, float]]:
    """Сравнение времени QUERIES поисков в BST и в списке (оператор in).

    Возвращает список (n, h(BST), t_bst, t_list) — используется plot_results().
    """
    rng = random.Random(seed)
    results: list[tuple[int, int, float, float]] = []
    print(f"{'n':>8} {'h(BST)':>7} {'t_bst, c':>10} {'t_list, c':>10}")
    for n in SIZES:
        keys = rng.sample(range(10 * n), n)   # случайные ключи без повторов
        queries = [rng.randint(0, 10 * n) for _ in range(QUERIES)]
        tree = BST()
        for k in keys:
            tree.insert(k)

        def search_bst() -> None:
            for q in queries:
                tree.search(q)

        def search_list() -> None:
            for q in queries:
                q in keys  # линейный поиск для сравнения

        h = tree.height()
        t_bst = bench(search_bst)
        t_list = bench(search_list)
        print(f"{n:>8} {h:>7} {t_bst:>10.6f} {t_list:>10.6f}")
        results.append((n, h, t_bst, t_list))
    return results
    # Деградацию BST на упорядоченных ключах и гарантию высоты AVL-дерева
    # (не более 1,44*log2(n)) нужно объяснить в отчёте (КИМ-05, п.6) — мостик к ДЗ 2.


def plot_results(results: list[tuple[int, int, float, float]], out_dir: Path) -> None:
    """Графики: (1) t_bst и t_list от n в осях log-log; (2) h(BST) от n против
    log2(n) — показывает, насколько высота случайного дерева близка к теоретическому
    минимуму."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib не установлен — графики пропущены "
              "(pip install -r requirements.txt)")
        return

    ns = [n for n, _, _, _ in results]
    hs = [h for _, h, _, _ in results]
    t_bsts = [t for _, _, t, _ in results]
    t_lists = [t for _, _, _, t in results]

    fig, (ax_time, ax_height) = plt.subplots(1, 2, figsize=(12, 5))

    ax_time.plot(ns, t_bsts, marker="o", label="t_bst (поиск в BST)")
    ax_time.plot(ns, t_lists, marker="o", label="t_list (поиск в списке)")
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("размер входа n")
    ax_time.set_ylabel(f"время {QUERIES} поисков, с")
    ax_time.set_title("Поиск: BST против списка")
    ax_time.grid(True, which="both", linewidth=0.3)
    ax_time.legend(fontsize=8)

    ax_height.plot(ns, hs, marker="o", label="h(BST), случайные ключи")
    ax_height.plot(ns, [math.floor(math.log2(n)) for n in ns], marker="x",
                    linestyle="--", label="floor(log2 n) — теоретический минимум")
    ax_height.set_xscale("log")
    ax_height.set_xlabel("размер входа n")
    ax_height.set_ylabel("высота дерева h, рёбра")
    ax_height.set_title("Высота BST на случайных ключах")
    ax_height.grid(True, which="both", linewidth=0.3)
    ax_height.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "lab05_bst.png"
    fig.savefig(path, dpi=150)
    print(f"\nГрафик сохранён: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    ap.add_argument("--out", type=Path, default=Path.cwd(),
                    help="каталог для графика (по умолчанию текущий)")
    args = ap.parse_args()
    seed = 30 + args.variant  # seed по варианту — см. docs/reproducibility.md
    random.seed(seed)
    self_check()
    results = run_benchmarks(seed)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()
