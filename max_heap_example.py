# -*- coding: utf-8 -*-


import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
import sys
sys.path.append(os.path.abspath(".."))

from copy import copy
from max_heap import MaxHeap
from example_generator import gen_data
from time import time
from random import randint


def brutal_search(nums, k):
    res = []
    index = set()
    key = None
    for _ in range(k):
        val = float('inf')
        for i, num in enumerate(nums):
            if num < val and i not in index:
                key = i
                val = num
        index.add(key)
        res.append(val)
    return res


def main():
    print("Testing MaxHeap...")
    test_times = 10
    runtime_1 = runtime_2 = 0
    for _ in range(test_times):
        low = 0
        high = 1000
        n_rows = 10000
        k = 100
        nums = gen_data(low, high, n_rows)

        heap = MaxHeap(k, lambda x: x)
        start = time()
        for num in nums:
            heap.add(num)
        res1 = copy(heap.items)
        runtime_1 += time() - start

        start = time()
        res2 = brutal_search(nums, k)
        runtime_2 += time() - start

        res1.sort()
        assert res1 == res2, "target: %s\nk: %d\nresult1: %s\nresult2: %s\n" % (
            str(nums), k, str(res1), str(res2)
        )
    print("%d tests passed!" % test_times)
    print("MaxHeap Search %.2f s" % runtime_1)
    print("Brutal Search %.2f s" % runtime_2)


if __name__ == "__main__":
    main()