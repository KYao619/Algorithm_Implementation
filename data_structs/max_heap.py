# -*- coding: utf-8 -*-


class MaxHeap(object):
    def __init__(self, max_size, fn):
        self.max_size = max_size
        self.fn = fn
        self._items = [None] * max_size
        self.size = 0

    def __str__(self):
        item_values = str([self.fn(self.items[i]) for i in range(self.size)])
        return "Size: %d\nMAx Size: %d\nItems: %s\nItem_values: %s\n" % (self.size, self.max_size, self.items, item_values)

    @property
    def items(self):
        return self._items[:self.size]

    @property
    def full(self):
        return self.size == self.max_size

    def value(self, index):
        item = self.items[index]
        if item is None:
            res = float('-inf')
        else:
            res = self.fn(item)
        return res

    def add(self, item):
        if self.full:
            if self.fn(item) < self.value(0):
                self._items[0] = item
                self._shift_down(0)
        else:
            self._items[self.size] = item
            self.size += 1
            self._shift_up(self.size - 1)

    def pop(self):
        assert self.size > 0, "Cannot pop item! The MaxHeap is empty!"
        ret = self._items[0]
        self._items[0] = self.items[self.size - 1]
        self._items[self.size - 1] = None
        self.size -= 1
        self._shift_down(0)
        return ret

    def _shift_up(self, index):
        assert index < self.size, "The input index is invalid! The index must be less than the size of MaxHeap!"
        parent = (index - 1) // 2
        while parent >= 0 and self.value(parent) < self.value(index):
            self._items[parent], self._items[index] = self._items[index], self._items[parent]
            index = parent
            parent = (index - 1) // 2

    def _shift_down(self, index):
        assert index < self.size, "The input index is invalid! The index must be less than the size of MaxHeap!"
        child = index * 2 + 1
        while child < self.size:
            if child + 1 < self.size and self.value(child + 1) > self.value(child):
                child += 1
            if self.value(index) < self.value(child):
                self._items[index], self._items[child] = self._items[child], self._items[index]
                index, child = child, child * 2 + 1
            else:
                break

    def _is_valid(self):
        ret = []
        for i in range(1, self.size):
            parent = (i - 1) // 2
            ret.append(self.value(parent) >= self.value(i))
        return all(ret)


