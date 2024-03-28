from typing import Iterator, Iterable
 
import numpy as np
 
 
class SizedFIFO(Iterable):
    """Fixed-size First-In, First-Out (FIFO) queue.
 
    Properties:
        queue: Returns a copy of the queue.
 
    Methods:
        insert(value): Pushes a value at the start of the queue and removes the
            value at the end of the queue.
    """
 
    def __init__(self, items: list[int or float]):
        self._queue = np.array(items)
 
    def __len__(self) -> int:
        return len(self._queue)
 
    def __iter__(self) -> Iterator:
        return iter(self._queue)
 
    def __repr__(self) -> str:
        return f"{self._queue}"
 
    def __getitem__(self, index: int) -> int or float:
        return self._queue[index]
 
    def __setitem__(self, key: int, value: int or float):
        self._queue[key] = value
 
    def __sum__(self) -> int or float:
        return np.sum(self._queue).item()
 
    def copy(self):
        return SizedFIFO(self._queue.copy())
 
    @property
    def queue(self) -> np.array:
        return self._queue.copy()
 
    def insert(self, value: int or float) -> int or float:
        if len(self._queue) == 0:
            return value
        popped = self._queue[-1]
        self._queue = np.roll(self._queue, 1)
        self._queue[0] = value
        return popped