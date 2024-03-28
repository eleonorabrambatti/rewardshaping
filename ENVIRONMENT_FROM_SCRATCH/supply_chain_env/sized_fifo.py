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
    

    """ Questa classe SizedFIFO rappresenta una coda FIFO (First-In, First-Out) di dimensione fissa. Una coda FIFO è una struttura dati che gestisce elementi in ordine di inserimento, dove l'elemento inserito per primo è anche il primo ad essere rimosso (come una fila in un supermercato).
Ecco cosa fa ciascun metodo e attributo della classe:
__init__(self, items: list[int or float]): Il costruttore accetta una lista di elementi (interi o float) e inizializza la coda FIFO con questi elementi.
__len__(self) -> int: Restituisce la lunghezza della coda FIFO.
__iter__(self) -> Iterator: Restituisce un iteratore per iterare attraverso gli elementi della coda FIFO.
__repr__(self) -> str: Restituisce una rappresentazione stringa della coda FIFO.
__getitem__(self, index: int) -> int or float: Permette l'accesso agli elementi della coda FIFO tramite l'operatore di indicizzazione.
__setitem__(self, key: int, value: int or float): Permette di impostare il valore di un elemento della coda FIFO tramite l'operatore di indicizzazione.
__sum__(self) -> int or float: Calcola la somma di tutti gli elementi nella coda FIFO.
copy(self): Restituisce una copia della coda FIFO.
queue(self) -> np.array: Proprietà che restituisce una copia dell'array interno che rappresenta la coda FIFO.
insert(self, value: int or float) -> int or float: Inserisce un valore all'inizio della coda FIFO e rimuove il valore alla fine della coda FIFO se la coda è piena.
In sintesi, questa classe fornisce una struttura dati FIFO di dimensione fissa con funzionalità per inserire, accedere, modificare e ottenere informazioni sulla coda. """