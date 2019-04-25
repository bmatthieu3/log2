import pytest
import numpy as np

from log2 import naive, lookup_table, fast_lookup_table

def test_naive(benchmark):
    inputs = np.random.randint(low=0, high=(1<<32), size=100000, dtype=np.uint32)
    expected = np.floor(np.log2(inputs))

    v_naive = np.vectorize(naive)
    result = benchmark(v_naive, inputs)
    assert (result == expected).all()

def test_lookup_table(benchmark):
    inputs = np.random.randint(low=0, high=(1<<32), size=100000, dtype=np.uint32)
    expected = np.floor(np.log2(inputs))
    
    v_lookup_table = np.vectorize(lookup_table)
    result = benchmark(v_lookup_table, inputs)
    assert (result == expected).all()

def test_fast_lookup_table(benchmark):
    inputs = np.random.randint(low=0, high=(1<<32), size=100000, dtype=np.uint32)
    expected = np.floor(np.log2(inputs))
    
    v_fast_lookup_table = np.vectorize(fast_lookup_table)
    result = benchmark(v_fast_lookup_table, inputs)
    assert (result == expected).all()