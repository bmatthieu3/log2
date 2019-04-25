import pytest
import numpy as np

from ..log2 import naive, lookup_table, fast_lookup_table, fast_lookup_table_c, fast_lookup_table_c_ufunc

def test_naive(benchmark):
    inputs = np.random.randint(low=1, high=(1<<32), size=1000, dtype=np.uint32)
    expected = np.floor(np.log2(inputs))

    v_naive = np.vectorize(naive)
    result = benchmark(v_naive, inputs)
    assert (result == expected).all()

def test_lookup_table(benchmark):
    inputs = np.random.randint(low=1, high=(1<<32), size=100, dtype=np.uint32)
    expected = np.floor(np.log2(inputs))
    
    v_lookup_table = np.vectorize(lookup_table)
    result = benchmark(v_lookup_table, inputs)
    assert (result == expected).all()

def test_fast_lookup_table(benchmark):
    inputs = np.random.randint(low=1, high=(1<<32), size=100, dtype=np.uint32)
    expected = np.floor(np.log2(inputs))
    
    v_fast_lookup_table = np.vectorize(fast_lookup_table)
    result = benchmark(v_fast_lookup_table, inputs)
    assert (result == expected).all()

def test_fast_lookup_table_c(benchmark):
    inputs = np.random.randint(low=1, high=(1<<32), size=100, dtype=np.uint32)
    expected = np.floor(np.log2(inputs))
    
    v_fast_lookup_table = np.vectorize(fast_lookup_table_c)
    result = benchmark(v_fast_lookup_table, inputs)
    assert (result == expected).all()

def test_fast_lookup_table_c_ufunc(benchmark):
    inputs = np.random.randint(low=1, high=(1<<32), size=100, dtype=np.uint32)
    expected = np.floor(np.log2(inputs))
    
    result = benchmark(fast_lookup_table_c_ufunc, inputs)
    assert (result == expected).all()