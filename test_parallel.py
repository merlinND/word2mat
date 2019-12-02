"""
Quick tests for parallel execution
"""

import os

# Multiprocessing library for more powerful parallelization
# https://github.com/uqfoundation/multiprocess
import multiprocess

if __name__ == '__main__':
    pool = multiprocess.Pool(os.cpu_count())

    f = lambda x: x ** 2

    inputs = list(range(100))

    results = pool.map(f, inputs)

    print(inputs)
    print(results)
