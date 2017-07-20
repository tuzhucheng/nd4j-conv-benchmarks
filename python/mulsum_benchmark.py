import time

import numpy as np


class MulSumBenchmark(object):

    def __init__(self, repeat, size):
        self.repeat = repeat
        self.size = size
        self.matrices_a = [np.random.randn(size, size) for _ in range(repeat)]
        self.matrices_b = [np.random.randn(size, size) for _ in range(repeat)]
        self.results = []

    def run(self):
        start = time.time()
        count = 0

        for i in range(self.repeat):
            result = np.multiply(self.matrices_a[i], self.matrices_b[i]).sum()
            self.results.append(result)
            count += 1

        end = time.time()
        elapsed = end - start
        print('Elapsed time (s):', elapsed)
        print('Operations:', count)
        print('Operations / second:', count / elapsed)
        return self.results

