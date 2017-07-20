import time

import numpy as np


class MulSumBenchmark(object):

    def __init__(self, repeat, height, width):
        self.repeat = repeat
        self.matrices_a = [np.random.randn(height, width) for _ in range(repeat)]
        self.matrices_b = [np.random.randn(height, width) for _ in range(repeat)]
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

