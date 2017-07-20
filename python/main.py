#!/usr/bin/env python

from sys import argv, exit

from mulsum_benchmark import MulSumBenchmark


if __name__ == '__main__':
    print('numpy benchmark driver')
    if len(argv) < 2:
        print('Arguments can be:')
        print('mulsum-test')
        exit(0)

    if argv[1] == 'mulsum-test':
        if len(argv) != 4:
            print('Usage: mulsum-test [repeat] [size]')
            exit(0)
        benchmark = MulSumBenchmark(int(argv[2]), int(argv[3]))
        benchmark.run()
