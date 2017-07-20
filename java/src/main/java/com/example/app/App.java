package com.example.app;


import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class App
{
    public static void main(String[] args) throws Exception {
        System.out.println("nd4h-conv-benchmarks driver");
        if (args.length < 1) {
            System.out.println("Arguments can be:");
            System.out.println("mulsum-test");
            System.out.println("run-nn-test");
            System.exit(0);
        }

        if (args[0].equals("mulsum-test")) {
            if (args.length != 3) {
                System.out.println("Usage: mulsum-test [repeat] [size]");
                System.out.println("Example: mulsum-test 20000 100");
                System.exit(0);
            }
            MulSumBenchmark benchmark = new MulSumBenchmark(Integer.valueOf(args[1]), Integer.valueOf(args[2]));
            benchmark.run();
        } else if (args[0].equals("run-nn-test")) {
            if (args.length != 9) {
                System.out.println("Usage: run-nn-test [repeat] [numFilters] [embeddingH] [embeddingW] [filterW] [padding] [numExtFeats] [numHiddenLayerUnits]");
                System.out.println("Example: run-nn-test 500 100 50 20 5 4 4 201");
                System.exit(0);
            }
            List<Integer> intArgs = Arrays.asList(args).subList(1, 9).stream().map(a -> Integer.valueOf(a)).collect(Collectors.toList());
            NNBenchmark benchmark = new NNBenchmark(intArgs.get(0), intArgs.get(1), intArgs.get(2), intArgs.get(3), intArgs.get(4), intArgs.get(5), intArgs.get(6), intArgs.get(7));
            benchmark.run();
        }
    }
}
