package com.example.app;


public class App
{
    public static void main(String[] args) throws Exception {
        System.out.println("nd4h-conv-benchmarks driver");
        if (args.length < 1) {
            System.out.println("Arguments can be:");
            System.out.println("run-conv-test");
            System.out.println("mulsum-test");
            System.exit(0);
        }

        if (args[0].equals("run-conv-test")) {
            throw new UnsupportedOperationException("run-conv-test is not yet implemented.");
        } else if (args[0].equals("mulsum-test")) {
            if (args.length != 3) {
                System.out.println("Usage: mulsum-test [repeat] [size]");
                System.exit(0);
            }
            MulSumBenchmark benchmark = new MulSumBenchmark(Integer.valueOf(args[1]), Integer.valueOf(args[2]));
            benchmark.run();
        }
    }
}
