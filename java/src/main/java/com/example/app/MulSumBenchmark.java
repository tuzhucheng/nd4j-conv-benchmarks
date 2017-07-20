package com.example.app;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

public class MulSumBenchmark {

    private int repeat;
    private int size;
    private ArrayList<INDArray> matricesA;
    private ArrayList<INDArray> matricesB;
    private ArrayList<Double> results;

    public MulSumBenchmark(int repeat, int size) {
        this.repeat = repeat;
        this.size = size;
        matricesA = new ArrayList<>(repeat);
        matricesB = new ArrayList<>(repeat);
        for (int i = 0; i < repeat; i++) {
            // Gaussian distribution, mean 0, std. dev. 1
            INDArray a = Nd4j.randn(size, size);
            INDArray b = Nd4j.randn(size, size);
            matricesA.add(a);
            matricesB.add(b);
        }
        results = new ArrayList<>();
    }

    public ArrayList<Double> run() {
        long start = System.nanoTime();
        int count = 0;

        for (int i = 0; i < repeat; i++) {
            double result = matricesA.get(i).mul(matricesB.get(i)).sumNumber().doubleValue();
            results.add(result);
            count++;
        }

        long end = System.nanoTime();
        long totalElapsedSeconds = TimeUnit.NANOSECONDS.toSeconds(end - start);
        System.out.println("Elapsed time (s): " + totalElapsedSeconds);
        System.out.println("Operations: " + count);
        System.out.println("Operations / second: " + count * 1.0 / totalElapsedSeconds);
        return results;
    }

}
