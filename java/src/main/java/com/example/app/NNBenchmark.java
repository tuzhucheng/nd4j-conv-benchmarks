package com.example.app;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

public class NNBenchmark {
    private int repeat;
    private ArrayList<INDArray> matricesA;
    private ArrayList<INDArray> matricesB;
    private ArrayList<INDArray> extFeats;
    private SiameseNN nnModel;
    private ArrayList<INDArray> results;

    public NNBenchmark(int repeat, int numFilters, int embeddingH, int embeddingW, int filterW, int padding, int numExtFeats, int numHiddenLayerUnits) {
        this.repeat = repeat;

        matricesA = new ArrayList<>(repeat);
        matricesB = new ArrayList<>(repeat);
        extFeats = new ArrayList<>(repeat);
        for (int i = 0; i < repeat; i++) {
            // Gaussian distribution, mean 0, std. dev. 1
            INDArray a = Nd4j.randn(embeddingH, embeddingW);
            INDArray b = Nd4j.randn(embeddingH, embeddingW);
            INDArray ext = Nd4j.randn(1, numExtFeats);
            matricesA.add(a);
            matricesB.add(b);
            extFeats.add(ext);
        }
        nnModel = new SiameseNN(numFilters, embeddingH, filterW, padding, numExtFeats, numHiddenLayerUnits);
        results = new ArrayList<>();
    }

    public ArrayList<INDArray> run() {
        long start = System.nanoTime();
        int count = 0;

        for (int i = 0; i < repeat; i++) {
            INDArray result = nnModel.forward(matricesA.get(i), matricesB.get(i), extFeats.get(i));
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
