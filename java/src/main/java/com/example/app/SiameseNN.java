package com.example.app;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LogSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SiameseNN {

    private int numFilters;
    private int padding;
    private int hiddenLayerInputUnits;
    private INDArray s1ConvFilters;
    private INDArray s1ConvFilterBiases;
    private INDArray s2ConvFilters;
    private INDArray s2ConvFilterBiases;
    private INDArray hiddenLayerWeights;
    private INDArray hiddenLayerBiases;
    private INDArray softmaxLayerWeights;
    private INDArray softmaxLayerBiases;

    public SiameseNN(int numFilters, int filterH, int filterW, int padding, int numExtFeats, int hiddenLayerUnits) {
        this.numFilters = numFilters;
        this.padding = padding;
        hiddenLayerInputUnits = 2*numFilters + numExtFeats;

        int[] filterDims = {numFilters, filterH, filterW};
        s1ConvFilters = Nd4j.randn(filterDims);
        s1ConvFilterBiases = Nd4j.randn(1, numFilters);
        s2ConvFilters = Nd4j.randn(filterDims);
        s2ConvFilterBiases = Nd4j.randn(1, numFilters);
        hiddenLayerWeights = Nd4j.randn(hiddenLayerInputUnits, hiddenLayerUnits);
        hiddenLayerBiases = Nd4j.randn(1, hiddenLayerUnits);
        softmaxLayerWeights = Nd4j.randn(hiddenLayerUnits, 2);
        softmaxLayerBiases = Nd4j.randn(1, 2);
    }

    // For benchmarking purposes - we don't need getters and setters for weights of this network

    private INDArray getConvFeatureMaps(INDArray input, INDArray filters, INDArray biases) {
        int kernelWidth = 5;
        int padding = 4;
        INDArray questionConvFeatureMaps = Nd4j.zeros(numFilters, input.columns() + 2*padding - kernelWidth + 1);
        Conv1dV1 conv = new Conv1dV1(1, 1, kernelWidth, 1, 4);
        for (int i = 0; i < filters.size(0); i++) {
            INDArray filter = filters.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());
            INDArray convOutput = conv.forward(input, filter);
            convOutput.addi(biases.get(NDArrayIndex.point(i)));
            INDArrayIndex featMapIndex[] = { NDArrayIndex.point(i), NDArrayIndex.all() };
            questionConvFeatureMaps.put(featMapIndex, convOutput);
        }
        Nd4j.getExecutioner().exec(new Tanh(questionConvFeatureMaps));
        return questionConvFeatureMaps;
    }

    public INDArray forward(INDArray s1, INDArray s2, INDArray externalFeatures) {
        // Convolution
        INDArray questionConvFeatureMaps = getConvFeatureMaps(s1, s1ConvFilters, s1ConvFilterBiases);
        INDArray answerConvFeatureMaps = getConvFeatureMaps(s2, s2ConvFilters, s2ConvFilterBiases);

        // Pooling
        INDArray questionPooled = Nd4j.max(questionConvFeatureMaps, 1);
        INDArray answerPooled = Nd4j.max(answerConvFeatureMaps, 1);

        // Join layer
        INDArray joinLayer = Nd4j.zeros(hiddenLayerInputUnits);
        INDArrayIndex questionIndex[] = { NDArrayIndex.interval(0, numFilters) };
        INDArrayIndex answerIndex[] = { NDArrayIndex.interval(numFilters, 2*numFilters) };
        INDArrayIndex extFeatsIndex[] = { NDArrayIndex.interval(2*numFilters, hiddenLayerInputUnits) };
        joinLayer.put(questionIndex, questionPooled);
        joinLayer.put(answerIndex, answerPooled);
        joinLayer.put(extFeatsIndex, externalFeatures);

        // Hidden layer
        INDArray hiddenLayer = joinLayer.mmul(hiddenLayerWeights).addi(hiddenLayerBiases);
        Nd4j.getExecutioner().exec(new Tanh(hiddenLayer));

        // Softmax
        INDArray finalLayer = hiddenLayer.mmul(softmaxLayerWeights).addi(softmaxLayerBiases);
        Nd4j.getExecutioner().exec(new LogSoftMax(finalLayer));
        return finalLayer;
    }
}
