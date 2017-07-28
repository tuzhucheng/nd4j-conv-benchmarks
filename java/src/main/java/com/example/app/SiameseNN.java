package com.example.app;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LogSoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SiameseNN {

    private int numFilters;
    private int hiddenLayerInputUnits;
    private INDArray s1ConvFilters;
    private INDArray s1ConvFilterBiases;
    private INDArray s2ConvFilters;
    private INDArray s2ConvFilterBiases;
    private INDArray hiddenLayerWeights;
    private INDArray hiddenLayerBiases;
    private INDArray softmaxLayerWeights;
    private INDArray softmaxLayerBiases;
    private Conv1d questionConv;
    private Conv1d answerConv;

    public SiameseNN(int numFilters, int filterH, int filterW, int embeddingW, int padding, int numExtFeats, int hiddenLayerUnits) {
        this.numFilters = numFilters;
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

        questionConv = new Conv1dV2(filterH, numFilters, filterW, embeddingW, 1, padding, s1ConvFilters, s1ConvFilterBiases);
        answerConv = new Conv1dV2(filterH, numFilters, filterW, embeddingW, 1, padding, s2ConvFilters, s2ConvFilterBiases);
    }

    public INDArray forward(INDArray s1, INDArray s2, INDArray externalFeatures) {
        // Convolution
        INDArray questionConvFeatureMaps = questionConv.forward(s1);
        INDArray answerConvFeatureMaps = answerConv.forward(s2);

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
