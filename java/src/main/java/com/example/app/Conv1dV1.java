package com.example.app;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

@Deprecated
public class Conv1dV1 extends Conv1d {

    /**
     * Old slow implementation of Conv1d that should not be used.
     */

    private int outChannels;
    private int kernelSize;
    private int padding;
    private INDArray filters;
    private INDArray biases;
    private INDArray temp;

    public Conv1dV1(int inChannels, int outChannels, int kernelSize, int embeddingW, int stride, int padding, INDArray filters, INDArray biases) {
        super(inChannels, outChannels, kernelSize, stride, padding, filters, biases);
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.filters = filters;
        this.biases = biases;
        this.padding = padding;
        this.temp = Nd4j.createUninitialized(filters.size(1), kernelSize);
    }

    public INDArray forward(INDArray input) {
        // TODO Sanity checking the input
        // TODO add support for non-one stride

        INDArray featureMaps = Nd4j.zeros(outChannels, input.columns() + 2*padding - kernelSize + 1);

        if (padding > 0) {
            INDArray paddedArray = Nd4j.zeros(input.rows(), input.columns() + 2*padding);
            INDArrayIndex index[] = {NDArrayIndex.all(), NDArrayIndex.interval(padding, padding + input.columns())};
            paddedArray.put(index, input);
            input = paddedArray;
        }

        for (int i = 0; i < outChannels; i++) {
            INDArray filter = filters.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all());

            int outputDim = input.shape()[1] - kernelSize + 1;
            INDArray convOutput = Nd4j.createUninitialized(outputDim);

            // loop over each position
            for (int j = 0; j < outputDim; j++) {
                INDArrayIndex interval = NDArrayIndex.interval(j, j + kernelSize);
                INDArray intervalElements = input.get(NDArrayIndex.all(), interval);
                double sum = intervalElements.mul(filter, temp).sumNumber().doubleValue();
                convOutput.putScalar(j, sum);
            }

            // put result for each filter
            convOutput.addi(biases.get(NDArrayIndex.point(i)));
            INDArrayIndex featMapIndex[] = { NDArrayIndex.point(i), NDArrayIndex.all() };
            featureMaps.put(featMapIndex, convOutput);
        }

        Nd4j.getExecutioner().exec(new Tanh(featureMaps));

        return featureMaps;
    }

}
