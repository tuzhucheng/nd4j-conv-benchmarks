package com.example.app;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Conv1dV2 extends Conv1d {

    private int inChannels;
    private int padding;
    private MultiLayerNetwork model;

    public Conv1dV2(int inChannels, int outChannels, int kernelSize, int embeddingW, int stride, int padding, INDArray filters, INDArray biases) {
        super(inChannels, outChannels, kernelSize, stride, padding, filters, biases);
        this.inChannels = inChannels;
        this.padding = padding;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new ConvolutionLayer.Builder(inChannels, kernelSize)
                        .nIn(1)
                        .stride(stride, stride)
                        .nOut(outChannels)
                        .padding(0, this.padding)
                        .activation(Activation.TANH)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);

        INDArray initialLayer = Nd4j.ones(1, 1, inChannels, embeddingW);
        model.initializeLayers(initialLayer);

//        For performance tests using various dimensions, do not set params
//        List<INDArray> params = new ArrayList<>();
//        params.add(biases);
//        params.add(filters);
//        model.setParams(Nd4j.toFlattened('f', params));
    }

    public INDArray forward(INDArray input) {
        model.initializeLayers(input);
        int[] shape = {1, 1, input.size(0), input.size(1)};
        INDArray reshapedInput = Nd4j.createUninitialized(shape);
        for (int j = 0; j < input.size(0); j++) {
            INDArrayIndex[] index = {NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(j), NDArrayIndex.all()};
            INDArrayIndex[] inputIndex = {NDArrayIndex.point(j), NDArrayIndex.all()};
            reshapedInput.put(index, input.get(inputIndex));
        }
        INDArray output = model.output(reshapedInput, false);
        output = output.reshape(output.size(1), output.size(3));
        return output;
    }
}
