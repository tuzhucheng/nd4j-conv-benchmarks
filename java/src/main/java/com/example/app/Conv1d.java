package com.example.app;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Conv1d {

    public Conv1d(int inChannels, int outChannels, int kernelSize, int stride, int padding, INDArray filters, INDArray biases) {};

    public abstract INDArray forward(INDArray input);

}
