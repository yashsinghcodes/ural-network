package main

import (
    "gonum.org/v1/gonum/mat"    
)


type neuralNetwork struct {
    config neuralNetConfig
    wHidden *mat.Dense
    bHidden *mat.Dense
    wOut    *mat.Dense
    bOut    *mat.Dense
}

type neuralNetConfig struct {
    inputNeurons    int
    outputNeurons   int
    hiddenNeuron    int
    numEpochs       int
    learningRate    float64
}

func newNetwork(config neuralNetConfig) *neuralNetwork{
    return &neuralNetwork{config: config}
}
