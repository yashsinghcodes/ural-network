package main

import (
	"math"
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

func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
    return sigmoid(x) * (1.0 - sigmoid(x))
}

func (nn *neuralNetwork) train(x, y *mat.Dense) error {
    
    randSource := rand.NewSource(time.Now().UnoxNano())
    randGen := rand.New(randSource)

    wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeuron, nil)
    bHidden := mat.NewDense(1, nn.config.hiddenNeuron, nil)
    wOut := mat.NewDense(nn.config.hiddenNeuron, nn.config.outputNeurons, nil)
    bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

    wHiddenRaw := wHidden.RawMatrix().Data
    bHiddenRaw := bHidden.RawMatrix().Data
    wOutRaw := wOut.RawMatrix().Data
    bOutRaw := bOut.RawMatrix().Data

    for _, param := range [][]float64{
        wHiddenRaw,
        bHiddenRaw,
        wOutRaw,
        bOutRaw,
    } {
        for i := range param {
            param[i] = randGen.Float64()
        }
    }

    
    output := new(mat.Dense)
    
    if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
        return err
    }

    nn.wHidden = wHidden
    nn.bHidden = bHidden
    nn.wOut = wOut

    return nil
}


// TODO: Write a backpropagate function for the NN.
