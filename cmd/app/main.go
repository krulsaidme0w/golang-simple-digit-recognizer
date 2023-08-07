package main

import (
	n "github.com/krulsaidme0w/golang-simple-neural-network/internal/nnetwork"
)

func main() {
	nn := n.NewNeuralNetwork(4, 2, 3, 2, 0.1)

	inputs := [][]float64{
		{0.1, 0.2, 0.3, 0},
	}
	targets := [][]float64{
		{0.1, 0.2, 0.3},
	}
	nn.Train(inputs, targets)
	nn.Query(inputs[0])
}
