package handler

import "github.com/krulsaidme0w/golang-simple-neural-network/internal/nnetwork"

type Hander struct {
	nn *nnetwork.NeuralNetwork
}

func NewHandler(nn *nnetwork.NeuralNetwork) *Hander {
	return &Hander{
		nn: nn,
	}
}
