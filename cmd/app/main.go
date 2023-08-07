package main

import (
	"fmt"
	"strconv"
	"time"

	"github.com/krulsaidme0w/golang-simple-neural-network/internal/data"
	n "github.com/krulsaidme0w/golang-simple-neural-network/internal/nnetwork"
	"github.com/krulsaidme0w/golang-simple-neural-network/pkg/helper"
)

func main() {
	inputNodes := 784
	hiddenNodes := 300
	outputNodes := 10
	epochesCount := 7
	learningRate := 0.1

	nn := n.NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes, epochesCount, learningRate)

	trainingDataRaw, err := data.ReadCSVFile("data/mnist_train_100.csv")
	if err != nil {
		return
	}

	trainingDataInputs := make([][]float64, 0, len(trainingDataRaw))
	trainingDataTargets := make([][]float64, 0, len(trainingDataRaw))
	for _, record := range trainingDataRaw {
		inputs, targets := data.PrepareData(record, outputNodes)
		trainingDataInputs = append(trainingDataInputs, inputs)
		trainingDataTargets = append(trainingDataTargets, targets)
	}

	start := time.Now()
	nn.Train(trainingDataInputs, trainingDataTargets)
	elapsed := time.Since(start)
	fmt.Println("train took ", elapsed)

	testDataRaw, err := data.ReadCSVFile("data/mnist_test.csv")
	if err != nil {
		return
	}

	rightAnswers := 0
	wrongAnswers := 0
	for _, record := range testDataRaw {
		input, _ := data.PrepareData(record, outputNodes)

		result := nn.Query(input)
		target, _ := strconv.Atoi(record[0])

		if helper.FindIndexWithMaxElem(result) == target {
			rightAnswers++
			continue
		}

		wrongAnswers++
	}

	fmt.Println("performance ", float64(rightAnswers)/float64(rightAnswers+wrongAnswers))
}
