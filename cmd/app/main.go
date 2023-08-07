package main

import (
	"fmt"
	"net/http"
	"time"

	d "github.com/krulsaidme0w/golang-simple-neural-network/internal/data"
	h "github.com/krulsaidme0w/golang-simple-neural-network/internal/handler"
	n "github.com/krulsaidme0w/golang-simple-neural-network/internal/nnetwork"
)

const (
	inputNodes   = 784
	hiddenNodes  = 300
	outputNodes  = 10
	epochesCount = 7
	learningRate = 0.1
)

func main() {
	nn := n.NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes, epochesCount, learningRate)

	trainingDataRaw, err := d.ReadCSVFile("data/mnist_train_100.csv")
	if err != nil {
		return
	}

	start := time.Now()

	trainingDataInputs := make([][]float64, 0, len(trainingDataRaw))
	trainingDataTargets := make([][]float64, 0, len(trainingDataRaw))
	for _, record := range trainingDataRaw {
		inputs, targets := d.PrepareData(record, outputNodes)
		trainingDataInputs = append(trainingDataInputs, inputs)
		trainingDataTargets = append(trainingDataTargets, targets)
	}

	nn.Train(trainingDataInputs, trainingDataTargets)

	elapsed := time.Since(start)
	fmt.Println("train took", elapsed)

	handler := h.NewHandler(nn)

	http.HandleFunc("/recognise", handler.RecogniseHandler)
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	http.ListenAndServe(":8080", nil)

	// testDataRaw, err := data.ReadCSVFile("data/mnist_test.csv")
	// if err != nil {
	// 	return
	// }

	// rightAnswers := 0
	// wrongAnswers := 0
	// for _, record := range testDataRaw {
	// 	input, _ := data.PrepareData(record, outputNodes)

	// 	result := nn.Query(input)
	// 	target, _ := strconv.Atoi(record[0])

	// 	if helper.FindIndexWithMaxElem(result) == target {
	// 		rightAnswers++
	// 		continue
	// 	}

	// 	wrongAnswers++
	// }

	// fmt.Println("performance", float64(rightAnswers)/float64(rightAnswers+wrongAnswers))
}
