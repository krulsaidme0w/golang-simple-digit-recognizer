package data

import (
	"bufio"
	"encoding/csv"
	"os"
	"strconv"
)

func ReadCSVFile(filename string) ([][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReader(file))

	var data [][]string
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		data = append(data, record)
	}

	return data, nil
}

func PrepareData(record []string, outputNodes int) ([]float64, []float64) {
	inputs := make([]float64, len(record)-1)
	for i, valStr := range record[1:] {
		val, _ := strconv.ParseFloat(valStr, 64)
		inputs[i] = (val / 255.0 * 0.98) + 0.01
	}

	targets := make([]float64, outputNodes)
	targetIdx, _ := strconv.Atoi(record[0])
	targets[targetIdx] = 0.99

	return inputs, targets
}
