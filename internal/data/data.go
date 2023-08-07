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

func PrepareMatrix(m [][]int) []float64 {
	arr := make([]float64, 0, len(m)*len(m))

	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[i]); j++ {
			arr = append(arr, (float64(m[i][j])/255.0*0.98)+0.01)
		}
	}

	return arr
}
