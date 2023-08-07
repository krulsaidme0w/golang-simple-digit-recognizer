package helper

func FindIndexWithMaxElem(arr []float64) int {
	if len(arr) == 0 {
		return -1
	}

	index := 0
	for i, v := range arr {
		if v > arr[index] {
			index = i
		}
	}

	return index
}
