package nnetwork

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type NeuralNetwork struct {
	InputNodes     int
	HiddenNodes    int
	OutputNodes    int
	EpochesCount   int
	LearningRate   float64
	WeightsInput   *mat.Dense
	WeightsOutput  *mat.Dense
	ActivationFunc func(float64) float64
}

func NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes, epochesCount int, learningRate float64) *NeuralNetwork {
	return &NeuralNetwork{
		InputNodes:     inputNodes,
		HiddenNodes:    hiddenNodes,
		OutputNodes:    outputNodes,
		EpochesCount:   epochesCount,
		LearningRate:   learningRate,
		WeightsInput:   randomMatrix(hiddenNodes, inputNodes),
		WeightsOutput:  randomMatrix(outputNodes, hiddenNodes),
		ActivationFunc: sigmoid,
	}
}

func (nn *NeuralNetwork) Query(inputArr []float64) []float64 {
	input := mat.NewDense(nn.InputNodes, 1, inputArr)

	hiddenOutput := mat.NewDense(nn.HiddenNodes, 1, nil)
	hiddenOutput.Mul(nn.WeightsInput, input)
	applyFunc(hiddenOutput, nn.ActivationFunc)

	output := mat.NewDense(nn.OutputNodes, 1, nil)
	output.Mul(nn.WeightsOutput, hiddenOutput)
	applyFunc(output, nn.ActivationFunc)

	return output.RawMatrix().Data
}

func (nn *NeuralNetwork) Train(inputArr, targetArr [][]float64) {
	if len(inputArr) != len(targetArr) {
		panic("input and target len must be the same")
	}

	for i := 0; i < nn.EpochesCount; i++ {
		for j := 0; j < len(inputArr); j++ {
			input := mat.NewDense(nn.InputNodes, 1, inputArr[j])
			target := mat.NewDense(nn.OutputNodes, 1, targetArr[j])

			hiddenOutput := mat.NewDense(nn.HiddenNodes, 1, nil)
			hiddenOutput.Mul(nn.WeightsInput, input)
			applyFunc(hiddenOutput, nn.ActivationFunc)

			output := mat.NewDense(nn.OutputNodes, 1, nil)
			output.Mul(nn.WeightsOutput, hiddenOutput)
			applyFunc(output, nn.ActivationFunc)

			outputError := mat.NewDense(nn.OutputNodes, 1, nil)
			outputError.Sub(target, output)

			hiddenError := mat.NewDense(nn.HiddenNodes, 1, nil)
			hiddenError.Mul(nn.WeightsOutput.T(), outputError)

			multiplyElementByElement(outputError, output)
			applyFunc(output, func(x float64) float64 {
				return 1. - x
			})
			multiplyElementByElement(outputError, output)
			m := mat.NewDense(outputError.RawMatrix().Rows, hiddenOutput.RawMatrix().Rows, nil)
			m.Mul(outputError, hiddenOutput.T())
			m.Scale(nn.LearningRate, m)
			nn.WeightsOutput.Add(nn.WeightsOutput, m)

			multiplyElementByElement(hiddenError, hiddenOutput)
			applyFunc(hiddenOutput, func(x float64) float64 {
				return 1. - x
			})
			multiplyElementByElement(hiddenError, hiddenOutput)
			m = mat.NewDense(hiddenError.RawMatrix().Rows, input.RawMatrix().Rows, nil)
			m.Mul(hiddenError, input.T())
			m.Scale(nn.LearningRate, m)
			nn.WeightsInput.Add(nn.WeightsInput, m)
		}
	}
}

func multiplyElementByElement(a *mat.Dense, b *mat.Dense) {
	rows, cols := a.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			a.Set(i, j, a.At(i, j)*b.At(i, j))
		}
	}
}

func applyFunc(m *mat.Dense, f func(float64) float64) {
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Set(i, j, f(m.At(i, j)))
		}
	}
}

func randomMatrix(rows, cols int) *mat.Dense {
	normalDist := distuv.Normal{
		Mu:    0.0,
		Sigma: math.Pow(float64(cols), -0.5),
	}

	weights := make([]float64, rows*cols)
	for i := range weights {
		weights[i] = normalDist.Rand()
	}
	return mat.NewDense(rows, cols, weights)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
