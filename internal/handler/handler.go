package handler

import (
	"encoding/json"
	"net/http"

	"github.com/krulsaidme0w/golang-simple-neural-network/internal/data"
	"github.com/krulsaidme0w/golang-simple-neural-network/internal/nnetwork"
	"github.com/krulsaidme0w/golang-simple-neural-network/pkg/helper"
)

type Hander struct {
	nn *nnetwork.NeuralNetwork
}

func NewHandler(nn *nnetwork.NeuralNetwork) *Hander {
	return &Hander{
		nn: nn,
	}
}

func (h *Hander) RecogniseHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	requestMatrix := make([][]int, 0)
	err := json.NewDecoder(r.Body).Decode(&requestMatrix)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	input := data.PrepareMatrix(requestMatrix)
	result := helper.FindIndexWithMaxElem(h.nn.Query(input))
	resp := recognitionResponse{
		Answer: result,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(resp)
}

type recognitionResponse struct {
	Answer int `json:"answer"`
}
