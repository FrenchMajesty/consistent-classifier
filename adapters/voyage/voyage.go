package voyage

import (
	"context"
	"fmt"
	"sync"

	"github.com/austinfhunter/voyageai"
)

var client *voyageai.VoyageClient
var once sync.Once

const EMBEDDING_DIMENSIONS = 1024

const VOYAGEAI_EMBEDDING_MODEL = "voyage-3.5-lite"

type VoyageEmbeddingType string

const (
	VoyageEmbeddingTypeDocument VoyageEmbeddingType = "document"
	VoyageEmbeddingTypeQuery    VoyageEmbeddingType = "query"
	VoyageEmbeddingTypeDefault  VoyageEmbeddingType = ""
)

// embeddingService handles generating embeddings for text
type voyageService struct {
	dimensions int
	model      string
}

// NewEmbeddingService creates a new embedding service
func NewEmbeddingService(apiKey string) *voyageService {
	once.Do(func() {
		client = voyageai.NewClient(&voyageai.VoyageClientOpts{
			Key: apiKey,
		})
	})

	instance := &voyageService{
		dimensions: EMBEDDING_DIMENSIONS,
		model:      VOYAGEAI_EMBEDDING_MODEL,
	}

	return instance
}

// SetDimensions sets the dimensions for the embedding model
func (es *voyageService) SetDimensions(dimensions int) {
	es.dimensions = dimensions
}

// SetModel sets the model for the embedding model
func (es *voyageService) SetModel(model string) {
	es.model = model
}

// GenerateEmbedding generates an embedding for a single text using VoyageAI
func (es *voyageService) GenerateEmbedding(ctx context.Context, text string, embeddingType VoyageEmbeddingType) ([]float32, error) {
	dimensions := es.GetEmbeddingDimensions()
	inputType := parseEmbeddingType(embeddingType)

	embeddings, err := client.Embed(
		[]string{text},
		es.model,
		&voyageai.EmbeddingRequestOpts{
			InputType:       inputType,
			OutputDimension: &dimensions,
		},
	)

	if err != nil {
		return nil, fmt.Errorf("could not get embedding: %w", err)
	}

	return embeddings.Data[0].Embedding, nil
}

// GenerateEmbeddings generates embeddings for multiple texts using VoyageAI
func (es *voyageService) GenerateEmbeddings(ctx context.Context, texts []string, embeddingType VoyageEmbeddingType) ([]voyageai.EmbeddingObject, error) {
	dimensions := es.GetEmbeddingDimensions()
	inputType := parseEmbeddingType(embeddingType)

	embeddings, err := client.Embed(
		texts,
		es.model,
		&voyageai.EmbeddingRequestOpts{
			InputType:       inputType,
			OutputDimension: &dimensions,
		},
	)

	if err != nil {
		return nil, fmt.Errorf("could not get embeddings: %w", err)
	}

	return embeddings.Data, nil
}

func parseEmbeddingType(embeddingType VoyageEmbeddingType) *string {
	if embeddingType != VoyageEmbeddingTypeDefault {
		value := string(embeddingType)
		return &value
	}
	return nil
}

// GetEmbeddingDimensions returns the dimension count for the embedding model
func (es *voyageService) GetEmbeddingDimensions() int {
	return es.dimensions
}
