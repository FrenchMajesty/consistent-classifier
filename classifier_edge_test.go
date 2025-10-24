package classifier

import (
	"context"
	"strings"
	"testing"

	"github.com/FrenchMajesty/consistent-classifier/internal/disjoint_set"
	"github.com/FrenchMajesty/consistent-classifier/types"
)

// MockEmbeddingClient for testing
type MockEmbeddingClient struct {
	shouldError bool
}

func (m *MockEmbeddingClient) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	if m.shouldError {
		return nil, nil
	}
	// Return a dummy embedding
	return []float32{0.1, 0.2, 0.3}, nil
}

// MockVectorClient for testing
type MockVectorClient struct{}

func (m *MockVectorClient) Search(ctx context.Context, vector []float32, topK int) ([]types.VectorMatch, error) {
	return []types.VectorMatch{}, nil
}

func (m *MockVectorClient) Upsert(ctx context.Context, id string, vector []float32, metadata map[string]any) error {
	return nil
}

// MockLLMClient for testing
type MockLLMClient struct {
	returnLabel string
}

func (m *MockLLMClient) Classify(ctx context.Context, text string) (string, error) {
	return m.returnLabel, nil
}

// TestClassifyWithEmptyText tests that empty text is rejected
func TestClassifyWithEmptyText(t *testing.T) {
	tests := []struct {
		name        string
		text        string
		wantError   bool
		errorContains string
	}{
		{
			name:        "empty string",
			text:        "",
			wantError:   true,
			errorContains: "cannot classify empty text",
		},
		{
			name:        "whitespace only - spaces",
			text:        "   ",
			wantError:   true,
			errorContains: "cannot classify empty text",
		},
		{
			name:        "whitespace only - tabs",
			text:        "\t\t",
			wantError:   true,
			errorContains: "cannot classify empty text",
		},
		{
			name:        "whitespace only - newlines",
			text:        "\n\n",
			wantError:   true,
			errorContains: "cannot classify empty text",
		},
		{
			name:        "whitespace only - mixed",
			text:        " \t\n ",
			wantError:   true,
			errorContains: "cannot classify empty text",
		},
		{
			name:        "valid text",
			text:        "Hello world",
			wantError:   false,
		},
		{
			name:        "valid text with surrounding whitespace",
			text:        "  Hello world  ",
			wantError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a minimal classifier for testing
			c := &Classifier{
				embedding:            &MockEmbeddingClient{},
				vectorContent:        &MockVectorClient{},
				vectorLabel:          &MockVectorClient{},
				llm:                  &MockLLMClient{returnLabel: "test"},
				dsu:                  disjoint_set.NewDSU(),
				minSimilarityContent: 0.8,
				minSimilarityLabel:   0.8,
			}

			ctx := context.Background()
			_, err := c.Classify(ctx, tt.text)

			if tt.wantError {
				if err == nil {
					t.Errorf("Classify() error = nil, want error containing %q", tt.errorContains)
					return
				}
				if !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("Classify() error = %v, want error containing %q", err, tt.errorContains)
				}
			} else {
				if err != nil {
					t.Errorf("Classify() unexpected error = %v", err)
				}
			}
		})
	}
}

// TestClassifyWithEmptyLabel tests that empty labels from LLM are rejected
func TestClassifyWithEmptyLabel(t *testing.T) {
	tests := []struct {
		name        string
		llmLabel    string
		wantError   bool
		errorContains string
	}{
		{
			name:        "empty label",
			llmLabel:    "",
			wantError:   true,
			errorContains: "LLM returned empty label",
		},
		{
			name:        "whitespace label",
			llmLabel:    "   ",
			wantError:   true,
			errorContains: "LLM returned empty label",
		},
		{
			name:        "tab label",
			llmLabel:    "\t",
			wantError:   true,
			errorContains: "LLM returned empty label",
		},
		{
			name:        "newline label",
			llmLabel:    "\n",
			wantError:   true,
			errorContains: "LLM returned empty label",
		},
		{
			name:        "valid label",
			llmLabel:    "tech",
			wantError:   false,
		},
		{
			name:        "valid label with whitespace",
			llmLabel:    "  tech  ",
			wantError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Classifier{
				embedding:            &MockEmbeddingClient{},
				vectorContent:        &MockVectorClient{},
				vectorLabel:          &MockVectorClient{},
				llm:                  &MockLLMClient{returnLabel: tt.llmLabel},
				dsu:                  disjoint_set.NewDSU(),
				minSimilarityContent: 0.8,
				minSimilarityLabel:   0.8,
			}

			ctx := context.Background()
			_, err := c.Classify(ctx, "test text")

			if tt.wantError {
				if err == nil {
					t.Errorf("Classify() error = nil, want error containing %q", tt.errorContains)
					return
				}
				if !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("Classify() error = %v, want error containing %q", err, tt.errorContains)
				}
			} else {
				if err != nil {
					t.Errorf("Classify() unexpected error = %v", err)
				}
			}
		})
	}
}

// TestUpdateLabelClusteringWithEmptyLabel tests empty label handling
func TestUpdateLabelClusteringWithEmptyLabel(t *testing.T) {
	tests := []struct {
		name      string
		label     string
		wantError bool
	}{
		{
			name:      "empty label",
			label:     "",
			wantError: false, // Should return nil without error
		},
		{
			name:      "whitespace label",
			label:     "   ",
			wantError: false, // Should return nil without error
		},
		{
			name:      "valid label",
			label:     "tech",
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Classifier{
				embedding:   &MockEmbeddingClient{},
				vectorLabel: &MockVectorClient{},
				dsu:         disjoint_set.NewDSU(),
			}

			ctx := context.Background()
			err := c.updateLabelClustering(ctx, tt.label)

			if tt.wantError && err == nil {
				t.Error("updateLabelClustering() error = nil, want error")
			}
			if !tt.wantError && err != nil {
				t.Errorf("updateLabelClustering() unexpected error = %v", err)
			}
		})
	}
}

// TestCacheLabelEmbeddingWithEmptyLabel tests empty label handling
func TestCacheLabelEmbeddingWithEmptyLabel(t *testing.T) {
	tests := []struct {
		name      string
		label     string
		wantError bool
	}{
		{
			name:      "empty label",
			label:     "",
			wantError: false, // Should return nil without error
		},
		{
			name:      "whitespace label",
			label:     "  \t  ",
			wantError: false, // Should return nil without error
		},
		{
			name:      "valid label",
			label:     "sports",
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Classifier{
				embedding:   &MockEmbeddingClient{},
				vectorLabel: &MockVectorClient{},
				dsu:         disjoint_set.NewDSU(),
			}

			ctx := context.Background()
			err := c.cacheLabelEmbedding(ctx, tt.label)

			if tt.wantError && err == nil {
				t.Error("cacheLabelEmbedding() error = nil, want error")
			}
			if !tt.wantError && err != nil {
				t.Errorf("cacheLabelEmbedding() unexpected error = %v", err)
			}
		})
	}
}
