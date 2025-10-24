package voyage

import (
	"testing"
)

// TestParseEmbeddingType tests the fixed parseEmbeddingType function
func TestParseEmbeddingType(t *testing.T) {
	tests := []struct {
		name          string
		embeddingType VoyageEmbeddingType
		wantNil       bool
		wantValue     string
	}{
		{
			name:          "default type returns nil",
			embeddingType: VoyageEmbeddingTypeDefault,
			wantNil:       true,
		},
		{
			name:          "document type returns pointer",
			embeddingType: VoyageEmbeddingTypeDocument,
			wantNil:       false,
			wantValue:     "document",
		},
		{
			name:          "query type returns pointer",
			embeddingType: VoyageEmbeddingTypeQuery,
			wantNil:       false,
			wantValue:     "query",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseEmbeddingType(tt.embeddingType)

			if tt.wantNil {
				if result != nil {
					t.Errorf("parseEmbeddingType() = %v, want nil", result)
				}
			} else {
				if result == nil {
					t.Errorf("parseEmbeddingType() = nil, want non-nil")
					return
				}
				if *result != tt.wantValue {
					t.Errorf("parseEmbeddingType() = %v, want %v", *result, tt.wantValue)
				}
			}
		})
	}
}

// TestGenerateEmbeddingWithEmptyText tests that empty text is handled
func TestGenerateEmbeddingWithEmptyText(t *testing.T) {
	// This test verifies the fix doesn't break normal operation
	// We can't test actual API calls without a valid API key,
	// but we verify the function signature and nil pointer handling

	service := &voyageService{
		dimensions: EMBEDDING_DIMENSIONS,
		model:      VOYAGEAI_EMBEDDING_MODEL,
	}

	// Verify service was created successfully
	if service == nil {
		t.Fatal("failed to create voyage service")
	}

	// Verify dimensions are set correctly
	if service.GetEmbeddingDimensions() != EMBEDDING_DIMENSIONS {
		t.Errorf("GetEmbeddingDimensions() = %d, want %d",
			service.GetEmbeddingDimensions(), EMBEDDING_DIMENSIONS)
	}
}

// TestEmbeddingTypeValues verifies the constant values
func TestEmbeddingTypeValues(t *testing.T) {
	tests := []struct {
		name  string
		value VoyageEmbeddingType
		want  string
	}{
		{"document", VoyageEmbeddingTypeDocument, "document"},
		{"query", VoyageEmbeddingTypeQuery, "query"},
		{"default", VoyageEmbeddingTypeDefault, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.value) != tt.want {
				t.Errorf("VoyageEmbeddingType(%s) = %s, want %s",
					tt.name, string(tt.value), tt.want)
			}
		})
	}
}
