package classifier

const (
	// DefaultMinSimilarity is the default threshold for vector similarity matching
	DefaultMinSimilarity = 0.80

	// DefaultDSUFilePath is the default location for DSU state persistence
	DefaultDSUFilePath = "./dsu_state.bin"
)

// Config holds configuration for the Classifier
type Config struct {
	// EmbeddingClient generates embeddings for text. If nil, uses the default (Voyage AI).
	EmbeddingClient EmbeddingClient

	// VectorClient performs vector search and storage. If nil, uses the default (Pinecone).
	VectorClientLabel   VectorClient
	VectorClientContent VectorClient

	// LLMClient performs text classification. If nil, uses the default (OpenAI).
	LLMClient   LLMClient
	Model       string
	BaseUrl     string
	Temperature *float32 // Optional temperature for LLM. If nil, uses model default.

	// DSUPersistence handles loading/saving the label clustering state. If nil, uses file-based persistence at ./dsu_state.bin
	DSUPersistence DisjointSetPersistence

	// MinSimilarity is the threshold for vector similarity matching (0.0 to 1.0). If 0, uses DefaultMinSimilarity.
	MinSimilarityContent float32
	MinSimilarityLabel   float32
}

// applyDefaults fills in default values for unset config fields
func (c *Config) applyDefaults() {
	if c.MinSimilarityContent == 0 {
		c.MinSimilarityContent = DefaultMinSimilarity
	}

	if c.MinSimilarityLabel == 0 {
		c.MinSimilarityLabel = DefaultMinSimilarity
	}
}
