package classifier_test

import (
	"context"
	"fmt"
	"log"

	"github.com/FrenchMajesty/consistent-classifier/adapters"
	"github.com/FrenchMajesty/consistent-classifier"
)

// Example shows basic usage of the classifier
func Example_basic() {
	// Create classifier - no clients provided, rely on defaults with environment variables
	clf, err := classifier.NewClassifier(classifier.Config{})
	if err != nil {
		log.Fatal(err)
	}

	// Classify some text
	result, err := clf.Classify(context.Background(), "Thanks for the help!")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Label: %s\n", result.Label)
	fmt.Printf("Cache Hit: %v\n", result.CacheHit)
	fmt.Printf("Latency: %v\n", result.UserFacingLatency)

	// Gracefully shutdown and save DSU state
	if err := clf.Close(); err != nil {
		log.Fatal(err)
	}
}

// Example shows customizing the configuration
func Example_customConfig() {
	// Create clients
	embeddingClient, err := adapters.NewVoyageEmbeddingAdapter(nil)
	if err != nil {
		log.Fatal(err)
	}

	vectorClientLabel, err := adapters.NewPineconeVectorAdapter(nil, nil, "my_namespace_label")
	if err != nil {
		log.Fatal(err)
	}

	vectorClientContent, err := adapters.NewPineconeVectorAdapter(nil, nil, "my_namespace_content")
	if err != nil {
		log.Fatal(err)
	}

	llmClient, err := adapters.NewDefaultLLMClient(nil, "", "", "", nil)
	if err != nil {
		log.Fatal(err)
	}

	// Customize configuration with higher similarity threshold
	clf, err := classifier.NewClassifier(classifier.Config{
		EmbeddingClient:      embeddingClient,
		VectorClientLabel:    vectorClientLabel,
		VectorClientContent:  vectorClientContent,
		LLMClient:            llmClient,
		MinSimilarityContent: 0.90, // Higher threshold for cache hits
		MinSimilarityLabel:   0.75, // Lower threshold for cache hits
		DSUPersistence:       classifier.NewFileDSUPersistence("./my_labels.bin"),
	})
	if err != nil {
		log.Fatal(err)
	}

	// Classify text
	result, err := clf.Classify(context.Background(), "How do I install this package?")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Label: %s\n", result.Label)

	// Get metrics
	metrics := clf.GetMetrics()
	fmt.Printf("Unique Labels: %d\n", metrics.UniqueLabels)
	fmt.Printf("Converged Labels: %d\n", metrics.ConvergedLabels)
	fmt.Printf("Cache Hit Rate: %.2f%%\n", metrics.CacheHitRate)

	// Gracefully shutdown and save state
	if err := clf.Close(); err != nil {
		log.Fatal(err)
	}
}
