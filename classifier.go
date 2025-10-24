package classifier

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/FrenchMajesty/consistent-classifier/adapters"
	"github.com/FrenchMajesty/consistent-classifier/internal/disjoint_set"
	"github.com/google/uuid"
)

// Classifier performs text classification with vector caching and label clustering
type Classifier struct {
	embedding            EmbeddingClient
	vectorContent        VectorClient
	vectorLabel          VectorClient
	llm                  LLMClient
	dsu                  *disjoint_set.DSU
	dsuPersist           DisjointSetPersistence
	minSimilarityContent float32
	minSimilarityLabel   float32

	// Metrics tracking
	totalClassifications int
	cacheHits            int
	metricsLock          sync.RWMutex

	// Background task tracking for graceful shutdown
	backgroundTasks sync.WaitGroup
	shutdownOnce    sync.Once
	closing         bool
	closeLock       sync.RWMutex
}

// NewClassifier creates a new Classifier with the given configuration
func NewClassifier(cfg Config) (*Classifier, error) {
	cfg.applyDefaults()

	// Initialize clients
	var embeddingClient EmbeddingClient
	if cfg.EmbeddingClient != nil {
		embeddingClient = cfg.EmbeddingClient
	} else {
		client, err := adapters.NewVoyageEmbeddingAdapter(nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create default embedding client: %w", err)
		}
		embeddingClient = client
	}

	var vectorClientLabel VectorClient
	if cfg.VectorClientLabel != nil {
		vectorClientLabel = cfg.VectorClientLabel
	} else {
		client, err := adapters.NewPineconeVectorAdapter(nil, nil, "label")
		if err != nil {
			return nil, fmt.Errorf("failed to create default vector client (label): %w", err)
		}
		vectorClientLabel = client
	}

	var vectorClientContent VectorClient
	if cfg.VectorClientContent != nil {
		vectorClientContent = cfg.VectorClientContent
	} else {
		client, err := adapters.NewPineconeVectorAdapter(nil, nil, "content")
		if err != nil {
			return nil, fmt.Errorf("failed to create default vector client (content): %w", err)
		}
		vectorClientContent = client
	}

	var llmClient LLMClient
	if cfg.LLMClient != nil {
		llmClient = cfg.LLMClient
	} else {
		client, err := adapters.NewDefaultLLMClient(nil, "", cfg.Model, cfg.BaseUrl, cfg.Temperature)
		if err != nil {
			return nil, fmt.Errorf("failed to create default LLM client: %w", err)
		}
		llmClient = client
	}

	var dsuPersist DisjointSetPersistence
	if cfg.DSUPersistence != nil {
		dsuPersist = cfg.DSUPersistence
	} else {
		dsuPersist = NewFileDSUPersistence(DefaultDSUFilePath)
	}

	// Load DSU from persistence
	dsu, err := dsuPersist.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load DSU: %w", err)
	}

	return &Classifier{
		embedding:            embeddingClient,
		vectorContent:        vectorClientContent,
		vectorLabel:          vectorClientLabel,
		llm:                  llmClient,
		dsu:                  dsu,
		dsuPersist:           dsuPersist,
		minSimilarityContent: cfg.MinSimilarityContent,
		minSimilarityLabel:   cfg.MinSimilarityLabel,
	}, nil
}

// Classify classifies the given text and returns the classification result
func (c *Classifier) Classify(ctx context.Context, text string) (*Result, error) {
	// Check if classifier is shutting down
	c.closeLock.RLock()
	if c.closing {
		c.closeLock.RUnlock()
		return nil, fmt.Errorf("classifier is shutting down")
	}
	c.closeLock.RUnlock()

	// Skip empty or whitespace-only text
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, fmt.Errorf("cannot classify empty text")
	}

	userFacingStart := time.Now()

	// Step 1: Generate embedding for this text
	embedding, err := c.embedding.GenerateEmbedding(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Step 2: Search vector cache for similar text
	matches, err := c.vectorContent.Search(ctx, embedding, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to search vector cache: %w", err)
	}

	// Check if we have a cache hit
	if len(matches) > 0 && matches[0].Score >= c.minSimilarityContent {
		// Cache HIT - return cached label
		userFacingLatency := time.Since(userFacingStart)
		label, ok := matches[0].Metadata["label"].(string)
		if !ok {
			return nil, fmt.Errorf("cached vector missing label metadata")
		}

		c.recordCacheHit()

		rootLabel := c.dsu.FindLabel(c.dsu.FindOrCreate(label))

		return &Result{
			Label:             rootLabel,
			CacheHit:          true,
			Confidence:        matches[0].Score,
			UserFacingLatency: userFacingLatency,
			BackgroundLatency: 0,
		}, nil
	}

	// Cache MISS - call LLM for classification
	label, err := c.llm.Classify(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to classify with LLM: %w", err)
	}

	// Validate label from LLM
	label = strings.TrimSpace(label)
	if label == "" {
		return nil, fmt.Errorf("LLM returned empty label")
	}

	userFacingLatency := time.Since(userFacingStart)
	c.recordClassification()

	// Track background task for graceful shutdown
	c.backgroundTasks.Add(1)
	defer c.backgroundTasks.Done()

	// Background processing - run asynchronously but wait for completion
	backgroundStart := time.Now()
	err = c.processBackgroundTasks(ctx, text, embedding, label)
	if err != nil {
		// Don't fail the classification, just log the error
		// In production you might want to handle this differently
		log.Printf("Error: background processing failed: %v\n", err)
	}
	backgroundLatency := time.Since(backgroundStart)

	return &Result{
		Label:             label,
		CacheHit:          false,
		Confidence:        0,
		UserFacingLatency: userFacingLatency,
		BackgroundLatency: backgroundLatency,
	}, nil
}

// processBackgroundTasks handles label clustering and vector caching
func (c *Classifier) processBackgroundTasks(ctx context.Context, text string, embedding []float32, label string) error {
	// Check if context is already cancelled
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	var wg sync.WaitGroup
	errChan := make(chan error, 3)

	// Task 1: Find similar labels and update DSU
	wg.Add(1)
	go func() {
		defer wg.Done()
		select {
		case <-ctx.Done():
			errChan <- ctx.Err()
			return
		default:
		}
		if err := c.updateLabelClustering(ctx, label); err != nil {
			errChan <- fmt.Errorf("label clustering failed: %w", err)
		}
	}()

	// Task 2: Cache the text embedding for future lookups
	wg.Add(1)
	go func() {
		defer wg.Done()
		select {
		case <-ctx.Done():
			errChan <- ctx.Err()
			return
		default:
		}
		if err := c.cacheTextEmbedding(ctx, text, embedding, label); err != nil {
			errChan <- fmt.Errorf("text caching failed: %w", err)
		}
	}()

	// Task 3: Cache the label embedding
	wg.Add(1)
	go func() {
		defer wg.Done()
		select {
		case <-ctx.Done():
			errChan <- ctx.Err()
			return
		default:
		}
		if err := c.cacheLabelEmbedding(ctx, label); err != nil {
			errChan <- fmt.Errorf("label caching failed: %w", err)
		}
	}()

	wg.Wait()
	close(errChan)

	// Return first error if any
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	return nil
}

// updateLabelClustering finds similar labels and merges them in the DSU
func (c *Classifier) updateLabelClustering(ctx context.Context, label string) error {
	// Skip empty or whitespace-only labels
	label = strings.TrimSpace(label)
	if label == "" {
		return nil
	}

	// Generate embedding for the label
	labelEmbedding, err := c.embedding.GenerateEmbedding(ctx, label)
	if err != nil {
		return err
	}

	// Search for similar labels
	matches, err := c.vectorLabel.Search(ctx, labelEmbedding, 1)
	if err != nil {
		return err
	}

	// Find the root label
	rootLabel := label
	if len(matches) > 0 && matches[0].Score >= c.minSimilarityLabel {
		root, ok := matches[0].Metadata["root"].(string)
		if ok {
			rootLabel = root
		}
	}

	// Union the label with the root label in DSU
	c.dsu.Union(c.dsu.FindOrCreate(rootLabel), c.dsu.FindOrCreate(label))

	return nil
}

// cacheTextEmbedding stores the text embedding in the vector database
func (c *Classifier) cacheTextEmbedding(ctx context.Context, text string, embedding []float32, label string) error {
	id := uuid.New().String()
	metadata := map[string]any{
		"vector_text": text,
		"label":       label,
	}
	return c.vectorContent.Upsert(ctx, id, embedding, metadata)
}

// cacheLabelEmbedding stores the label embedding in the vector database
func (c *Classifier) cacheLabelEmbedding(ctx context.Context, label string) error {
	// Skip empty or whitespace-only labels
	label = strings.TrimSpace(label)
	if label == "" {
		return nil
	}

	// Generate embedding for the label
	labelEmbedding, err := c.embedding.GenerateEmbedding(ctx, label)
	if err != nil {
		return err
	}

	// Find root label from DSU
	rootIdx := c.dsu.FindOrCreate(label)
	rootLabel := c.dsu.FindLabel(rootIdx)
	if rootLabel == "" {
		rootLabel = label
	}

	metadata := map[string]any{
		"vector_text": label,
		"label":       label,
		"root":        rootLabel,
	}
	return c.vectorLabel.Upsert(ctx, label, labelEmbedding, metadata)
}

// SaveDSU saves the current DSU state to persistent storage
// This method is thread-safe and waits for any pending background tasks to complete
func (c *Classifier) SaveDSU() error {
	// Wait for all background tasks to complete before saving
	c.backgroundTasks.Wait()
	return c.dsuPersist.Save(c.dsu)
}

// Close gracefully shuts down the classifier, waiting for background tasks to complete
// and saving the DSU state. It's safe to call Close multiple times.
func (c *Classifier) Close() error {
	var saveErr error

	c.shutdownOnce.Do(func() {
		// Mark as closing to reject new classifications
		c.closeLock.Lock()
		c.closing = true
		c.closeLock.Unlock()

		// Wait for all background tasks to complete
		c.backgroundTasks.Wait()

		// Save DSU state
		saveErr = c.dsuPersist.Save(c.dsu)
	})

	return saveErr
}

// GetMetrics returns current classification metrics
func (c *Classifier) GetMetrics() Metrics {
	c.metricsLock.RLock()
	defer c.metricsLock.RUnlock()

	var cacheHitRate float32
	if c.totalClassifications > 0 {
		cacheHitRate = float32(c.cacheHits) / float32(c.totalClassifications) * 100
	}

	return Metrics{
		UniqueLabels:    c.dsu.Size(),
		ConvergedLabels: c.dsu.CountSets(),
		CacheHitRate:    cacheHitRate,
	}
}

// recordCacheHit records a cache hit for metrics
func (c *Classifier) recordCacheHit() {
	c.metricsLock.Lock()
	defer c.metricsLock.Unlock()
	c.totalClassifications++
	c.cacheHits++
}

// recordClassification records a classification (cache miss) for metrics
func (c *Classifier) recordClassification() {
	c.metricsLock.Lock()
	defer c.metricsLock.Unlock()
	c.totalClassifications++
}
