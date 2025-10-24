package adapters

import (
	"context"
	"fmt"
	"strings"

	"github.com/FrenchMajesty/consistent-classifier/adapters/openai"
)

// DefaultLLMClient implements LLMClient using OpenAI
type DefaultLLMClient struct {
	client       openai.LanguageModelClient
	systemPrompt string
	model        string
	baseUrl      string
	temperature  *float32 // Optional temperature. If nil, omit from request.
}

const defaultModel = "gpt-4.1-mini"
const defaultSystemPrompt = `You are a text classification assistant. Given a text, classify it into a concise category label.

Rules:
- Return ONLY the category label, nothing else
- Use lowercase with underscores (e.g., "technical_question", "expressing_gratitude")
- Keep labels short and descriptive (2-5 words max)
- Be consistent: similar texts should get the same label`

// NewDefaultLLMClient creates a new LLM client using OpenAI with API key from environment
func NewDefaultLLMClient(apiKey *string, systemPrompt string, model string, baseUrl string, temperature *float32) (*DefaultLLMClient, error) {
	key, err := loadEnvVar(apiKey, "OPENAI_API_KEY")
	if err != nil {
		return nil, err
	}

	instance := DefaultLLMClient{
		client:       openai.NewClient(*key),
		systemPrompt: defaultSystemPrompt,
		model:        defaultModel,
		baseUrl:      baseUrl,
		temperature:  temperature,
	}

	if systemPrompt != "" {
		instance.systemPrompt = systemPrompt
	}

	if model != "" {
		instance.model = model
	}

	return &instance, nil
}

// Classify classifies text into a category label using LLM
func (c *DefaultLLMClient) Classify(ctx context.Context, text string) (string, error) {
	req := openai.ChatCompletionRequest{
		Model: c.model,
		Messages: []openai.ChatMessage{
			{
				Role:    openai.MessageRoleSystem,
				Content: &c.systemPrompt,
			},
			{
				Role:    openai.MessageRoleUser,
				Content: &text,
			},
		},
		MaxCompletionTokens: 50,
	}

	// Only set temperature if specified (some models like gpt-5-nano don't support it)
	if c.temperature != nil {
		req.Temperature = *c.temperature
	}

	resp, err := c.client.ChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to get LLM response: %w", err)
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == nil {
		return "", fmt.Errorf("no response from LLM")
	}

	label := strings.TrimSpace(*resp.Choices[0].Message.Content)
	label = strings.ToLower(label)

	return label, nil
}
