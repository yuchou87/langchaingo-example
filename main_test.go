package main

import (
	"context"
	"fmt"
	"log"
	"net/url"
	"testing"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores/qdrant"
)

var (
	llm *ollama.LLM
	err error
	ctx context.Context
)

const (
	QdrantUrl      = "http://localhost:6333"
	CollectionName = "llm_ollama_example"
)

func init() {
	ctx = context.Background()
	llm, err = ollama.New(ollama.WithModel("phi"))
	if err != nil {
		panic(fmt.Sprintf("new ollama got a error: %+v", err))
	}
}

func Test_Ollama_Chat(t *testing.T) {
	content := []llms.MessageContent{
		llms.TextParts(schema.ChatMessageTypeSystem, "You are a company branding design wizard."),
		llms.TextParts(schema.ChatMessageTypeHuman, "What would be a good company name a company that makes colorful socks?"),
	}

	completion, err := llm.GenerateContent(ctx, content, llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
		fmt.Print(string(chunk))
		return nil
	}))

	if err != nil {
		log.Fatal(err)
	}

	_ = completion
}

func Test_Ollama_Completion(t *testing.T) {
	completion, err := llms.GenerateFromSinglePrompt(
		ctx,
		llm,
		"Human: Who was the first man to walk on the moon?\nAssistant:",
		llms.WithTemperature(0.8),
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			fmt.Print(string(chunk))
			return nil
		}),
	)

	if err != nil {
		log.Fatal(err)
	}

	_ = completion
}

func Test_Ollama_Qdrant_Vectorstore(t *testing.T) {
	ollamaEmbeder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatal(err)
	}

	// Create a new Qdrant vector store.
	qdrantURL, err := url.Parse(QdrantUrl)
	if err != nil {
		log.Fatal(err)
	}

	store, err := qdrant.New(
		qdrant.WithURL(*qdrantURL),
		qdrant.WithEmbedder(ollamaEmbeder),
		qdrant.WithCollectionName("llm_ollama_example"),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Add documents to the Qdrant vector store.
	_, err = store.AddDocuments(context.Background(), []schema.Document{
		{
			PageContent: "A city in texas",
			Metadata: map[string]any{
				"area": 3251,
			},
		},
		{
			PageContent: "A country in Asia",
			Metadata: map[string]any{
				"area": 2342,
			},
		},
		{
			PageContent: "A country in South America",
			Metadata: map[string]any{
				"area": 432,
			},
		},
		{
			PageContent: "An island nation in the Pacific Ocean",
			Metadata: map[string]any{
				"area": 6531,
			},
		},
		{
			PageContent: "A mountainous country in Europe",
			Metadata: map[string]any{
				"area": 1211,
			},
		},
		{
			PageContent: "A lost city in the Amazon",
			Metadata: map[string]any{
				"area": 1223,
			},
		},
		{
			PageContent: "A city in England",
			Metadata: map[string]any{
				"area": 4324,
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	// Search for similar documents.
	docs, err := store.SimilaritySearch(ctx, "england", 1)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(docs)
}
