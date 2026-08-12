package main

// webModel describes a model that can be selected from the Web UI.
type webModel struct {
	ID          string `json:"id"`
	Provider    string `json:"provider"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Default     bool   `json:"default,omitempty"`
	Custom      bool   `json:"custom,omitempty"`
}

// webModelCatalog intentionally contains chat-capable models supported by the
// providers implemented by go-agent. The custom entry allows arbitrary model
// IDs, including local Ollama models and newly released provider models.
var webModelCatalog = []webModel{
	{ID:"gpt-5.4",Provider:"openai",Name:"GPT-5.4",Description:"OpenAI flagship reasoning model"},
	{ID:"gpt-5.4-mini",Provider:"openai",Name:"GPT-5.4 Mini",Description:"OpenAI fast, efficient reasoning model"},
	{ID:"gpt-5.4-nano",Provider:"openai",Name:"GPT-5.4 Nano",Description:"OpenAI low-latency model"},
	{ID:"gpt-5.3-codex",Provider:"openai",Name:"GPT-5.3 Codex",Description:"OpenAI coding model"},
	{ID:"claude-opus-4-1",Provider:"anthropic",Name:"Claude Opus 4.1",Description:"Anthropic high-capability model"},
	{ID:"claude-sonnet-4-1",Provider:"anthropic",Name:"Claude Sonnet 4.1",Description:"Anthropic balanced model"},
	{ID:"claude-haiku-4-1",Provider:"anthropic",Name:"Claude Haiku 4.1",Description:"Anthropic fast model"},
	{ID:"gemini-3.6-flash",Provider:"gemini",Name:"Gemini 3.6 Flash",Description:"Google fast agentic/coding model"},
	{ID:"gemini-3.5-flash",Provider:"gemini",Name:"Gemini 3.5 Flash",Description:"Google general-purpose agentic model"},
	{ID:"gemini-3.5-flash-lite",Provider:"gemini",Name:"Gemini 3.5 Flash Lite",Description:"Google low-latency model"},
	{ID:"gemini-3.1-pro-preview",Provider:"gemini",Name:"Gemini 3.1 Pro Preview",Description:"Google advanced reasoning preview"},
	{ID:"gemini-3.1-flash-lite",Provider:"gemini",Name:"Gemini 3.1 Flash Lite",Description:"Google efficient model"},
	{ID:"gemini-3.1-flash-image",Provider:"gemini",Name:"Gemini 3.1 Flash Image",Description:"Google multimodal/image model"},
	{ID:"gemini-3-pro-image",Provider:"gemini",Name:"Gemini 3 Pro Image",Description:"Google professional image model"},
	{ID:"gemini-omni-flash-preview",Provider:"gemini",Name:"Gemini Omni Flash Preview",Description:"Google multimodal preview"},
	{ID:"vertex:gemini-3.6-flash",Provider:"vertex",Name:"Vertex Gemini 3.6 Flash",Description:"Google Cloud Vertex AI"},
	{ID:"vertex:gemini-3.5-flash",Provider:"vertex",Name:"Vertex Gemini 3.5 Flash",Description:"Google Cloud Vertex AI"},
	{ID:"llama3.3",Provider:"ollama",Name:"Llama 3.3",Description:"Local Ollama model"},
	{ID:"qwen3",Provider:"ollama",Name:"Qwen 3",Description:"Local Ollama model"},
	{ID:"deepseek-r1",Provider:"ollama",Name:"DeepSeek R1",Description:"Local Ollama reasoning model"},
	{ID:"custom",Provider:"custom",Name:"Custom model…",Description:"Enter any model ID supported by a configured provider",Custom:true},
}
