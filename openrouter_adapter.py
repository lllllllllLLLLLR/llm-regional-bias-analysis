"""
OpenRouter adapter for accessing multiple LLM providers via unified API.

OpenRouter provides access to models from OpenAI, Anthropic, Meta, Mistral,
Google, and many others through an OpenAI-compatible API.

Usage:
    from openrouter_adapter import OpenRouterClient, AVAILABLE_MODELS

    client = OpenRouterClient(api_key="your-api-key")
    response = client.generate("What is 2+2?", model="meta-llama/llama-3.1-8b-instruct")
"""

import os
from typing import Optional
from openai import OpenAI


# Model registry with popular models available on OpenRouter
AVAILABLE_MODELS = {
    # Meta Llama models
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-3-70b": "meta-llama/llama-3-70b-instruct",
    "llama-3-8b": "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-2-13b-chat": "meta-llama/llama-2-13b-chat",

    # Mistral models
    "mistral-large": "mistralai/mistral-large-2411",
    "mistral-medium": "mistralai/mistral-medium",
    "mistral-small": "mistralai/mistral-small-2501",
    "mistral-7b": "mistralai/mistral-7b-instruct",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
    "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct",
    "mistralai/ministral-8b-2512": "mistralai/ministral-8b-2512",

    # OpenAI models
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",

    # Anthropic models
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-haiku": "anthropic/claude-3-haiku",

    # Google models
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "gemini-1.5-flash": "google/gemini-flash-1.5",

    # Qwen models
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "qwen-2.5-32b": "qwen/qwen-2.5-32b-instruct",
    "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct",

    # DeepSeek models
    "deepseek-v3": "deepseek/deepseek-chat",
    "deepseek-r1": "deepseek/deepseek-r1",

    # Cohere models
    "command-r-plus": "cohere/command-r-plus",
    "command-r": "cohere/command-r",

    # Other popular models
    "phi-3-medium": "microsoft/phi-3-medium-128k-instruct",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "gemma-2-9b": "google/gemma-2-9b-it",
}


class OpenRouterClient:
    """
    Client for OpenRouter API - provides access to multiple LLM providers.

    Args:
        api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
        site_url: Optional URL of your site for rankings on openrouter.ai
        site_name: Optional name of your site for rankings
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )

        self.site_url = site_url
        self.site_name = site_name

        # Build extra headers for OpenRouter
        extra_headers = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_name:
            extra_headers["X-Title"] = site_name

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url=self.BASE_URL,
            api_key=self.api_key,
            default_headers=extra_headers if extra_headers else None,
        )

    def _resolve_model(self, model: str) -> str:
        """Resolve model shorthand to full OpenRouter model ID."""
        return AVAILABLE_MODELS.get(model, model)

    def generate(
        self,
        prompt: str,
        model: str = "llama-3.1-8b",
        system_prompt: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 100,
        top_p: float = 1.0,
    ) -> str:
        """
        Generate a response from the specified model.

        Args:
            prompt: User prompt text
            model: Model name (shorthand or full OpenRouter ID)
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            Generated text response
        """
        model_id = self._resolve_model(model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        return response.choices[0].message.content.strip()

    def list_models(self) -> dict:
        """List available model shorthands."""
        return AVAILABLE_MODELS.copy()


def run_sentiment_analysis(
    client: OpenRouterClient,
    prompt: str,
    model: str = "llama-3.1-8b",
) -> str:
    """
    Run sentiment analysis using OpenRouter.

    Args:
        client: OpenRouterClient instance
        prompt: The full sentiment analysis prompt
        model: Model to use

    Returns:
        Predicted sentiment (negative/neutral/positive)
    """
    response = client.generate(
        prompt=prompt,
        model=model,
        temperature=0,
        max_tokens=16,
    )
    return normalize_prediction(response)


def normalize_prediction(raw: str) -> str:
    """Normalize model output to one of: negative, neutral, positive."""
    if not raw:
        return ""
    w = raw.strip().lower()
    valid = {"negative", "neutral", "positive"}
    if w in valid:
        return w
    if w.startswith("neg"):
        return "negative"
    if w.startswith("neu"):
        return "neutral"
    if w.startswith("pos"):
        return "positive"
    # Try to find sentiment word in response
    for sentiment in valid:
        if sentiment in w:
            return sentiment
    return w


# Convenience function for batch processing
def process_entity_file(
    client: OpenRouterClient,
    input_path: str,
    output_path: str,
    model: str = "llama-3.1-8b",
    prompt_column: str = "prompt",
) -> None:
    """
    Process an entity CSV file and add predicted sentiments.

    Args:
        client: OpenRouterClient instance
        input_path: Path to input CSV
        output_path: Path for output CSV
        model: Model to use
        prompt_column: Name of the column containing prompts
    """
    import pandas as pd

    df = pd.read_csv(input_path)
    predictions = []

    for prompt_text in df[prompt_column]:
        pred = run_sentiment_analysis(client, prompt_text, model)
        predictions.append(pred)

    df["predicted_sentiment"] = predictions
    df["model"] = model  # Track which model made the predictions
    df.to_csv(output_path, index=False)


def get_model_folder_name(model: str) -> str:
    """Convert model name to a safe folder name."""
    return model.replace("/", "_").replace(".", "-")


def process_all_entities(
    client: OpenRouterClient,
    input_dir: str,
    output_base_dir: str,
    models: list,
    prompt_column: str = "prompt",
    max_entities: int = None,
    verbose: bool = True,
) -> None:
    """
    Process all entity CSV files with multiple models.
    Creates a separate folder for each model's results.

    Args:
        client: OpenRouterClient instance
        input_dir: Directory containing entity CSV files
        output_base_dir: Base directory for output (model folders created here)
        models: List of models to run (shorthands or full IDs)
        prompt_column: Name of the column containing prompts
        max_entities: Limit number of entities (for testing)
        verbose: Print progress updates

    Output structure:
        output_base_dir/
            llama-3-1-8b/
                Entity1.csv
                Entity2.csv
            mistral-7b/
                Entity1.csv
                Entity2.csv
    """
    from pathlib import Path
    import pandas as pd

    input_path = Path(input_dir)
    output_base = Path(output_base_dir)

    entity_files = sorted(input_path.glob("*.csv"))
    if max_entities:
        entity_files = entity_files[:max_entities]

    if verbose:
        print(f"Found {len(entity_files)} entity files")
        print(f"Running {len(models)} model(s): {models}")
        print("=" * 60)

    for model in models:
        # Create model-specific output folder
        model_folder = get_model_folder_name(model)
        model_output_dir = output_base / model_folder
        model_output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\nModel: {model}")
            print(f"Output folder: {model_output_dir}")
            print("-" * 40)

        for i, csv_path in enumerate(entity_files):
            df = pd.read_csv(csv_path)
            predictions = []

            for prompt_text in df[prompt_column]:
                pred = run_sentiment_analysis(client, prompt_text, model)
                predictions.append(pred)

            df["predicted_sentiment"] = predictions
            df["model"] = model

            out_path = model_output_dir / csv_path.name
            df.to_csv(out_path, index=False)

            if verbose and ((i + 1) % 10 == 0 or i == 0):
                print(f"  [{i + 1}/{len(entity_files)}] {csv_path.name}")

        if verbose:
            print(f"Saved {len(entity_files)} files to {model_output_dir}")


if __name__ == "__main__":
    # Quick test
    print("Available models:")
    for short, full in AVAILABLE_MODELS.items():
        print(f"  {short}: {full}")
