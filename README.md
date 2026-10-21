# LLM Regional Bias Analysis — Multi-Model Sentiment Evaluation

Coursework for the *Fairness in AI* M1 class: measuring regional and political
bias in LLMs via target-oriented sentiment classification.

## Highlights

- Built a unified OpenRouter adapter (`openrouter_adapter.py`) that routes
  prompts to ~30 models (Llama, Mistral, Qwen, GPT, Claude, Gemini, DeepSeek,
  ...) through a single OpenAI-compatible client.
- Ran cross-entity sentiment evaluation over a hand-built set of named entities
  grouped by world region (Africa, Asia, Europe, North America, South America)
  and aggregated per-region bias scores (`regional_bias_results.csv`).
- Compared open-weight models run locally (via `ollama` / Hugging Face) against
  hosted APIs (OpenAI, OpenRouter) to disentangle training vs. serving effects.
- Wrote up findings as an ACL-style report (`docs/TEMPO_BIAS_C3_Report.pdf`)
  and a group presentation (`docs/Fairness_AI_G7_T4_presentation.pdf`).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...   # required by openrouter_adapter
jupyter lab                            # open Graph2.ipynb / results.ipynb
```

Local model weights (Meta-Llama-3-8B, llama-3-1-8b, etc.) are **not** included;
use `clone_model.py` to pull them from Hugging Face when needed.

## Layout

```
.
├── Graph2.ipynb                  # graph-based bias visualization
├── Region.ipynb                  # per-region aggregation
├── results.ipynb                 # consolidated results notebook
├── sentiment_openrouter.ipynb     # multi-model sentiment run via OpenRouter
├── openrouter_adapter.py          # unified OpenAI-compatible OpenRouter client
├── clone_model.py                 # Hugging Face snapshot_download helper
├── regional_bias_results.csv      # aggregated per-region bias scores
├── src/
│   └── outputs/fairness_ai_dataset/entity_sentence_pairs.csv   # tiny sample
└── docs/
    ├── main.tex / _main.tex       # ACL-style report source
    ├── TEMPO_BIAS_C3_Report.pdf   # compiled report
    ├── Fairness_AI_G7_T4_presentation.pdf
    └── figures/                   # result figures
```

## Notes

- This is **group coursework** for the Fairness in AI M1 course (group G7,
  T4). Notebooks and adapters here are the parts written by me; upstream
  reference pipelines (the cloned `TEMPO-BIAS` repo) are intentionally not
  included.
- A hardcoded OpenRouter API key that used to sit as a fallback in
  `sentiment_openrouter.ipynb` has been removed; the key is now read exclusively
  from `OPENROUTER_API_KEY`.
- Evaluation output folders (`per_entity_llm*/`, `data/`, `data 2/`) and model
  weights are excluded from version control.
