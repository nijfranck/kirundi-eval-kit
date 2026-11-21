# Kirundi evaluation kit

Tools to measure how well language models handle Kirundi, and to prepare Kirundi SMS text for them. Built for Turikumwe, the AI layer of a listening and referral line in Burundi where young people ask about sexual and reproductive health by SMS and WhatsApp, and where a human moderator approves every reply.

The benchmark is not tied to Kirundi: the language is a parameter, so the same commands run for any of the 200 languages in FLORES-200.

Status: first public release, November 2026. The tools work and are tested; the results are early and small in scale (see the limits below). The code belongs to AJEBUDI (Yaga Burundi) and was written with the help of an AI coding assistant; every score comes from a real run.

## What is here

| Path | What it does |
| --- | --- |
| `benchmark/translation.py` | Compares models on translation to and from a language, on telling it apart from a close relative, and on drifting into that relative. Works with any OpenAI-compatible API (OpenRouter, Ollama, vLLM, llama.cpp). |
| `benchmark/local_mt.py` | Runs small open translation models (NLLB-200, MADLAD-400) on your own machine and scores them in the same table. |
| `benchmark/risk_detection.py` | Measures whether a model flags emergencies in short messages, alone and together with a danger-phrase list. |
| `kirundi_kit/text.py` | Kirundi SMS normalisation, glossary matching that tolerates dropped vowels and glued words, markers of Kinyarwanda spelling, and a quick Kirundi-or-French guess. Standard library only. |
| `kirundi_kit/risk.py` | A deterministic danger-phrase check to run before any model. Standard library only. |
| `data/*_example.json` | Small files that show the expected formats. They are not test sets. |
| `results/` | Scores from our own runs. |

## What is not here, and why

- No real conversations, phone numbers or moderator corrections. They are private and stay private.
- No glossary, danger-phrase list or test messages yet. Ours are still drafts. They will be published only after review by native speakers, health professionals and a safeguarding lead.
- No model outputs on our private health sentences, only the scores.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and Python 3.10 or later.

```sh
# 1. The public test sentences (FLORES-200, human translations, about 25 MB)
curl -sL https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz | tar xz -C data

# 2. A model behind an API
export OPENROUTER_API_KEY=...
uv run --with httpx --with sacrebleu python benchmark/translation.py \
    --flores data/flores200_dataset --models google/gemma-4-31b-it

# 3. A model on your own machine, through Ollama
uv run --with httpx --with sacrebleu python benchmark/translation.py \
    --flores data/flores200_dataset --base-url http://localhost:11434/v1 --models gemma4:e4b --concurrency 2

# 4. A small translation model on your own machine
uv run --with torch --with transformers --with sentencepiece --with protobuf --with httpx --with sacrebleu \
    python benchmark/local_mt.py --flores data/flores200_dataset --model facebook/nllb-200-distilled-1.3B

# 5. The table again, from what is cached
uv run --with httpx --with sacrebleu python benchmark/translation.py --flores data/flores200_dataset --report
```

Another language, for example Kiswahili against English:

```sh
uv run --with httpx --with sacrebleu python benchmark/translation.py --flores data/flores200_dataset \
    --lang swh_Latn --lang-name Kiswahili --pivots eng_Latn --confusable '' --models google/gemma-4-31b-it
```

Your own sentences (`--domain pairs.json`) are a JSON list such as `[{"run_Latn": "...", "fra_Latn": "..."}]`, written by people on both sides. On OpenRouter they are sent only to providers that do not keep data.

The text tools:

```python
from kirundi_kit import Glossary, kinyarwanda_markers, normalise

normalise('Mahorooo!! N,impore')            # "mahoro n'impore"
kinyarwanda_markers('Icyo ni byiza')        # two warnings: cy… and by… are Kinyarwanda spellings
Glossary.load('data/glossary_example.json').match('ndashaka dukingirizo').gloss()
```

Tests: `uv run --with pytest pytest tests`

## Results so far

See [results/RESULTS.md](results/RESULTS.md) for the tables and how to read them. In short, on our sentences:

- Only the NLLB-200 family writes Kirundi without drifting into Kinyarwanda; NLLB-200 1.3B scores 41.1 chrF++ against 43.5 for the best large model.
- Small models trained on African languages (AfriqueQwen 9B and 4B) read Kirundi as well as general models about three times their size.
- General models under 10 billion parameters fail at writing Kirundi: they loop, or write Kinyarwanda spellings.

## How to read the scores, and their limits

- chrF++ counts how many character and word sequences a translation shares with a human reference, from 0 to 100. It is an automatic score. It does not tell you whether a health message is safe to send.
- Each direction uses 60 FLORES sentences. Differences of two or three points are within the noise; the cached results carry a confidence interval for each score.
- FLORES is news-style text, not SMS. Models that saw FLORES in training may score higher than they deserve.
- Our private health set is small (22 sentence pairs), so treat its scores as a first signal only.
- A good average can hide the error that matters: a translation that keeps the topic but changes a symptom, a dose or a time. Only review by native speakers finds those.

## Licence

Apache License 2.0 for everything in this repository (see `LICENSE`). Copyright 2026 AJEBUDI. FLORES-200 is CC BY-SA 4.0 and is not redistributed here. Model licences vary: NLLB-200 is CC BY-NC 4.0 (non-commercial), MADLAD-400 is Apache 2.0.
