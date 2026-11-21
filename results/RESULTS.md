# Kirundi results so far

Runs from September to November 2026. The first 27 rows are in `models_summary_2026-11.json` with confidence intervals; the small open models below were added on 20 November 2026 on the same sentences.

## How the numbers were produced

- **Translation:** chrF++ (sacrebleu, word order 2) on 60 FLORES-200 devtest sentences per direction, sampled with seed 13.
- **Health sentences:** 22 private Kirundi–French sentence pairs about sexual and reproductive health. Only the scores are published.
- **Kirundi vs Kinyarwanda:** share of 50 sentences whose language the model names correctly.
- **Kinyarwanda drift:** share of the model's Kirundi output that contains Kinyarwanda spellings, as detected by `kirundi_kit.text.kinyarwanda_markers`.
- **Failed batches:** batches of ten sentences with no usable answer. They are left as failures, not repaired.

## Small open models (under 10 billion parameters, plus two larger Gemma models for reference)

| Model | Size | Licence | Kirundi→French | French→Kirundi | Health Kirundi→French | Health French→Kirundi | Kirundi vs Kinyarwanda | Kinyarwanda drift | Failed batches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NLLB-200 distilled 1.3B | 1.3B | CC BY-NC 4.0 | 46.2 | 41.1 | 37.7 | 30.6 | not tested | 0% | 0 |
| NLLB-200 distilled 600M | 0.6B | CC BY-NC 4.0 | 43.0 | 39.7 | 34.2 | 30.4 | not tested | 0% | 0 |
| NLLB 600M tuned for Kirundi–French (crespo12, v2) | 0.6B | CC BY-NC 4.0 | 42.7 | 39.2 | 31.4 | 29.4 | not tested | 0.7% | 0 |
| AfriqueQwen3.5-9B-50Langs-Instruct-v1 (4-bit) | 9B | CC BY 4.0 | 43.6 | 33.2 | 43.6 | 32.0 | 58% | 71.1% | 0 |
| AfriqueQwen3.5-4B-Instruct-v2 (4-bit) | 4B | CC BY 4.0 | 42.1 | 32.6 | 40.6 | 31.3 | 64% | 71.1% | 0 |
| Qwen 3.5 9B | 9B | Apache 2.0 | 42.8 | 27.9 | 41.4 | 25.1 | 50% | 81.0% | 10 of 35 |
| MADLAD-400 3B | 3B | Apache 2.0 | 41.5 | 23.3 | 31.1 | 24.7 | not tested | 69.7% | 0 |
| Hunyuan-MT2 7B | 7B | see model card | 34.8 | 19.3 | 34.3 | 19.3 | 48% | 11.7% | 10 of 35 |
| Gemma 4 E4B | about 4B effective | Apache 2.0 | 33.1 | 25.6 | 36.1 | 25.6 | 56% | 89.4% | 0 |
| Granite 4.2 8B | 8B | see model card | 25.3 | 17.5 | 25.0 | 14.3 | 50% | 34.1% | 1 of 35 |
| Ministral 8B | 8B | see model card | 25.2 | none completed | 24.9 | none completed | 46% | not scored | 15 of 35 |
| Gemma 4 26B-A4B | 26B (about 4B active) | Apache 2.0 | 41.9 | 31.5 | 45.2 | 30.2 | 84% | 12.7% | 0 |
| Gemma 4 31B | 31B | Apache 2.0 | 45.1 | 32.6 | 48.3 | 30.2 | 82% | 14.1% | 0 |

For comparison, the strongest large models on the same sentences: Claude Opus 5.5 reads at 58.2 (health 49.0); GPT-6 Astra writes at 43.5 (health 39.5).

## Urgency detection

104 invented messages in Kirundi, French, Kiswahili, English and mixed: 17 critical, 15 urgent, 72 that must not raise an alert. Model together with a danger-phrase list. The messages and the phrase list are drafts that native speakers have not yet reviewed, and they are not published.

| Model | Critical graded critical (of 17) | Urgent found (of 15) | No false alert (of 72) | High-risk messages with any alert (of 32) |
| --- | --- | --- | --- | --- |
| Phrase list alone | 4 | 4 | 72 | not computed |
| Gemma 4 31B | 17 | 13 | 69 | 30 |
| Gemma 4 26B-A4B | 17 | 12 | 65 | 29 |
| Qwen 3.5 9B | 16 | 13 | 68 | 30 |
| AfriqueQwen 9B | 13 | 15 | 64 | 32 |
| AfriqueQwen 4B | 13 | 14 | 68 | 30 |
| Gemma 4 E4B | 14 | 13 | 63 | 28 |
| Granite 4.2 8B | 14 | 12 | 67 | not computed |
| Ministral 8B | 11 | 12 | 62 | not computed |

AfriqueQwen ignores system messages, so it was run with `--single-turn` and `--extra-body '{"reasoning_effort":"none"}'`.

## What we take from this

- Only the NLLB family writes Kirundi without drifting into Kinyarwanda. Every instruction-following model under 10B drifts or fails.
- Training on African languages matters more than size for reading Kirundi: AfriqueQwen 4B scores 42.1 where a general model of similar size scores 33.1, and AfriqueQwen 9B is level with a general 26B model.
- A general Kirundi fine-tune of NLLB brought no gain on health sentences, which points to missing domain data.
- AfriqueQwen 9B raised an alert on every high-risk message but graded 4 critical ones as urgent.

## Limits

- 60 sentences per direction and 22 health pairs: differences of two or three points are within the noise.
- FLORES is news-style text, not SMS Kirundi, and some models may have seen it in training.
- The urgency messages are invented and unreviewed. A clean result on them is not proof of safety.
- AfriqueQwen was run at 4-bit on a laptop; the API models ran at their providers' precision.
- Not tested: Sunflower-Qwen3.5-9B (Sunbird AI) and the NLLB fine-tune by Ijwi ry'Ikirundi AI.
