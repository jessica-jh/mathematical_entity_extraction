# MMEE — Mathematical Entity Extraction

Fine-tuning an LLM to extract structured entities from raw mathematical text.

## Overview

Math textbooks read as a wall of undifferentiated text once you strip the PDF formatting —
`\textbf{Theorem}` and `\begin{proof}` markup is gone, and what's left is just paragraphs.
But a human reader still knows, sentence by sentence, "this is the claim, this is the proof,
this is a worked example." That structure is exactly what's missing from raw OCR/markdown
dumps of math sources, and it's exactly what you'd need before you could do anything more
interesting downstream: build a searchable index of "every theorem about compact groups,"
feed clean (statement, proof) pairs to a math-reasoning model instead of noisy prose, or
generate a study guide that separates definitions from examples automatically.

Off-the-shelf NER doesn't help here — general-purpose taggers aren't trained on math prose,
and the entities aren't neat single tokens; a "proof" span can run for three paragraphs and
be interleaved with LaTeX, references to earlier theorems, and worked examples. This project
treats it as a **span-extraction problem over raw document text**: given a chunk of a math
document, output character-offset spans labeled as one of six entity types — **definition,
theorem, proof, example, name, reference**.

The core question this project explores: **can a math-specialized 7B model, fine-tuned on a
small labeled set, beat prompting a base model for structured span extraction?** (Spoiler:
few-shot prompting collapses almost completely on this task — see [Results](#results).)

```
{"fileid": "algebra_lang.mmd", "start": 1024, "end": 1310, "tag": "theorem"}
```

## Results

Three stages were evaluated in order, each fixing what broke the previous one:

```
Token-level F1 (validation set)

BIO baseline (few-shot)    ▏ 0.000
Span baseline (few-shot)   ▏ 0.002
LoRA fine-tune             ████████████████████████████████████ 0.434
                            0.0        0.1        0.2        0.3        0.4
```

| Stage | Approach | Precision | Recall | F1 | What happened |
|---|---|:---:|:---:|:---:|---|
| 1 | Few-shot, **BIO** output format | 0.000 | 0.000 | 0.000 | Model couldn't reliably emit valid token-index spans in the requested format — near-total format failure. |
| 2 | Few-shot, **span (JSON)** output format | 0.174 | 0.001 | 0.002 | Switching to direct `{tag, start, end}` JSON let the model produce *some* valid output, but it found almost none of the true spans. |
| 3 | **LoRA fine-tune** on Qwen2.5-Math-7B-Instruct | 0.451 | 0.418 | **0.434** | Task-specific supervision, not a better prompt, is what the base model was missing — F1 jumps ~200x over the best baseline. |

*All numbers are token-level P/R/F1 on the held-out validation set.*

### Per-entity breakdown (fine-tuned model)

Performance isn't uniform across entity types — structurally distinctive entities are much
easier than short, ambiguous ones:

| Tag | Precision | Recall | F1 | Support (chars) |
|---|:---:|:---:|:---:|:---:|
| definition | 0.454 | 0.503 | **0.477** | 4,755 |
| proof | 0.507 | 0.383 | 0.437 | 4,240 |
| theorem | 0.432 | 0.385 | 0.407 | 3,448 |
| example | 0.430 | 0.385 | 0.406 | 1,294 |
| name | 0.169 | 0.164 | 0.166 | 421 |

*`reference` is omitted — 0 occurrences in the validation set, so it's not measurable here.*

`definition`, `theorem`, `proof`, and `example` — the long, structurally distinctive spans —
land in a fairly tight 0.41–0.48 F1 band. `name` (short spans like a mathematician's name or
a single symbol) lags well behind at 0.17: short spans give the model far less surrounding
context to key off of, and are easy to miss or mis-bound by a few characters, which sinks
character-level F1 disproportionately.

## Approach

1. **Baseline (no fine-tuning).** Two few-shot prompting strategies against the base model
   were compared:
   - *BIO*: whitespace-tokenize the input, ask the model to output token-index spans per tag.
   - *Span extraction*: ask the model to directly emit `{tag, start, end}` JSON objects over
     raw character offsets.
   Both were evaluated to see which output format a base (non-fine-tuned) model could follow
   more reliably — span extraction won, but both were far from usable.

2. **Fine-tuning.** [`Qwen2.5-Math-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
   (a math-specialized base model) was fine-tuned with **LoRA** (rank 16, all attention +
   MLP projection matrices) via [Unsloth](https://github.com/unslothai/unsloth), in 4-bit,
   on an instruction-formatted version of the training annotations: given a text chunk, emit
   a JSON list of `{tag, text}` entity spans.

3. **Inference & postprocessing.** Raw model output (JSON strings) is parsed, and predicted
   `text` snippets are matched back to character offsets (`start`/`end`) in the source
   document to produce the final span format.

4. **Error analysis.** The fine-tuned model was run on unannotated documents outside the
   training/validation set, and predictions were paired with surrounding context for manual
   review (see [Limitations](#limitations-known-issues) below for what that surfaced).

## Project Structure

```
├── data/
│   ├── train.json, val.json        # character-span annotations (fileid, start, end, tag)
│   ├── file_contents.json          # full source text, keyed by fileid
│   └── unannotated_mmds/           # unlabeled documents used for error analysis
├── src/
│   ├── baseline/                   # Part 1: few-shot prompting (BIO + span variants)
│   │   ├── inference_bio.py / postprocess_bio.py / evaluate_bio.py
│   │   └── inference_span.py / postprocess_span.py / evaluate_span.py
│   ├── training/                   # Part 2: LoRA fine-tuning + validation
│   │   ├── preprocess.py           # train.json + file_contents.json -> train.jsonl
│   │   ├── train_lora.py           # LoRA fine-tune on Qwen2.5-Math-7B-Instruct
│   │   ├── val_inference.py / val_postprocess.py / val_evaluate.py
│   ├── inference/                  # test set + unannotated document inference
│   │   ├── test_inference.py / test_postprocess.py
│   │   └── unannotated_inference.py / unannotated_postprocess.py / unannotated_review.py
│   └── analysis/
│       └── error_analysis.py       # false positive / false negative inspection
└── submissions/                    # generated predictions (csv/jsonl) — see below
```

All scripts resolve paths relative to the project root (via `PROJECT_ROOT` computed from
`__file__`), so the pipeline runs unmodified from any clone location.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA GPU (fine-tuning and inference use 4-bit quantization via
`bitsandbytes` + [Unsloth](https://github.com/unslothai/unsloth)).

## Running the Pipeline

All commands are run from the project root.

**1. Baseline (few-shot, no fine-tuning)**
```bash
python src/baseline/inference_span.py     # -> submissions/baseline_raw_predictions.jsonl
python src/baseline/postprocess_span.py   # -> submissions/baseline_val_predictions.csv
python src/baseline/evaluate_span.py      # token-level F1 on validation
```

**2. Fine-tuning**
```bash
python src/training/preprocess.py    # -> data/train.jsonl
python src/training/train_lora.py    # -> outputs/checkpoints/math_lora_model/final_lora_model

python src/training/val_inference.py    # -> submissions/val_raw_predictions.jsonl
python src/training/val_postprocess.py  # -> submissions/val_predictions.csv
python src/training/val_evaluate.py     # token-level F1 on validation
```

**3. Inference on new/unlabeled documents**
```bash
python src/inference/unannotated_inference.py    # -> submissions/unannotated_analysis_predictions.jsonl
python src/inference/unannotated_postprocess.py  # -> submissions/unannotated_predictions.json
python src/inference/unannotated_review.py       # -> submissions/unannotated_manual_review.txt (for error analysis)
```

## Limitations & Known Issues

- **Recall ceiling.** Even the fine-tuned model misses ~58% of true entity tokens (F1 0.43,
  recall 0.42) — long theorem/proof spans that run across paragraph or equation breaks are
  the most common miss.
- **Chunking loses context.** Documents are processed in overlapping character chunks;
  entities that span a chunk boundary can be truncated or duplicated.
- **Small training set.** Fine-tuning used a few hundred annotated spans across a handful of
  source documents — the model likely overfits to the surface style of those specific texts.
- **Manual review tooling** (`unannotated_review.py`) currently surfaces the *tag index*
  rather than the *tag name* in its report — a labeling bug in the review formatter, not in
  the model's predictions, but it made manual QA slower than it should have been.

## Tech Stack

`Qwen2.5-Math-7B-Instruct` · LoRA (PEFT) · [Unsloth](https://github.com/unslothai/unsloth)
· 4-bit quantization (`bitsandbytes`) · `transformers` / `trl` · `pandas`

---
*Originally built for a graduate NLP course assignment; restructured here as a standalone project.*
