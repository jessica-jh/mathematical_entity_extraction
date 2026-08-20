# MMEE: Mathematical Entity Extraction

Fine-tuning an LLM to find and label definitions, theorems, and proofs inside raw math text.

## What this is

Math textbooks have a clear structure when you read them: this paragraph is a definition,
this one states a theorem, this one proves it. But once you strip away formatting (for
example, when a PDF is converted to plain text or markdown), that structure disappears and
you're just left with paragraphs. This project tries to recover it automatically: given raw
text from a math document, find the spans of text that are definitions, theorems, proofs,
examples, names, or references, and label them.

```
{"fileid": "algebra_lang.mmd", "start": 1024, "end": 1310, "tag": "theorem"}
```

Why bother? If you can tag a math document this way, you can do things like search for "every
theorem about compact groups" across a library, or pull out clean (statement, proof) pairs to
train a math model on, instead of feeding it messy prose. Standard NER tools don't really work
here. They're not trained on math writing, and the spans you're looking for aren't single
words. A "proof" can run for several paragraphs and be full of LaTeX and references to earlier
results.

The main question I wanted to answer: does a math-specialized model actually need to be
fine-tuned for this, or can you get by with a good prompt? Short answer: fine-tuning matters a
lot here. See [Results](#results) below.

## Results

I tried this three ways, in order:

```
Token-level F1 on the validation set

BIO baseline (few-shot)    ▏ 0.000
Span baseline (few-shot)   ▏ 0.002
LoRA fine-tune             ████████████████████████████████████ 0.434
                            0.0        0.1        0.2        0.3        0.4
```

| Stage | Approach | Precision | Recall | F1 | Notes |
|---|---|:---:|:---:|:---:|---|
| 1 | Few-shot prompting, BIO-style output | 0.000 | 0.000 | 0.000 | The model couldn't reliably output valid token spans in this format. |
| 2 | Few-shot prompting, JSON span output | 0.174 | 0.001 | 0.002 | Better format, but it barely found any real spans. |
| 3 | LoRA fine-tune on Qwen2.5-Math-7B-Instruct | 0.451 | 0.418 | **0.434** | Big jump. The base model just hadn't seen this task before, and fine-tuning on a few hundred labeled examples was enough to fix that. |

(F1 here is token-level: for every character position, is it correctly labeled? This is
computed across all six tags on the validation set.)

### Breaking it down by entity type

The fine-tuned model isn't equally good at every tag:

| Tag | Precision | Recall | F1 | Support (chars) |
|---|:---:|:---:|:---:|:---:|
| definition | 0.454 | 0.503 | **0.477** | 4,755 |
| proof | 0.507 | 0.383 | 0.437 | 4,240 |
| theorem | 0.432 | 0.385 | 0.407 | 3,448 |
| example | 0.430 | 0.385 | 0.406 | 1,294 |
| name | 0.169 | 0.164 | 0.166 | 421 |

(`reference` doesn't appear at all in the validation set, so there's no score for it.)

The four long-form tags (`definition`, `proof`, `theorem`, `example`) all land around
0.40–0.48 F1. `name` is much worse, at 0.17. That makes sense: a name is often just one or
two words with little surrounding context to go on, so the model has a harder time knowing
exactly where it starts and ends, and even a small mistake in the boundary hurts the score a
lot more on a short span than a long one.

## How it works

1. **Baseline, no fine-tuning.** I tried two ways of prompting the base model:
   - *BIO*: split the text into tokens and ask the model for token-index spans per tag.
   - *Span extraction*: ask the model to directly output `{tag, start, end}` JSON.

   Span extraction worked better (the model could follow that format more reliably), but
   neither approach found real spans consistently.

2. **Fine-tuning.** I used [`Qwen2.5-Math-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct),
   a math-specialized 7B model, and fine-tuned it with LoRA (rank 16, on the attention and
   MLP projection layers) using [Unsloth](https://github.com/unslothai/unsloth) in 4-bit. The
   training data is formatted as: given a chunk of text, output a JSON list of
   `{tag, text}` entities.

3. **Turning predictions back into spans.** The model outputs the *text* of each entity, not
   its character position. A postprocessing step finds that text back in the source document
   to recover `start`/`end` offsets.

4. **Error analysis.** I also ran the fine-tuned model on documents it had never seen
   (no labels at all), and manually reviewed a sample of its predictions against the source
   text. What that turned up is in [Limitations](#limitations) below.

## Project layout

```
├── data/
│   ├── train.json, val.json        # labeled spans: (fileid, start, end, tag)
│   ├── file_contents.json          # full source text, keyed by fileid
│   └── unannotated_mmds/           # unlabeled documents used for error analysis
├── src/
│   ├── baseline/                   # few-shot prompting, BIO and span variants
│   │   ├── inference_bio.py / postprocess_bio.py / evaluate_bio.py
│   │   └── inference_span.py / postprocess_span.py / evaluate_span.py
│   ├── training/                   # LoRA fine-tuning + validation
│   │   ├── preprocess.py           # train.json + file_contents.json -> train.jsonl
│   │   ├── train_lora.py           # LoRA fine-tune on Qwen2.5-Math-7B-Instruct
│   │   ├── val_inference.py / val_postprocess.py / val_evaluate.py
│   ├── inference/                  # inference on the test set and unlabeled documents
│   │   ├── test_inference.py / test_postprocess.py
│   │   └── unannotated_inference.py / unannotated_postprocess.py / unannotated_review.py
│   └── analysis/
│       └── error_analysis.py       # inspect false positives / false negatives
└── submissions/                    # generated predictions (csv/jsonl)
```

Every script figures out the project root from its own file path, so the pipeline runs the
same way no matter where you clone it.

## Setup

```bash
pip install -r requirements.txt
```

You'll need Python 3.10+ and a CUDA GPU. Fine-tuning and inference both use 4-bit
quantization via `bitsandbytes` and [Unsloth](https://github.com/unslothai/unsloth).

## Running it

All commands assume you're in the project root.

**Baseline (no fine-tuning):**
```bash
python src/baseline/inference_span.py     # -> submissions/baseline_raw_predictions.jsonl
python src/baseline/postprocess_span.py   # -> submissions/baseline_val_predictions.csv
python src/baseline/evaluate_span.py      # prints token-level F1 on validation
```

**Fine-tuning:**
```bash
python src/training/preprocess.py    # -> data/train.jsonl
python src/training/train_lora.py    # -> outputs/checkpoints/math_lora_model/final_lora_model

python src/training/val_inference.py    # -> submissions/val_raw_predictions.jsonl
python src/training/val_postprocess.py  # -> submissions/val_predictions.csv
python src/training/val_evaluate.py     # prints token-level F1 on validation
```

**Running on new, unlabeled documents:**
```bash
python src/inference/unannotated_inference.py    # -> submissions/unannotated_analysis_predictions.jsonl
python src/inference/unannotated_postprocess.py  # -> submissions/unannotated_predictions.json
python src/inference/unannotated_review.py       # -> submissions/unannotated_manual_review.txt
```

## Limitations

- **It misses a lot.** Recall is 0.42, so more than half of the true entity text isn't caught.
  The most common miss is a long theorem or proof that runs across a paragraph or equation
  break. The model tends to stop early.
- **Chunking cuts things off.** Documents are split into overlapping chunks before being fed
  to the model, so an entity that happens to sit on a chunk boundary can get cut in half or
  counted twice.
- **Not much training data.** Fine-tuning used a few hundred labeled spans from a handful of
  textbooks, so the model has probably picked up some habits specific to those particular
  books rather than math writing in general.
- **A bug in the review tool.** `unannotated_review.py` prints the tag as a number instead of
  its name (e.g. `6` instead of `theorem`). That's a bug in the reporting script, not in the
  model's predictions, but it made manually checking the output more annoying than it needed
  to be.

## Built with

`Qwen2.5-Math-7B-Instruct` · LoRA (PEFT) · [Unsloth](https://github.com/unslothai/unsloth)
· 4-bit quantization (`bitsandbytes`) · `transformers` / `trl` · `pandas`
