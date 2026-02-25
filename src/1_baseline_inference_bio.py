import os
import json
import ast
import re
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel

# Base model
MODEL_NAME = "unsloth/Qwen2.5-Math-7B-Instruct"

DATA_DIR = "/home/jkim829/hw2/data"
VAL_JSON_PATH = os.path.join(DATA_DIR, "val.json")
CONTENTS_PATH = os.path.join(DATA_DIR, "file_contents.json")

# Output file for BIO baseline predictions (raw per-chunk)
RAW_OUTPUT_PATH = "/home/jkim829/hw2/submissions/baseline_bio_raw_predictions.jsonl"

TAGS = ["definition", "theorem", "proof", "example", "name", "reference"]

# IMPORTANT: We output token-index spans (start_i inclusive, end_i exclusive).
# This is still "token-level labeling" but much more stable than emitting a label for every token.
FEW_SHOT_PROMPT = f"""### Instruction:
You are given whitespace-delimited tokens from a math text. Identify spans corresponding to these tags:
{TAGS}

### Output format (STRICT):
Output ONLY valid JSON: a list of span objects.
Each span object must be:
  {{"tag": "<one of {TAGS}>", "start_i": <int>, "end_i": <int>}}
where start_i is inclusive and end_i is exclusive (Python slicing),
and spans refer to token indices in the provided list.

### Constraints:
- Use ONLY the allowed tags.
- Do NOT output token text, only indices.
- Do NOT include any extra keys or any commentary.

### Example:
Tokens:
0: By
1: S
2: \\subset
3: \\mathbb{{N}}
4: having
5: a
6: least
7: element,
8: we
9: mean
10: that
11: ...

Output:
[
  {{"tag":"definition", "start_i": 0, "end_i": 12}},
  {{"tag":"name", "start_i": 6, "end_i": 8}}
]

### Now do it for the following input.

Tokens:
{{TOKENS_BLOCK}}

Output:
"""

def robust_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

def extract_json_array(text: str):
    """
    Extract the first top-level JSON array substring from model output.
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if c == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        escape = (c == "\\" and not escape)
    return None

def tokenize_whitespace(text: str):
    return re.findall(r"\S+", text)

def build_tokens_block(tokens):
    # Keep prompt size stable
    MAX_TOKENS = 500
    tokens = tokens[:MAX_TOKENS]
    lines = [f"{i}: {tok}" for i, tok in enumerate(tokens)]
    return "\n".join(lines), len(tokens)

def main():
    print("Loading BASE model for Part 1 (Few-shot BIO token-span baseline)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    with open(CONTENTS_PATH, "r", encoding="utf-8") as f:
        contents = json.load(f)

    val_df = pd.read_json(VAL_JSON_PATH)
    val_fileids = val_df["fileid"].unique().tolist()

    CHUNK_SIZE = 1500
    STRIDE = 750

    os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)

    with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for fileid in val_fileids:
            text = contents[fileid]
            print(f"Generating BIO token-span baseline for: {fileid}")

            for start_idx in tqdm(range(0, len(text), STRIDE)):
                chunk = text[start_idx:start_idx + CHUNK_SIZE]
                if not chunk.strip():
                    break

                tokens = tokenize_whitespace(chunk)
                tokens_block, n_tokens = build_tokens_block(tokens)

                prompt = FEW_SHOT_PROMPT.replace("{TOKENS_BLOCK}", tokens_block)

                messages = [{"role": "user", "content": prompt}]
                chat_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer([chat_prompt], return_tensors="pt").to("cuda")

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=600,      # spans only -> short output
                    temperature=0.0,         # reduce formatting drift
                    eos_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0]

                arr_str = extract_json_array(response)
                parsed = robust_parse_json(arr_str) if arr_str else None

                spans = []
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        tag = str(item.get("tag", "")).strip()
                        if tag not in TAGS:
                            continue
                        try:
                            s = int(item.get("start_i"))
                            e = int(item.get("end_i"))
                        except Exception:
                            continue
                        if 0 <= s < e <= n_tokens:
                            spans.append({"tag": tag, "start_i": s, "end_i": e})

                raw_record = {
                    "fileid": fileid,
                    "chunk_start": start_idx,
                    "chunk_size": CHUNK_SIZE,
                    "tokens": tokens[:n_tokens],
                    "spans": spans,
                    # Debug fields (keep for troubleshooting; OK to remove later)
                    "extracted_json": arr_str or "",
                    "raw_response_head": response[:2000],
                }

                out_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"BIO token-span raw predictions saved to {RAW_OUTPUT_PATH}")

if __name__ == "__main__":
    main()