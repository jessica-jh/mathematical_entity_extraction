import os
import json
import pandas as pd
import ast
from tqdm import tqdm
from unsloth import FastLanguageModel

# Calling baseline model (Qwen2.5-Math-7B-Instruct)
MODEL_NAME = "unsloth/Qwen2.5-Math-7B-Instruct" 
DATA_DIR = "/home/jkim829/hw2/data"
VAL_JSON_PATH = os.path.join(DATA_DIR, "val.json")
CONTENTS_PATH = os.path.join(DATA_DIR, "file_contents.json")

# Output file for baseline predictions
RAW_OUTPUT_PATH = "/home/jkim829/hw2/submissions/baseline_raw_predictions.jsonl"

# Few-shot prompt based on the assignment guidelines
few_shot_prompt = """### Instruction:
Extract all mathematical entities (definition, theorem, proof, example, name, reference) from the input. 

### Constraints:
- Output ONLY a valid JSON list of objects.
- Each object must have a "tag" and "text" field.
- DO NOT repeat the input or add any conversational text.

### Example 1 Input:
By S \\subset \\mathbb{N} having a least element, we mean that there exists an x \\in S such that for every y \\in S we have x \\le y.

### Example 1 Response:
[
  {"tag": "definition", "text": "By S \\subset \\mathbb{N} having a least element, we mean that there exists an x \\in S such that for every y \\in S we have x \\le y."},
  {"tag": "name", "text": "least element"}
]

### Example 2 Input:
Corollary 7.2. Let E be a finitely generated module and F a submodule. By lemma 5, then E is finitely generated.

### Example 2 Response:
[
  {"tag": "theorem", "text": "Corollary 7.2. Let E be a finitely generated module and F a submodule. By lemma 5, then E is finitely generated."},
  {"tag": "name", "text": "Corollary 7.2"},
  {"tag": "reference", "text": "finitely generated module"},
  {"tag": "reference", "text": "submodule"},
  {"tag": "reference", "text": "lemma 5"},
  {"tag": "reference", "text": "finitely generated"}
]

### Input:
{}

### Response:
[""" 

def extract_json_objects(text):
    objects = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, char in enumerate(text):
        if char == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if char == '{':
                if depth == 0: start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    objects.append(text[start:i+1])
                    start = -1
        escape = (char == '\\' and not escape)
    return objects

def robust_parse(json_str):
    fixed_chars = []
    in_string = False
    escape = False
    for char in json_str:
        if char == '"' and not escape:
            in_string = not in_string
        if in_string and char == '\n': fixed_chars.append('\\n')
        elif in_string and char == '\r': fixed_chars.append('\\r')
        elif in_string and char == '\t': fixed_chars.append('\\t')
        else: fixed_chars.append(char)
        escape = (char == '\\' and not escape)
    fixed_str = "".join(fixed_chars)
    try: return json.loads(fixed_str)
    except:
        try: return ast.literal_eval(fixed_str)
        except: return None

def main():
    print(f"Loading BASE model for Part 1 (Few-shot Baseline)")
    # Loading base model without LoRA weights
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_NAME, max_seq_length=2048, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    
    with open(CONTENTS_PATH, 'r', encoding='utf-8') as f:
        contents = json.load(f)
        
    val_df = pd.read_json(VAL_JSON_PATH)
    val_fileids = val_df['fileid'].unique().tolist()
    
    with open(RAW_OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for fileid in val_fileids:
            text = contents[fileid]
            print(f"\Generating Baseline for: {fileid}")
            
            CHUNK_SIZE = 1500  
            STRIDE = 750       
            
            for start_idx in tqdm(range(0, len(text), STRIDE)):
                chunk = text[start_idx:start_idx + CHUNK_SIZE]
                if not chunk.strip(): break
                
                messages = [{"role": "user", "content": few_shot_prompt.replace("{}", chunk)}]
                chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([chat_prompt], return_tensors="pt").to("cuda")
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1024, 
                    temperature=0.1,
                    eos_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
                
                full_res = "[" + response 
                found_objects = extract_json_objects(full_res)
                
                for obj_str in found_objects:
                    ent = robust_parse(obj_str) 
                    
                    if ent and 'tag' in ent and 'text' in ent:
                        raw_record = {
                            "fileid": fileid,
                            "chunk_start": start_idx, 
                            "tag": str(ent.get("tag", "")).strip(),
                            "text": str(ent.get("text", "")).strip()
                        }
                        out_f.write(json.dumps(raw_record, ensure_ascii=False) + '\n')
                        out_f.flush()

    print(f"Baseline predictions saved to {RAW_OUTPUT_PATH}")

if __name__ == "__main__":
    main()