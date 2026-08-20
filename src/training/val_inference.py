import os
import json
import pandas as pd
import ast
from tqdm import tqdm
from unsloth import FastLanguageModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "math_lora_model", "final_lora_model") 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VAL_JSON_PATH = os.path.join(DATA_DIR, "val.json")
CONTENTS_PATH = os.path.join(DATA_DIR, "file_contents.json")
RAW_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "submissions", "val_raw_predictions.jsonl")

alpaca_prompt = """### Instruction:
Extract all mathematical entities (definition, theorem, proof, example, name, reference) from the input. 

### Constraints:
- Output ONLY a valid JSON list of objects.
- DO NOT repeat the input, instruction, or add any conversational text.
- Each 'text' must be exactly as it appears in the input.

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
            
        if in_string and char == '\n':
            fixed_chars.append('\\n')
        elif in_string and char == '\r':
            fixed_chars.append('\\r')
        elif in_string and char == '\t':
            fixed_chars.append('\\t')
        else:
            fixed_chars.append(char)
            
        escape = (char == '\\' and not escape)
        
    fixed_str = "".join(fixed_chars)
    try:
        return json.loads(fixed_str)
    except:
        try: return ast.literal_eval(fixed_str)
        except: return None

def main():
    print(f"Loading MATH LoRA weights for MAX Context Generation")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=MODEL_PATH, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    
    with open(CONTENTS_PATH, 'r', encoding='utf-8') as f:
        contents = json.load(f)
        
    val_df = pd.read_json(VAL_JSON_PATH)
    val_fileids = val_df['fileid'].unique().tolist()
    
    with open(RAW_OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for fileid in val_fileids:
            text = contents[fileid]
            print(f"\nGenerating for: {fileid}")
            
            # 긴 증명(Proof)이 잘리지 않도록 청크 사이즈 대폭 확대
            CHUNK_SIZE = 2500
            STRIDE = 1250  
            
            for start_idx in tqdm(range(0, len(text), STRIDE)):
                chunk = text[start_idx:start_idx + CHUNK_SIZE]
                if not chunk.strip(): break
                
                inputs = tokenizer([alpaca_prompt.format(chunk)], return_tensors="pt").to("cuda")
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=2048, 
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
                            "llm_start": int(ent.get("start", 0)), 
                            "tag": str(ent.get("tag", "")).strip(),
                            "text": str(ent.get("text", "")).strip()
                        }
                        out_f.write(json.dumps(raw_record, ensure_ascii=False) + '\n')
                        out_f.flush()

    print(f"Raw generations saved to {RAW_OUTPUT_PATH}")

if __name__ == "__main__":
    main()