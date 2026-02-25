import os
import json
import ast
import glob
from tqdm import tqdm
from unsloth import FastLanguageModel

MODEL_PATH = "/home/jkim829/hw2/outputs/checkpoints/math_lora_model/final_lora_model" 
UNANNOTATED_DIR = "/home/jkim829/hw2/data/unannotated_mmds"
OUTPUT_PATH = "/home/jkim829/hw2/submissions/unannotated_analysis_predictions.jsonl"

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
        if in_string and char == '\n': fixed_chars.append('\\n')
        elif in_string and char == '\r': fixed_chars.append('\\r')
        elif in_string and char == '\t': fixed_chars.append('\\t')
        else: fixed_chars.append(char)
    fixed_str = "".join(fixed_chars)
    try: return json.loads(fixed_str)
    except:
        try: return ast.literal_eval(fixed_str)
        except: return None

def main():
    print(f"🚀 Loading model for Unannotated Error Analysis")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH, 
        max_seq_length=2048, 
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    
    file_paths = glob.glob(os.path.join(UNANNOTATED_DIR, "*"))
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            print(f"\n📄 Analyzing: {filename}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            CHUNK_SIZE = 1500
            STRIDE = 800       
            
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
                            "filename": filename,
                            "tag": str(ent.get("tag", "")).strip(),
                            "text": str(ent.get("text", "")).strip()
                        }
                        out_f.write(json.dumps(raw_record, ensure_ascii=False) + '\n')
                        out_f.flush()

    print(f"\nAnalysis complete! Check {OUTPUT_PATH}")

if __name__ == "__main__":
    main()