import json
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "data"
OUTPUT_DIR = "data"  
CHUNK_SIZE = 1500       # Text chunk length (adjust based on your GPU memory)
OVERLAP = 200           # Length of the overlapping section to prevent cutting entities

def create_dataset(content_file, annotation_file, output_filename):
    print(f"Processing {annotation_file}...")
    
    # Load raw text contents
    with open(os.path.join(DATA_DIR, content_file), 'r', encoding='utf-8') as f:
        contents = json.load(f)
        
    # Load annotations using pandas 
    anno_path = os.path.join(DATA_DIR, annotation_file)
    df_anno = pd.read_json(anno_path)
    
    # 1. Group entities by document (fileid)
    doc_entities = defaultdict(list)
    for _, row in df_anno.iterrows():
        doc_entities[row['fileid']].append({
            "tag": row['tag'],
            "start": int(row['start']),
            "end": int(row['end']),
            "text": row['text']
        })
    
    dataset = []
    
    # 2. Apply sliding window to each document
    for fileid, text in tqdm(contents.items(), desc=f"Parsing {annotation_file}"):
        if fileid not in doc_entities:
            continue
            
        current_entities = doc_entities[fileid]
        text_len = len(text)
        
        # Sliding window loop
        start_idx = 0
        while start_idx < text_len:
            end_idx = min(start_idx + CHUNK_SIZE, text_len)
            chunk_text = text[start_idx:end_idx]
            
            # 3. Find entities completely contained within the current window
            chunk_entities = []
            for ent in current_entities:
                # Check if the entity is within the window's boundaries
                if ent['start'] >= start_idx and ent['end'] <= end_idx:
                    # IMPORTANT: Recalculate indices relative to the chunk
                    new_ent = {
                        "tag": ent['tag'],
                        "start": ent['start'] - start_idx,
                        "end": ent['end'] - start_idx,
                        "text": ent['text']
                    }
                    chunk_entities.append(new_ent)
            
            # Note: To reduce dataset size and speed up training, you can skip empty chunks:
            # if not chunk_entities: 
            #     start_idx += (CHUNK_SIZE - OVERLAP)
            #     continue
            
            # 4. Training format (Instruction tuning style)
            prompt = "Extract the following mathematical entities from the text: definition, theorem, proof, example, name, reference. Return the result as a JSON list of objects with 'tag', 'text', 'start', and 'end' keys."
            
            dataset.append({
                "instruction": prompt,
                "input": chunk_text,
                "output": json.dumps(chunk_entities, ensure_ascii=False)
            })
            
            # Break if we reached the end of the text
            if end_idx == text_len:
                break
                
            # Move the window forward, minus the overlap
            start_idx += (CHUNK_SIZE - OVERLAP)

    # 5. Save the dataset as JSONL
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ Saved {len(dataset)} samples to {out_path}")

if __name__ == "__main__":
    # Generate Train data
    create_dataset('file_contents.json', 'train.json', 'train.jsonl')
    
    # Generate Validation data (for evaluating F1 score later)
    create_dataset('file_contents.json', 'val.json', 'val.jsonl')