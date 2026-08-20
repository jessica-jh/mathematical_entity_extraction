import json
import os
import pandas as pd
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONTENTS_PATH = os.path.join(DATA_DIR, "file_contents.json")
RAW_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "submissions", "baseline_raw_predictions.jsonl")
FINAL_CSV_PATH = os.path.join(PROJECT_ROOT, "submissions", "baseline_val_predictions.csv")

def find_ignoring_whitespace(window_text, extracted_text):
    text_no_space = re.sub(r'\s+', '', window_text)
    pattern_no_space = re.sub(r'\s+', '', extracted_text)
    idx = text_no_space.find(pattern_no_space)
    if idx == -1: return -1, -1
    mapping = [i for i, char in enumerate(window_text) if not char.isspace()]
    real_start = mapping[idx]
    real_end = mapping[idx + len(pattern_no_space) - 1] + 1
    return real_start, real_end

def main():
    print("Starting Merge Post-processing")
    with open(CONTENTS_PATH, 'r', encoding='utf-8') as f:
        contents = json.load(f)
    raw_preds = []
    with open(RAW_OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            raw_preds.append(json.loads(line))
    
    all_predictions = []
    for pred in raw_preds:
        fileid, extracted_text, tag, chunk_start = pred["fileid"], pred["text"].strip(), pred["tag"], pred["chunk_start"]
        if not extracted_text: continue
        text_content = contents[fileid]
        window_start, window_end = max(0, chunk_start - 500), min(len(text_content), chunk_start + 3500)
        window_text = text_content[window_start:window_end]
        
        local_idx = window_text.find(extracted_text)
        if local_idx != -1:
            real_start, real_end = window_start + local_idx, window_start + local_idx + len(extracted_text)
        else:
            rs, re_idx = find_ignoring_whitespace(window_text, extracted_text)
            real_start, real_end = (window_start + rs, window_start + re_idx) if rs != -1 else (-1, -1)

        if real_start != -1:
            all_predictions.append({"fileid": fileid, "start": real_start, "end": real_end, "tag": tag})


    df = pd.DataFrame(all_predictions).sort_values(by=['fileid', 'tag', 'start'])
    merged = []
    if not df.empty:
        for (fileid, tag), group in df.groupby(['fileid', 'tag']):
            curr_s, curr_e = -1, -1
            for _, row in group.iterrows():
                if curr_s == -1: 
                    curr_s, curr_e = row['start'], row['end']
                elif row['start'] <= curr_e: 
                    curr_e = max(curr_e, row['end']) 
                else:
                    merged.append({"fileid": fileid, "start": curr_s, "end": curr_e, "tag": tag})
                    curr_s, curr_e = row['start'], row['end']
            if curr_s != -1: 
                merged.append({"fileid": fileid, "start": curr_s, "end": curr_e, "tag": tag})

    # Filtering name and reference inside proof
    final_merged = []
    df_merged = pd.DataFrame(merged)
    
    if not df_merged.empty:
        for fileid, group in df_merged.groupby('fileid'):
            # Finding all proof intervals in the current file
            proofs = group[group['tag'] == 'proof']
            
            for _, row in group.iterrows():
                # Checking only name and reference
                if row['tag'] in ['name', 'reference']:
                    is_in_proof = False
                    for _, p_row in proofs.iterrows():
                        # Checking if the intervals overlap at all (max(start) < min(end))
                        if max(row['start'], p_row['start']) < min(row['end'], p_row['end']):
                            is_in_proof = True
                            break
                    
                    # If it is inside the proof, skip and don't add to the result list
                    if is_in_proof:
                        continue 
                        
                # Adding only the normal entities that don't meet the conditions
                final_merged.append(row.to_dict())
                
        final_df = pd.DataFrame(final_merged).sort_values(by=['fileid', 'start'])
    else:
        final_df = df_merged

    final_df.to_csv(FINAL_CSV_PATH, index=False)
    print(f"Smart merge & proof filtering completed! Output: {FINAL_CSV_PATH}")

if __name__ == "__main__":
    main()