import pandas as pd
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path setup (same as the other scripts)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VAL_JSON_PATH = os.path.join(DATA_DIR, "val.json")
CONTENTS_PATH = os.path.join(DATA_DIR, "file_contents.json")
PRED_CSV_PATH = os.path.join(PROJECT_ROOT, "submissions", "val_predictions.csv")

def main():
    print("Error Analysis Script Running\n")
    
    # Loading data
    val_df = pd.read_json(VAL_JSON_PATH)
    pred_df = pd.read_csv(PRED_CSV_PATH)
    
    with open(CONTENTS_PATH, 'r', encoding='utf-8') as f:
        contents = json.load(f)
        
    fileids = val_df['fileid'].unique()
    
    fp_examples = []
    fn_examples = []
    
    for fileid in fileids:
        text = contents[fileid]
        
        true_subset = val_df[val_df['fileid'] == fileid]
        pred_subset = pred_df[pred_df['fileid'] == fileid]
        
        # Extracting true and predicted texts
        true_texts = [(row['tag'], text[int(row['start']):int(row['end'])]) for _, row in true_subset.iterrows()]
        pred_texts = [(row['tag'], text[int(row['start']):int(row['end'])]) for _, row in pred_subset.iterrows()]
        
        # 1. False Positive (model wrongly found)
        for tag, p_text in pred_texts:
            # If the predicted text is not included in the true text, consider it as an error
            if not any(p_text in t_text or t_text in p_text for t_tag, t_text in true_texts if t_tag == tag):
                fp_examples.append((tag, p_text))
                
        # 2. False Negative (model missed)
        for tag, t_text in true_texts:
            # If the true text is not included in the predicted text, consider it as an error
            if not any(t_text in p_text or p_text in t_text for p_tag, p_text in pred_texts if p_tag == tag):
                fn_examples.append((tag, t_text))

    # Output results
    print("[False Positive examples] Model wrongly found (answer is not the answer)")
    for i, (tag, txt) in enumerate(fp_examples[:5]): # Output 5 samples
        # Clean up line breaks
        clean_txt = txt.replace('\n', ' ')
        print(f"  {i+1}. Tag: [{tag}]\n     Text: {clean_txt[:150]}...\n")
        
    print("[False Negative examples] Model missed (obvious answer but not found)")
    for i, (tag, txt) in enumerate(fn_examples[:5]):
        clean_txt = txt.replace('\n', ' ')
        print(f"  {i+1}. 태그: [{tag}]\n     텍스트: {clean_txt[:150]}...\n")

    print("Analysis guide: Think about why the model got confused by the above examples and answer questions 1 and 2 in the report!")

if __name__ == "__main__":
    main()