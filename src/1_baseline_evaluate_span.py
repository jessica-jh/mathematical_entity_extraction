import pandas as pd
import json
import os

# Define paths
DATA_DIR = "/home/jkim829/hw2/data"
VAL_JSON_PATH = os.path.join(DATA_DIR, "val.json")
PRED_CSV_PATH = "/home/jkim829/hw2/submissions/baseline_val_predictions.csv"

def main():
    # Load ground truth and predictions
    val_df = pd.read_json(VAL_JSON_PATH)
    pred_df = pd.read_csv(PRED_CSV_PATH)
    
    # Get all unique fileids and tags
    files = set(val_df['fileid']).union(set(pred_df['fileid']))
    tags = ['definition', 'theorem', 'proof', 'example', 'name', 'reference']
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for fileid in files:
        for tag in tags:
            # Create a set of true character indices for this file and tag
            true_intervals = val_df[(val_df['fileid'] == fileid) & (val_df['tag'] == tag)]
            true_indices = set()
            for _, row in true_intervals.iterrows():
                true_indices.update(range(int(row['start']), int(row['end'])))
                
            # Create a set of predicted character indices for this file and tag
            pred_intervals = pred_df[(pred_df['fileid'] == fileid) & (pred_df['tag'] == tag)]
            pred_indices = set()
            for _, row in pred_intervals.iterrows():
                pred_indices.update(range(int(row['start']), int(row['end'])))
                
            # Calculate token/character-level overlaps
            tp = len(true_indices.intersection(pred_indices))
            fp = len(pred_indices - true_indices)
            fn = len(true_indices - pred_indices)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
    # Calculate final metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("====== Token-Level Evaluation Results ======")
    print(f"Total True Positive tokens : {total_tp}")
    print(f"Total False Positive tokens: {total_fp}")
    print(f"Total False Negative tokens: {total_fn}")
    print("-" * 44)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1_score:.4f}")
    print("============================================")

if __name__ == "__main__":
    main()