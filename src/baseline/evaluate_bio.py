import argparse
import json
import os
from collections import defaultdict
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TAGS = ["definition", "theorem", "proof", "example", "name", "reference"]


def load_df(path: str):
    if path.endswith(".json"):
        return pd.read_json(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_intervals(df: pd.DataFrame):
    """Return dict[(fileid, tag)] -> list[(start,end)] with cleaned ints."""
    intervals = defaultdict(list)
    if df.empty:
        return intervals

    for _, row in df.iterrows():
        fileid = row["fileid"]
        tag = row["tag"]
        if tag not in TAGS:
            continue
        s = int(row["start"])
        e = int(row["end"])
        if e <= s:
            continue
        intervals[(fileid, tag)].append((s, e))
    return intervals


def merge_intervals(intervals):
    """Merge overlapping intervals; intervals is list[(s,e)]."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def interval_overlap_len(a, b):
    """Compute total overlap length between two merged interval lists."""
    i = j = 0
    overlap = 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if e > s:
            overlap += (e - s)
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return overlap


def total_len(intervals):
    return sum(e - s for s, e in intervals)


def evaluate_interval_based(gt_map, pr_map):
    """
    Interval-based token/character-level evaluation:
      TP = overlap length
      FP = predicted length - overlap
      FN = gold length - overlap
    Works on merged intervals.
    """
    # union of keys
    keys = set(gt_map.keys()) | set(pr_map.keys())

    total_tp = total_fp = total_fn = 0
    for key in keys:
        gt = merge_intervals(gt_map.get(key, []))
        pr = merge_intervals(pr_map.get(key, []))
        ov = interval_overlap_len(gt, pr)
        gt_len = total_len(gt)
        pr_len = total_len(pr)

        total_tp += ov
        total_fp += (pr_len - ov)
        total_fn += (gt_len - ov)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BIO baseline predictions (token/character-level).")
    parser.add_argument("--val_json", default=os.path.join(PROJECT_ROOT, "data", "val.json"))
    parser.add_argument("--pred_csv", default=os.path.join(PROJECT_ROOT, "submissions", "baseline_val_predictions_bio.csv"))
    parser.add_argument("--save_json", default="", help="Optional path to save metrics as JSON.")
    args = parser.parse_args()

    val_df = load_df(args.val_json)
    pred_df = load_df(args.pred_csv)

    # Normalize to interval maps
    gt_map = normalize_intervals(val_df)
    pr_map = normalize_intervals(pred_df)

    metrics = evaluate_interval_based(gt_map, pr_map)

    print("====== BIO Baseline Evaluation Results ======")
    print(f"TP (overlap chars) : {metrics['tp']}")
    print(f"FP (extra chars)   : {metrics['fp']}")
    print(f"FN (missed chars)  : {metrics['fn']}")
    print("-" * 44)
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print("============================================")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics JSON to: {args.save_json}")


if __name__ == "__main__":
    main()