import os
import json
import pandas as pd
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONTENTS_PATH = os.path.join(DATA_DIR, "file_contents.json")

RAW_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "submissions", "baseline_bio_raw_predictions.jsonl")
FINAL_CSV_PATH = os.path.join(PROJECT_ROOT, "submissions", "baseline_val_predictions_bio.csv")

TAGS = ["definition", "theorem", "proof", "example", "name", "reference"]

def locate_tokens_in_chunk(chunk_text: str, tokens: list[str]):
    """
    Map each token to (start,end) char offsets within chunk_text using sequential find().
    Returns list of (s,e) or (None,None).
    """
    pos = []
    cursor = 0
    for tok in tokens:
        if not tok:
            pos.append((None, None))
            continue
        idx = chunk_text.find(tok, cursor)
        if idx == -1:
            idx = chunk_text.find(tok, max(0, cursor - 50))
        if idx == -1:
            pos.append((None, None))
            continue
        s = idx
        e = idx + len(tok)
        pos.append((s, e))
        cursor = e
    return pos

def merge_overlaps(df: pd.DataFrame):
    if df.empty:
        return df
    df = df.sort_values(by=["fileid", "tag", "start"]).reset_index(drop=True)
    merged = []
    for (fileid, tag), group in df.groupby(["fileid", "tag"], sort=False):
        curr_s, curr_e = None, None
        for _, row in group.iterrows():
            s, e = int(row["start"]), int(row["end"])
            if curr_s is None:
                curr_s, curr_e = s, e
            elif s <= curr_e:
                curr_e = max(curr_e, e)
            else:
                merged.append({"fileid": fileid, "start": curr_s, "end": curr_e, "tag": tag})
                curr_s, curr_e = s, e
        if curr_s is not None:
            merged.append({"fileid": fileid, "start": curr_s, "end": curr_e, "tag": tag})
    return pd.DataFrame(merged)

def filter_name_ref_inside_proof(df: pd.DataFrame):
    if df.empty:
        return df
    final_rows = []
    for fileid, group in df.groupby("fileid", sort=False):
        proofs = group[group["tag"] == "proof"][["start", "end"]].to_records(index=False)
        for _, row in group.iterrows():
            if row["tag"] in ["name", "reference"] and len(proofs) > 0:
                s, e = int(row["start"]), int(row["end"])
                in_proof = False
                for ps, pe in proofs:
                    if max(s, int(ps)) < min(e, int(pe)):
                        in_proof = True
                        break
                if in_proof:
                    continue
            final_rows.append(row.to_dict())
    return pd.DataFrame(final_rows)

def main():
    print("Starting BIO (token-span) post-processing")

    with open(CONTENTS_PATH, "r", encoding="utf-8") as f:
        contents = json.load(f)

    raw_records = []
    with open(RAW_OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            raw_records.append(json.loads(line))

    all_preds = []
    for rec in raw_records:
        fileid = rec["fileid"]
        chunk_start = int(rec["chunk_start"])
        chunk_size = int(rec.get("chunk_size", 1500))
        tokens = rec.get("tokens", [])
        spans = rec.get("spans", [])

        if not tokens or not spans:
            continue

        text_content = contents[fileid]
        chunk_text = text_content[chunk_start:chunk_start + chunk_size]
        token_pos = locate_tokens_in_chunk(chunk_text, tokens)

        for sp in spans:
            tag = sp.get("tag", "")
            if tag not in TAGS:
                continue
            try:
                start_i = int(sp.get("start_i"))
                end_i = int(sp.get("end_i"))
            except Exception:
                continue
            if not (0 <= start_i < end_i <= len(token_pos)):
                continue

            # token span -> char span (global)
            s0, _ = token_pos[start_i]
            _, e1 = token_pos[end_i - 1]
            if s0 is None or e1 is None:
                continue

            all_preds.append({
                "fileid": fileid,
                "start": chunk_start + s0,
                "end": chunk_start + e1,
                "tag": tag
            })

    df = pd.DataFrame(all_preds)

    # Always write a valid CSV (header-only if empty) so evaluator won't crash
    os.makedirs(os.path.dirname(FINAL_CSV_PATH), exist_ok=True)
    if df.empty:
        print("No predictions found; writing header-only CSV.")
        pd.DataFrame(columns=["fileid", "start", "end", "tag"]).to_csv(FINAL_CSV_PATH, index=False)
        print(f"Output: {FINAL_CSV_PATH}")
        return

    df_merged = merge_overlaps(df)
    df_filtered = filter_name_ref_inside_proof(df_merged)

    final_df = df_filtered.sort_values(by=["fileid", "start"]).reset_index(drop=True)
    final_df.to_csv(FINAL_CSV_PATH, index=False)
    print(f"BIO token-span merge & proof filtering completed! Output: {FINAL_CSV_PATH}")

if __name__ == "__main__":
    main()