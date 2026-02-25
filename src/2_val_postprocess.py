import json
import os
import pandas as pd
import re

DATA_DIR = "/home/jkim829/hw2/data"
CONTENTS_PATH = os.path.join(DATA_DIR, "file_contents.json")
RAW_OUTPUT_PATH = "/home/jkim829/hw2/submissions/val_raw_predictions.jsonl"
FINAL_CSV_PATH = "/home/jkim829/hw2/submissions/val_predictions.csv"

VALID_TAGS = {"definition", "theorem", "proof", "example", "name", "reference"}
BLOCK_TAGS = {"definition", "theorem", "proof", "example"}

MIN_LENGTHS = {
    "definition": 10, "theorem": 10, "proof": 10,
    "example": 10, "name": 3, "reference": 3
}

# Regex helpers (precision-friendly) 
NEXT_BLOCK_RE = re.compile(
    r'\n\s*(?:\*\*|_)*\s*(?:'
    r'Theorem|Lemma|Proposition|Corollary|Remark|Claim|Fact|'
    r'Definition|Notation|Convention|Example|Counterexample|Proof'
    r')\b',
    re.IGNORECASE
)

MERGE_STOP_RE = NEXT_BLOCK_RE  # same stop words are good merge boundaries
SECTION_RE = re.compile(r'\n\s*#')  # new section headers


def expand_block_end(text_content: str, real_end: int, hard_cap: int = 900) -> int:
    """
    Expand block span end, but stop early if another block header appears.
    This reduces token-level FP inflation.
    """
    if real_end >= len(text_content):
        return len(text_content)

    remainder = text_content[real_end: min(len(text_content), real_end + hard_cap)]

    # 1) Stop before next block header
    m = NEXT_BLOCK_RE.search(remainder)
    if m:
        return real_end + m.start()

    # 2) Otherwise stop at next paragraph break if present
    pb = remainder.find("\n\n")
    if pb != -1:
        return real_end + pb

    # 3) Fallback: cap expansion
    return real_end + len(remainder)


def safe_to_merge(text_content: str, curr_e: int, next_s: int) -> bool:
    """
    Only merge if the gap doesn't look like it contains a new block header / section start.
    """
    if curr_e >= next_s:
        return True
    gap = text_content[curr_e:next_s]
    if MERGE_STOP_RE.search(gap):
        return False
    if SECTION_RE.search(gap):
        return False
    return True


def find_aggressive_and_expand(window_text, extracted_text, tag, window_start, text_content):
    """
    1) Aggressive locate using whitespace-stripped prefix matching
    2) Convert to original indices
    3) If block tag, expand end carefully (stop at next block header)
    """
    clean_extracted = re.sub(r"\.{2,}$", "", extracted_text).strip()
    if not clean_extracted:
        return -1, -1

    text_no_space = re.sub(r"\s+", "", window_text)
    pattern_no_space = re.sub(r"\s+", "", clean_extracted)

    idx, matched_len = -1, 0
    for ratio in [1.0, 0.8, 0.6, 0.4]:
        search_pattern = pattern_no_space[:max(5, int(len(pattern_no_space) * ratio))]
        idx = text_no_space.find(search_pattern)
        if idx != -1:
            matched_len = len(search_pattern)
            break

    # Optional: two-end anchor fallback (helps recall without regex "sweeping")
    if idx == -1 and len(pattern_no_space) >= 40:
        prefix = pattern_no_space[:20]
        suffix = pattern_no_space[-20:]
        s_idx = text_no_space.find(prefix)
        e_idx = text_no_space.rfind(suffix)
        if s_idx != -1 and e_idx != -1 and s_idx <= e_idx:
            idx = s_idx
            matched_len = (e_idx - s_idx) + len(suffix)

    if idx == -1:
        return -1, -1

    mapping = [i for i, ch in enumerate(window_text) if not ch.isspace()]
    try:
        local_start = mapping[idx]
        local_end = mapping[idx + matched_len - 1] + 1
    except IndexError:
        return -1, -1

    real_start = window_start + local_start
    real_end = window_start + local_end

    if tag in BLOCK_TAGS:
        real_end = expand_block_end(text_content, real_end, hard_cap=900)

    return real_start, real_end


def main():
    print("Starting FINAL Recovery (precision-friendly) + Proof Filtering.")

    with open(CONTENTS_PATH, "r", encoding="utf-8") as f:
        contents = json.load(f)

    raw_preds = []
    with open(RAW_OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                raw_preds.append(json.loads(line))
            except Exception:
                continue

    all_predictions = []
    for pred in raw_preds:
        fileid = pred.get("fileid") or pred.get("filename")
        extracted_text = pred.get("text", "").strip()
        tag = str(pred.get("tag", "")).lower().strip()
        chunk_start = pred.get("chunk_start", 0)

        if tag not in VALID_TAGS:
            continue
        if len(extracted_text) < MIN_LENGTHS.get(tag, 3):
            continue

        text_content = contents.get(fileid, "")
        if not text_content:
            continue

        window_start = max(0, chunk_start - 500)
        window_end = min(len(text_content), chunk_start + 3500)
        window_text = text_content[window_start:window_end]

        real_start, real_end = find_aggressive_and_expand(
            window_text, extracted_text, tag, window_start, text_content
        )

        if real_start != -1 and real_end != -1 and real_end > real_start:
            all_predictions.append(
                {"fileid": fileid, "start": int(real_start), "end": int(real_end), "tag": tag}
            )

    df = pd.DataFrame(all_predictions)

    # Safe merge (avoid FP from over-merging)
    merged = []
    if not df.empty:
        df = df.drop_duplicates(subset=["fileid", "start", "end", "tag"])
        df = df.sort_values(by=["fileid", "tag", "start"])

        for (f_id, t_tag), group in df.groupby(["fileid", "tag"]):
            text_content = contents.get(f_id, "")
            curr_s, curr_e = -1, -1

            for _, row in group.iterrows():
                s, e = int(row["start"]), int(row["end"])
                if curr_s == -1:
                    curr_s, curr_e = s, e
                elif s <= curr_e + 250:
                    # merge only if safe
                    if safe_to_merge(text_content, curr_e, s):
                        curr_e = max(curr_e, e)
                    else:
                        merged.append({"fileid": f_id, "start": curr_s, "end": curr_e, "tag": t_tag})
                        curr_s, curr_e = s, e
                else:
                    merged.append({"fileid": f_id, "start": curr_s, "end": curr_e, "tag": t_tag})
                    curr_s, curr_e = s, e

            if curr_s != -1:
                merged.append({"fileid": f_id, "start": curr_s, "end": curr_e, "tag": t_tag})

    final_df = pd.DataFrame(merged)

    if not final_df.empty:
        print("🧹 Filtering (name/reference) inside proofs...")
        proofs = final_df[final_df["tag"] == "proof"]

        def is_inside_proof(row):
            # Keep proof itself
            if row["tag"] == "proof":
                return False

            # Precision-friendly: only remove name/reference inside proofs
            if row["tag"] not in {"name", "reference"}:
                return False

            mid = (row["start"] + row["end"]) / 2.0
            same_file_proofs = proofs[proofs["fileid"] == row["fileid"]]
            for _, p in same_file_proofs.iterrows():
                if p["start"] <= mid <= p["end"]:
                    return True
            return False

        final_df = final_df[~final_df.apply(is_inside_proof, axis=1)]

        final_df = final_df.sort_values(by=["fileid", "start"]).reset_index(drop=True)
        final_df.insert(0, "id", final_df.index)

        final_df.to_csv(FINAL_CSV_PATH, index=False)
        print(f"Saved: {FINAL_CSV_PATH}")
        print(f"Total entities: {len(final_df)}")
    else:
        print("No predictions generated (final_df is empty).")


if __name__ == "__main__":
    main()