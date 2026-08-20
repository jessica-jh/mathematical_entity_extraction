import json
import os
import pandas as pd
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Unannotated Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

UNANNOTATED_DIR = os.path.join(DATA_DIR, "unannotated_mmds")

RAW_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "submissions", "unannotated_analysis_predictions.jsonl")
FINAL_JSON_PATH = os.path.join(PROJECT_ROOT, "submissions", "unannotated_predictions.json")

VALID_TAGS = {"definition", "theorem", "proof", "example", "name", "reference"}
BLOCK_TAGS = {"definition", "theorem", "proof", "example"}

MIN_LENGTHS = {
    "definition": 10, "theorem": 10, "proof": 10,
    "example": 10, "name": 3, "reference": 3
}

NEXT_BLOCK_RE = re.compile(
    r'\n\s*(?:\*\*|_)*\s*(?:'
    r'Theorem|Lemma|Proposition|Corollary|Remark|Claim|Fact|'
    r'Definition|Notation|Convention|Example|Counterexample|Proof'
    r')\b',
    re.IGNORECASE
)

MERGE_STOP_RE = NEXT_BLOCK_RE 
SECTION_RE = re.compile(r'\n\s*#') 

def expand_block_end(text_content: str, real_end: int, hard_cap: int = 900) -> int:
    if real_end >= len(text_content): return len(text_content)
    remainder = text_content[real_end: min(len(text_content), real_end + hard_cap)]
    m = NEXT_BLOCK_RE.search(remainder)
    if m: return real_end + m.start()
    pb = remainder.find("\n\n")
    if pb != -1: return real_end + pb
    return real_end + len(remainder)

def safe_to_merge(text_content: str, curr_e: int, next_s: int) -> bool:
    if curr_e >= next_s: return True
    gap = text_content[curr_e:next_s]
    if MERGE_STOP_RE.search(gap) or SECTION_RE.search(gap): return False
    return True

def find_aggressive_and_expand(window_text, extracted_text, tag, window_start, text_content):
    clean_extracted = re.sub(r"\.{2,}$", "", extracted_text).strip()
    if not clean_extracted: return -1, -1

    text_no_space = re.sub(r"\s+", "", window_text)
    pattern_no_space = re.sub(r"\s+", "", clean_extracted)

    idx, matched_len = -1, 0
    for ratio in [1.0, 0.8, 0.6, 0.4]:
        search_pattern = pattern_no_space[:max(5, int(len(pattern_no_space) * ratio))]
        idx = text_no_space.find(search_pattern)
        if idx != -1:
            matched_len = len(search_pattern)
            break

    if idx == -1 and len(pattern_no_space) >= 40:
        prefix = pattern_no_space[:20]
        suffix = pattern_no_space[-20:]
        s_idx = text_no_space.find(prefix)
        e_idx = text_no_space.rfind(suffix)
        if s_idx != -1 and e_idx != -1 and s_idx <= e_idx:
            idx = s_idx
            matched_len = (e_idx - s_idx) + len(suffix)

    if idx == -1: return -1, -1

    mapping = [i for i, ch in enumerate(window_text) if not ch.isspace()]
    try:
        local_start = mapping[idx]
        local_end = mapping[idx + matched_len - 1] + 1
    except IndexError:
        return -1, -1

    real_start, real_end = window_start + local_start, window_start + local_end

    if tag in BLOCK_TAGS:
        real_end = expand_block_end(text_content, real_end, hard_cap=900)

    return real_start, real_end

def main():
    print("Starting FINAL Recovery for Unannotated Data.")

    raw_preds = []
    if not os.path.exists(RAW_OUTPUT_PATH):
        print(f"Error: Cannot find {RAW_OUTPUT_PATH}. Run inference first!")
        return

    with open(RAW_OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try: raw_preds.append(json.loads(line))
            except Exception: continue

    all_predictions = []
    for pred in raw_preds:
        fileid = pred.get("fileid") or pred.get("filename")
        extracted_text = pred.get("text", "").strip()
        tag = str(pred.get("tag", "")).lower().strip()

        if tag not in VALID_TAGS or len(extracted_text) < MIN_LENGTHS.get(tag, 3): continue
        
        file_path = os.path.join(UNANNOTATED_DIR, fileid)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
            
        if not text_content: continue

        window_start = 0
        window_text = text_content

        real_start, real_end = find_aggressive_and_expand(
            window_text, extracted_text, tag, window_start, text_content
        )

        if real_start != -1 and real_end != -1 and real_end > real_start:
            all_predictions.append(
                {"fileid": fileid, "start": int(real_start), "end": int(real_end), "tag": tag}
            )

    df = pd.DataFrame(all_predictions)
    merged = []
    if not df.empty:
        df = df.drop_duplicates(subset=["fileid", "start", "end", "tag"])
        df = df.sort_values(by=["fileid", "tag", "start"])

        for (f_id, t_tag), group in df.groupby(["fileid", "tag"]):
            # 병합을 위해 여기서도 파일을 다시 읽어옵니다.
            file_path = os.path.join(UNANNOTATED_DIR, f_id)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            else:
                text_content = ""
                
            curr_s, curr_e = -1, -1
            for _, row in group.iterrows():
                s, e = int(row["start"]), int(row["end"])
                if curr_s == -1: curr_s, curr_e = s, e
                elif s <= curr_e + 250:
                    if safe_to_merge(text_content, curr_e, s): curr_e = max(curr_e, e)
                    else:
                        merged.append({"fileid": f_id, "start": curr_s, "end": curr_e, "tag": t_tag})
                        curr_s, curr_e = s, e
                else:
                    merged.append({"fileid": f_id, "start": curr_s, "end": curr_e, "tag": t_tag})
                    curr_s, curr_e = s, e
            if curr_s != -1: merged.append({"fileid": f_id, "start": curr_s, "end": curr_e, "tag": t_tag})

    final_df = pd.DataFrame(merged)
    if not final_df.empty:
        proofs = final_df[final_df["tag"] == "proof"]
        def is_inside_proof(row):
            if row["tag"] not in {"name", "reference"}: return False
            mid = (row["start"] + row["end"]) / 2.0
            for _, p in proofs[proofs["fileid"] == row["fileid"]].iterrows():
                if p["start"] <= mid <= p["end"]: return True
            return False

        final_df = final_df[~final_df.apply(is_inside_proof, axis=1)]
        final_df = final_df.sort_values(by=["fileid", "start"]).reset_index(drop=True)
        
        final_df.to_json(FINAL_JSON_PATH, orient="records", force_ascii=False)

        csv_path = FINAL_JSON_PATH.replace(".json", ".csv")
        final_df.to_csv(csv_path, index=False)
        print(f"Saved: {FINAL_JSON_PATH}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main()