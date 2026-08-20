import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


UNANNOTATED_DIR = os.path.join(PROJECT_ROOT, "data", "unannotated_mmds")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "submissions", "unannotated_analysis_predictions.jsonl")
REPORT_PATH = os.path.join(PROJECT_ROOT, "submissions", "unannotated_manual_review.txt")

def main():
    print("Generating Manual Review Report for Unannotated files!")
    
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"Error: {PREDICTIONS_PATH} not found.\nPlease run '4_unannotated_inference.py' first!")
        return

    preds_by_file = {}
    with open(PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            pred = json.loads(line)
            fname = pred['filename']
            if fname not in preds_by_file:
                preds_by_file[fname] = []
            preds_by_file[fname].append(pred)

    with open(REPORT_PATH, 'w', encoding='utf-8') as out_f:
        out_f.write("=== MANUAL ERROR ANALYSIS REPORT ===\n")
        out_f.write("Instructions: Please read the [Context] below and visually verify if the model correctly extracted the entity according to the [Tag].\n\n")

        for fname, preds in preds_by_file.items():
            file_path = os.path.join(UNANNOTATED_DIR, fname)
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                original_text = f.read()

            out_f.write(f"\n{'='*70}\n")
            out_f.write(f"📄 FILE: {fname}\n")
            out_f.write(f"{'='*70}\n")

            preds.sort(key=lambda x: x['tag'])

            for i, pred in enumerate(preds):
                tag = pred['tag']
                ext_text = pred['text']

                idx = original_text.find(ext_text)
                
                if idx != -1:
                    start_ctx = max(0, idx - 150)
                    end_ctx = min(len(original_text), idx + len(ext_text) + 150)
                    context = original_text[start_ctx:end_ctx].replace('\n', ' ')
                else:
                    context = "Context not found."

                clean_ext_text = ext_text.replace('\n', ' ')

                out_f.write(f"\n[{i+1}] TAG: {tag.upper()}\n")
                out_f.write(f"  Extracted Text: {clean_ext_text[:100]}...\n")
                out_f.write(f"  Surrounding Context: ...{context}...\n")
                out_f.write(f"  Evaluation: Is this genuinely a {tag} based on the context? (Mark 'O' if correct, otherwise 'False Positive!')\n")
                out_f.write(f"  {'-'*50}\n")

    print(f"✅ Review report generated successfully: {REPORT_PATH}")

if __name__ == "__main__":
    main()