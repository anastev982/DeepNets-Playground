from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SRC_BC  = ROOT / "case_studies" / "ds_lifecycle" / "outputs"
SRC_NLP = ROOT / "case_studies" / "nlp_sentiment" / "outputs"
DEST    = ROOT / "case_studies" / "overview_ai_ds" / "outputs"

DEST.mkdir(parents=True, exist_ok=True)

def copy_all(src: Path, prefix: str):
    if not src.exists():
        print("Skip (not found):", src)
        return
    for f in src.glob("*"):
        if f.is_file():
            # add prefix to avoid name clashes
            target = DEST / f"{prefix}_{f.name}"
            shutil.copy(f, target)
            print("Copied:", target.name)

copy_all(SRC_BC,  "bc")
copy_all(SRC_NLP, "nlp")

print("\nAll results collected in:", DEST.resolve())
