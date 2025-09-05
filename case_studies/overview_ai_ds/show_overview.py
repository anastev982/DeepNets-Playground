from pathlib import Path, json

OUT = Path("case_studies/overview_ai_ds/outputs")

print("\n--- Breast Cancer ---")
if (OUT/"metrics_breast_cancer.json").exists():
    print((OUT/"metrics_breast_cancer.json").read_text())

print("\n--- NLP (20news) ---")
if (OUT/"metrics_20news.json").exists():
    print((OUT/"metrics_20news.json").read_text())

print("\n--- Digits ---")
print("Run: python scripts/run_digits_mlp.py (results in outputs/)")

# (Optional) Just list images instead of displaying them
print("\nImages in outputs/:")
for p in OUT.glob("*.png"):
    print(" -", p.name)
