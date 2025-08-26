# DeepNets-Playground — Case Studies

This repository contains several **small case studies** in data science and machine learning:
- Digits MLP baseline (deep learning starter).
- Breast Cancer pipeline (full DS lifecycle).
- Sentiment Analysis (basic NLP workflow).

---

## Digits MLP Baseline

This project trains a simple MLP (neural network) on scikit-learn's built-in **Digits** dataset.

### How to run

#### Option A: Conda
```bash
conda create -n deepnets python=3.11 -y
conda activate deepnets
pip install -r requirements.txt
python scripts/run_digits_mlp.py
