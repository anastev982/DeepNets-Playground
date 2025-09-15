This project shows a full ML workflow using scikit-learn and PySpark.
With scikit-learn, I built a regression model (Ridge) with feature engineering, tuning, and evaluation (RMSE, R², residual plots, permutation importance).
With PySpark, I turned the problem into a classification, added queries and feature engineering, and trained Logistic Regression with metrics like Accuracy, F1, ROC/AUC, and confusion matrix.
The goal was to practice all steps of the ML lifecycle — from data exploration to model evaluation — in both Python and a big data framework.

# DeepNets-Playground — Case Studies

This repository contains several **small case studies** in data science and machine learning:
- Digits MLP baseline (deep learning starter).
- Breast Cancer pipeline (full DS lifecycle).
- Sentiment Analysis (basic NLP workflow).

## Digits MLP Baseline

This project trains a simple MLP (neural network) on scikit-learn's built-in **Digits** dataset.

### How to run

#### Option A: Conda
```bash
conda create -n deepnets python=3.11 -y
conda activate deepnets
pip install -r requirements.txt
python scripts/run_digits_mlp.py
