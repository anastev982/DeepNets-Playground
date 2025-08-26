# DeepNets-Playground — Digits MLP Baseline

This project trains a simple MLP (neural network) on scikit-learn's built-in **Digits** dataset.

## How to run

### Option A: Conda
```bash
conda create -n deepnets python=3.11 -y
conda activate deepnets
pip install -r requirements.txt
python scripts/run_digits_mlp.py

## 📂 Project Structure

- **scripts/** – contains runnable experiments (`run_digits_mlp.py`)
- **data/** – placeholder for datasets
- **outputs/** – results (plots + metrics JSON)
- **notebooks/** – (optional) exploratory work
- **requirements.txt** – dependencies
- **README.md** – project description

### Breast Cancer Pipeline
A full data science lifecycle demo (Logistic Regression + Random Forest) on the Breast Cancer Wisconsin dataset.  
Notebook: [01_pipeline_breast_cancer.ipynb](case_studies/ds_lifecycle/01_pipeline_breast_cancer.ipynb)