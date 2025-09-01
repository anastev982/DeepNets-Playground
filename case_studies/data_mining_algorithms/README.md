# Principles and Concepts of Data Mining Algorithms

This notebook demonstrates supervised and unsupervised learning algorithms.

## Contents
- **Decision Tree** (classification, with GridSearch)
- **k-NN** (classification, with scaling in Pipeline + GridSearch)
- **Naive Bayes (GaussianNB)** (simple baseline)
- **K-Means** (clustering, with elbow method, silhouette score, PCA visualization)
- **Association Rules (Apriori)** (mini example with mlxtend)

## Dataset
By default: `sklearn.datasets.load_breast_cancer`  
Optionally: `sklearn.datasets.load_iris`

## Requirements
- Python 3.9+  
- scikit-learn  
- matplotlib  
- pandas  
- numpy  
- mlxtend

Install with:
```bash
pip install scikit-learn matplotlib pandas numpy mlxtend
