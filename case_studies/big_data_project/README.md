 Big Data — Notebooks

This folder contains the *code notebooks* for the Big Data project.  
They complement the main pipeline by showing how different ML approaches were tested and compared.

 Contents

- *project2_sklearn_baseline.ipynb*

  A baseline workflow using **scikit-learn** (Logistic Regression, RandomForest).  
  Purpose: provide a lightweight reference implementation on smaller data.

- *project2_spark_end_to_end.ipynb*

  An end-to-end workflow using **PySpark MLlib** (Logistic Regression with parameter tuning and cross-validation).  
  Purpose: demonstrate how to scale the same pipeline to big data.

 Notes

- Input data is loaded from `data/raw_data/processed/`.
- Results (CSV, plots) are saved in the `outputs/` folder.
- Baseline vs Spark results can be directly compared to highlight scaling trade-offs.
