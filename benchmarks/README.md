# scikit-learn reference benchmark

Reference results used in the IBEX Machine Learning Studio software paper
(matched comparison of IBEX with scikit-learn).

| File | Content |
|---|---|
| `benchmark_sklearn.py` | Benchmark script (regression, classification, clustering) |
| `results_sklearn_regression.csv` | Mean cross-validated R² and RMSE per algorithm and dataset |
| `results_sklearn_classification.csv` | Mean cross-validated accuracy and macro-F1 per algorithm and dataset |
| `results_sklearn_clustering.csv` | Silhouette score and adjusted Rand index per algorithm and dataset |

## Protocol

- Features are z-scored inside each training fold.
- 5-fold cross-validation (stratified for classification), seed 42.
- Hyperparameters follow the IBEX defaults wherever scikit-learn has an equivalent parameter.
- Software versions: scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, NumPy 2.5.3, Python 3.13.

## Run

```
pip install numpy pandas scikit-learn lightgbm xgboost
python benchmark_sklearn.py          # all tasks
```

The script writes the three `results_sklearn_*.csv` files in the current directory.
