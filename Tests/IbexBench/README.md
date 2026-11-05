# IbexBench - run all IBEX algorithms headlessly

This folder runs the **real IBEX engines** (the same `ML/*.cs` files the app uses, linked not copied)
on the benchmark datasets and writes `results_ibex.csv`. `compare_results.py` then merges that file
with the scikit-learn results and produces the tables for the paper.

## Location
This folder lives at `Tests/IbexBench/` in the repository (`MLStudio.csproj` excludes `Tests/**`, so the app build is unaffected).
The scikit-learn reference script and its results are in `sklearn/`; the benchmark datasets are in `data/`.

## Run (Windows, .NET 8 SDK installed)
```
cd Tests/IbexBench
dotnet run -c Release -- data results_ibex.csv
```
Takes a few minutes (Digits + gradient-boosted classifiers are the slow part).
The console prints one line per run, and at the end the functional-coverage count
("N OK, M failed") that the paper needs.

## Then compare (needs Python + pandas + tabulate)
```
python compare_results.py sklearn results_ibex.csv comparison_tables.md
```

## What the harness does (so you can describe it in the paper)
- Linear regression (none/L1/L2): `LinearRegressionEngine.CrossValidate`, 5 folds, feature normalisation on,
  two configurations: application defaults (100 iterations, lr 1e-3) and "converged" (3000 iterations, lr 1e-2).
  IBEX fits linear models by gradient descent, scikit-learn by closed form, so only the converged
  configuration is expected to match.
- Polynomial (degree 2): `PolynomialRegressionEngine.CrossValidate`, lr 0.05, 3000 iterations.
- The other nine regression models: same fold logic as the application (`DatasetBuilder.Shuffle`, seed 42,
  contiguous folds) with the slider default hyperparameters; features and target are z-scored first.
- Classification: `ClassificationEngine.CrossValidate`, application defaults, 5 folds.
- Clustering: `ClusteringEngine.Run`, k = number of true classes, seed 42, DBSCAN eps 1.5, average linkage.
- Every run also trains a final model and makes one prediction; a run counts as "OK" only if all steps succeed.

## Known differences between the two sides (write them in the paper)
- Folds: IBEX shuffles with its own seeded generator, so fold membership differs from scikit-learn's `KFold`.
  Compare mean scores, not fold by fold.
- Scaling: IBEX standardises with statistics from the whole dataset; the scikit-learn script uses training-fold statistics.
- IBEX SVM is a linear one-vs-rest SVM; logistic regression is one-vs-rest gradient descent.
- IBEX FastTree has no scikit-learn counterpart (gradient boosting is the nearest equivalent).
