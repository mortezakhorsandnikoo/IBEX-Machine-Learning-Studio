# IBEX Machine Learning Studio: an open-source, no-code desktop environment for teaching and applying machine learning in architectural design

**Morteza Khorsand Nikoo**
Faculty of Architecture, Lightweight Structures Unit, University of Innsbruck, Austria
Correspondence: morteza.khorsandnikoo@uibk.ac.at

*Draft v0.1, 2 November 2026. Items marked **[TO FILL]** need your own data or confirmation before posting.*

---

## Abstract

Architects and design students increasingly need machine learning (ML), but most ML environments assume programming skills, and no-code tools are rarely open, locally executable and aimed at design practice. We present **IBEX Machine Learning Studio**, a free, open-source (GPL) Windows desktop application that provides a complete supervised and unsupervised learning workflow without writing code: dataset import, descriptive statistics, missing-value handling, normalisation, PCA, model training with adjustable hyperparameters, k-fold cross-validation, interactive visualisation and manual prediction. IBEX bundles 11 regression, 9 classification and 4 clustering algorithms and runs entirely offline. A companion Grasshopper plug-in, **Dendrite**, brings the same functionality into parametric design workflows in Rhino. We describe the architecture and feature set, and benchmark IBEX against scikit-learn on standard regression, classification and clustering datasets under an identical protocol (same folds, seed and preprocessing), reporting accuracy, run time and usability. The software is archived on Zenodo (DOI 10.5281/zenodo.23136448) and distributed through the Microsoft Store, so that researchers and educators can cite and reuse it.

**Keywords:** machine learning; no-code; architecture education; parametric design; Grasshopper; open-source software; scikit-learn

---

## 1. Motivation and significance

ML is entering architectural practice (performance prediction, generative design, surrogate modelling), yet architecture students rarely have programming or statistics backgrounds. Existing options fall into two groups:

- **Code libraries** (e.g. scikit-learn) are powerful and reproducible, but require Python and an understanding of the API.
- **GUI tools** (e.g. Weka, Orange, cloud ML studios) lower the barrier, but are often general-purpose, cloud-dependent, or disconnected from design tools such as Rhino/Grasshopper.

IBEX targets the gap: a **local, no-code, teaching-oriented** studio where every step of the ML workflow is visible (data statistics, preprocessing, hyperparameters, train/test and cross-validated metrics) and where the same models can be used inside a parametric design environment. It is in active use to teach ML to architecture students with no prior ML background. **[TO FILL: number of students/courses, semester]**

## 2. Software description

### 2.1 Implementation

IBEX is written in C# with WPF on .NET 8 (Windows x64). Numerical and ML components rely on established libraries: MathNet.Numerics (linear algebra), ML.NET (FastTree, LightGBM, online gradient descent), XGBoostSharp (XGBoost) and LiveCharts2 (interactive charts). Several algorithms (normal equation, locally weighted regression, random forest, MLP, RBF surrogate, k-NN, naive Bayes, clustering) are implemented directly in the code base. Input is CSV; models and preprocessing "recipes" can be saved and reloaded so that predictions on new data reuse exactly the same transformations.

### 2.2 Workflow and features

| Stage | Capabilities (from the source code) |
|---|---|
| Data | CSV import; summary statistics (count, mean, SD, min, quartiles, median, max, skewness, kurtosis, missing count); correlation matrix |
| Preprocessing | Missing values: mean / max / min / mode / remove row. Normalisation: z-score, min-max, max-abs. PCA with explained and cumulative variance, reconstruction; predictions accepted in original feature units |
| Regression (11) | Linear (none / L1 / L2 regularisation), polynomial, normal equation (ridge), locally weighted regression, random forest, MLP (ReLU/tanh/sigmoid; SGD/Adam), RBF surrogate, FastTree, LightGBM, online gradient descent, XGBoost |
| Classification (9) | Logistic regression, k-NN, naive Bayes, SVM, random forest, FastTree, LightGBM, XGBoost, ANN |
| Clustering (4) | K-means, DBSCAN, hierarchical (single/average/complete linkage), fuzzy c-means; silhouette score and 2-D PCA projection |
| Evaluation | Train/test split; k-fold cross-validation with per-fold scores and a combined score computed from pooled predictions; confusion matrix, per-class precision/recall/F1, macro-F1; validation curves over a primary hyperparameter |
| Usability | Hyperparameter sliders with sensible ranges; manual prediction; chart/table export; offline searchable help; printable ML cheat sheet |

### 2.3 Grasshopper plug-in (Dendrite)

Dendrite exposes data preprocessing, regression/classification, feature evaluation and visualisation as Grasshopper components so that ML models can be driven by, and drive, parametric geometry. It is distributed free through Food4Rhino (first listed 17 August 2025). **[TO FILL: Dendrite version, supported Rhino/Grasshopper versions, which IBEX algorithms are included, and how models trained in IBEX are exchanged with Dendrite.]**

### 2.4 Availability

- Source code: <https://github.com/mortezakhorsandnikoo/IBEX-Machine-Learning-Studio>
- Archive: Zenodo, DOI 10.5281/zenodo.23136448 (v1.0.1)
- Microsoft Store: IBEX Machine Learning Studio **[TO FILL: store URL]**
- Licence: GNU GPL **[TO FIX: README says "GPL-3.0-or-later", Zenodo says "GPL-3.0-only"; make these consistent]**

## 3. Benchmark against scikit-learn

### 3.1 Protocol

To make results comparable and reproducible, both tools use the same data, the same folds and the same preprocessing:

- **Cross-validation:** 5-fold (stratified for classification), shuffle with seed 42; features z-scored using training-fold statistics only; for regression the target is z-scored on the training fold and predictions are back-transformed (as in IBEX's target recipe).
- **Metrics:** R² and RMSE (regression); accuracy and macro-F1 (classification); silhouette and adjusted Rand index (clustering); wall-clock fit and predict time.
- **Hyperparameters:** IBEX slider defaults, mapped to the nearest scikit-learn / LightGBM / XGBoost parameters (Table 2).
- **Datasets:** Regression: Diabetes, Friedman #1 (n=500), synthetic linear (n=400). Classification and clustering: Iris, Wine, Breast Cancer, Digits (clustering excludes Digits). **[Optionally add OpenML datasets and one architectural dataset, e.g. daylight, structural or energy data.]**
- **Software versions:** scikit-learn 1.9.1, Python 3.13, LightGBM and XGBoost latest at run time **[TO FILL: IBEX version, CPU, RAM]**.

All scripts and raw outputs are provided with the paper (`benchmark_sklearn.py`, `results_sklearn_*.csv`).

**Table 2. Algorithm mapping.** IBEX Linear (L1/L2/none) ↔ `Lasso`/`Ridge`/`LinearRegression`; Polynomial ↔ degree-2 polynomial features + `Ridge`; Normal equation ↔ `LinearRegression`; Random forest ↔ `RandomForest*` (50 trees, depth 12); MLP/ANN ↔ `MLP*` (1×32 hidden); RBF surrogate ↔ `KernelRidge` (RBF); FastTree ↔ `GradientBoosting*` (closest equivalent, not identical); LightGBM, XGBoost ↔ same libraries; OGD ↔ `SGDRegressor`; k-NN, naive Bayes, SVM, logistic regression ↔ same-named estimators; K-means, DBSCAN, hierarchical ↔ same-named estimators. *No scikit-learn equivalent* exists for locally weighted regression and fuzzy c-means, so these are verified against reference implementations or analytic cases instead **[TO FILL]**.

### 3.2 Results: scikit-learn reference (completed)

**Table 3. Regression, mean cross-validated R² (5-fold).**

| Algorithm | Diabetes | Friedman #1 | Synthetic linear | IBEX D / F / S |
|---|---|---|---|---|
| Linear (no reg.) | 0.478 | 0.731 | 0.996 | **[TO FILL]** |
| Linear (L1) | 0.480 | 0.731 | 0.995 | |
| Linear (L2) | 0.479 | 0.731 | 0.996 | |
| Polynomial (deg 2) | 0.368 | 0.859 | 0.995 | |
| Normal equation | 0.478 | 0.731 | 0.996 | |
| Random forest | 0.449 | 0.744 | 0.714 | |
| MLP | 0.463 | 0.792 | 0.971 | |
| RBF surrogate | 0.171 | 0.050 | 0.249 | |
| FastTree-like (GBM) | 0.378 | 0.858 | 0.903 | |
| LightGBM | 0.348 | 0.872 | 0.907 | |
| Online GD | 0.481 | 0.731 | 0.996 | |
| XGBoost | 0.314 | 0.827 | 0.859 | |

*Note.* The RBF surrogate differs by parameterisation (IBEX `epsilon` vs scikit-learn `gamma`); the scikit-learn value here uses `gamma=1` and is not tuned, so it should be re-run with a matched kernel width before comparing.

**Table 4. Classification, mean cross-validated accuracy (5-fold).**

| Algorithm | Iris | Wine | Breast Cancer | Digits | IBEX (4 datasets) |
|---|---|---|---|---|---|
| Logistic regression | 0.953 | 0.983 | 0.974 | 0.971 | **[TO FILL]** |
| k-NN (k=5) | 0.973 | 0.972 | 0.963 | 0.979 | |
| Naive Bayes | 0.947 | 0.977 | 0.930 | 0.780 | |
| SVM (RBF) | 0.960 | 0.983 | 0.977 | 0.984 | |
| Random forest | 0.953 | 0.977 | 0.954 | 0.973 | |
| FastTree-like (GBM) | 0.947 | 0.950 | 0.954 | 0.967 | |
| LightGBM | 0.953 | 0.972 | 0.968 | 0.975 | |
| XGBoost | 0.953 | 0.955 | 0.972 | 0.965 | |
| ANN (MLP) | 0.947 | 0.989 | 0.974 | 0.974 | |

**Table 5. Clustering (standardised features; k = true number of classes where applicable).**

| Dataset | Algorithm | Clusters | Silhouette | ARI |
|---|---|---|---|---|
| Iris | K-means | 3 | 0.460 | 0.620 |
| Iris | DBSCAN (eps 1.5) | 2 | 0.582 | 0.568 |
| Iris | Hierarchical (average) | 3 | 0.480 | 0.562 |
| Wine | K-means | 3 | 0.285 | 0.897 |
| Wine | DBSCAN (eps 1.5) | 0 | n/a | 0.000 |
| Wine | Hierarchical (average) | 3 | 0.158 | −0.005 |
| Breast Cancer | K-means | 2 | 0.343 | 0.654 |
| Breast Cancer | DBSCAN (eps 1.5) | 1 | n/a | −0.023 |
| Breast Cancer | Hierarchical (average) | 2 | 0.634 | 0.007 |

DBSCAN with a single fixed `eps` finds no useful structure on Wine and Breast Cancer; `eps` should be set per dataset (e.g. via k-distance plot) for the IBEX comparison.

### 3.3 Results: IBEX **[TO FILL]**

Run each (algorithm, dataset) pair in IBEX with the same seed, 5 folds, z-score preprocessing and the mapped hyperparameters, and record the cross-validated score ("Combined" row), fit time and prediction time. The expected outcome to test is **agreement within cross-validation variability** for algorithms that share an implementation (linear, logistic, k-NN, naive Bayes, LightGBM, XGBoost); differences for FastTree vs gradient boosting and for RBF/MLP should be explained by implementation differences, not hidden.

### 3.4 Usability comparison **[TO FILL]**

Report for one standard task (e.g. load Wine, standardise, train SVM, 5-fold CV, read macro-F1): number of user actions in IBEX vs lines of Python for scikit-learn; and, if possible, task-completion time for a small group of architecture students.

## 4. Illustrative example in architectural design **[TO FILL]**

One short case (e.g. regression surrogate of a daylight or structural metric from parametric inputs; train in IBEX, deploy in Grasshopper through Dendrite), with dataset size, model, cross-validated R² and a screenshot.

## 5. Impact

- Free, offline and open-source, with a Microsoft Store release for non-technical users.
- Used in teaching ML to architecture students **[TO FILL: evidence]**.
- Bridge to parametric design through Dendrite (283 downloads on Food4Rhino at time of writing).
- Reproducible benchmark scripts that other tool authors can reuse.

## 6. Limitations

- Windows-only; CSV input only.
- Some algorithms (FastTree, RBF, LWR, fuzzy c-means) have no exact scikit-learn counterpart, so equivalence is approximate.
- The benchmark uses small, standard datasets; large-scale performance is not evaluated.
- Hyperparameters were not tuned per dataset, to keep the comparison symmetric.

## 7. Conclusions

IBEX offers a transparent, local, no-code route into ML for architects and educators, with a Grasshopper counterpart. A symmetric benchmark with scikit-learn, with all code and data released, allows others to verify its numerical behaviour and cite it as a reference implementation for teaching.

## Declarations

Funding: **[TO FILL]**. Competing interests: the author is the developer of IBEX; the app is free and donations are accepted through Ko-fi **[confirm wording]**. Data and code availability: GitHub and Zenodo (above).

## References **[verify each before submission]**

1. Pedregosa F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* 12 (2011) 2825–2830.
2. Hall M. et al. The WEKA data mining software: an update. *ACM SIGKDD Explorations* 11(1) (2009).
3. Demšar J. et al. Orange: data mining toolbox in Python. *Journal of Machine Learning Research* 14 (2013) 2349–2353.
4. Ahmed Z. et al. Machine Learning.NET (ML.NET). *Proc. ACM SIGMOD* (2019). **[verify]**
5. Chen T., Guestrin C. XGBoost: a scalable tree boosting system. *Proc. KDD* (2016).
6. Ke G. et al. LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS* (2017).
7. Khorsand Nikoo M. IBEX Machine Learning Studio v1.0.1. Zenodo (2026). doi:10.5281/zenodo.23136448.
