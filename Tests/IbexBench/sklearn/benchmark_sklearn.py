"""scikit-learn reference benchmark for the IBEX Machine Learning Studio paper.

Mirrors IBEX's algorithm list (read from ML/*.cs):
  Regression (11): Linear (None/L1/L2), Polynomial, NormalEquation, LWR, RandomForest,
                   MLP, RBF surrogate, FastTree, LightGBM, OGD, XGBoost
  Classification (9): LogReg, KNN, NaiveBayes, SVM, RandomForest, FastTree, LightGBM, XGBoost, ANN
  Clustering (4): KMeans, DBSCAN, Hierarchical, Fuzzy C-means
Protocol: z-score inside each fold, 5-fold CV (stratified for classification),
seed 42, default-ish hyperparameters (IBEX slider defaults where known).
Output: results_sklearn_{regression,classification,clustering}.csv
"""
import time, warnings, sys, numpy as np, pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, f1_score,
                             silhouette_score, adjusted_rand_score)
warnings.filterwarnings("ignore")
SEED, K = 42, 5

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    LGBMRegressor = LGBMClassifier = None
try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception:
    XGBRegressor = XGBClassifier = None


def reg_models():
    m = {
        "Linear (no reg.)": LinearRegression(),
        "Linear (L1)": Lasso(alpha=0.01, max_iter=10000),
        "Linear (L2)": Ridge(alpha=1.0),
        "Polynomial (deg 2)": make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3)),
        "Normal Equation": LinearRegression(),
        "Locally Weighted Reg.": None,  # no sklearn equivalent (kernel-weighted local fit); see paper
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_leaf=10, random_state=SEED),
        "MLP": MLPRegressor(hidden_layer_sizes=(32,), max_iter=200, learning_rate_init=1e-3, random_state=SEED),
        "RBF surrogate": KernelRidge(kernel="rbf", gamma=1.0, alpha=1e-3),
        "FastTree-like (GBM)": GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_leaf_nodes=31, min_samples_leaf=10, random_state=SEED),
        "Online GD": SGDRegressor(max_iter=50, eta0=0.01, learning_rate="constant", tol=None, random_state=SEED),
    }
    if LGBMRegressor: m["LightGBM"] = LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=10, verbose=-1, random_state=SEED)
    if XGBRegressor: m["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, random_state=SEED, verbosity=0)
    return {k: v for k, v in m.items() if v is not None}


def cls_models():
    m = {
        "Logistic Regression (OvR)": OneVsRestClassifier(LogisticRegression(max_iter=2000)),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM (linear, OvR)": OneVsRestClassifier(SVC(kernel="linear", C=1.0)),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=12, min_samples_leaf=5, random_state=SEED),
        "FastTree-like (GBM)": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_leaf_nodes=24, min_samples_leaf=5, random_state=SEED),
        "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=SEED),
    }
    if LGBMClassifier: m["LightGBM"] = LGBMClassifier(n_estimators=50, learning_rate=0.1, num_leaves=24, min_child_samples=5, verbose=-1, random_state=SEED)
    if XGBClassifier: m["XGBoost"] = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=12, random_state=SEED, verbosity=0)
    return m


def reg_data():
    d = {}
    X, y = datasets.load_diabetes(return_X_y=True); d["Diabetes"] = (X, y)
    X, y = datasets.make_friedman1(n_samples=500, noise=1.0, random_state=SEED); d["Friedman1"] = (X, y)
    X, y = datasets.make_regression(n_samples=400, n_features=8, n_informative=5, noise=10, random_state=SEED); d["Synthetic-linear"] = (X, y)
    return d


def cls_data():
    d = {}
    for n, f in [("Iris", datasets.load_iris), ("Wine", datasets.load_wine),
                 ("Breast Cancer", datasets.load_breast_cancer), ("Digits", datasets.load_digits)]:
        X, y = f(return_X_y=True); d[n] = (X, y)
    return d


def run_supervised(data, models, task):
    rows = []
    for dn, (X, y) in data.items():
        splitter = (StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED) if task == "cls"
                    else KFold(n_splits=K, shuffle=True, random_state=SEED))
        for mn, model in models.items():
            sc, t_fit, t_pred = [], 0.0, 0.0
            sc2 = []
            from sklearn.base import clone
            for tr, te in splitter.split(X, y):
                ss = StandardScaler().fit(X[tr])
                Xtr, Xte = ss.transform(X[tr]), ss.transform(X[te])
                m = clone(model)
                ytr = y[tr]
                if task == "reg":  # z-score the target inside the fold (as IBEX's target recipe does)
                    ym, ys = ytr.mean(), ytr.std() or 1.0
                    ytr = (ytr - ym) / ys
                t0 = time.perf_counter(); m.fit(Xtr, ytr); t1 = time.perf_counter()
                p = m.predict(Xte); t2 = time.perf_counter()
                if task == "reg":
                    p = p * ys + ym
                t_fit += t1 - t0; t_pred += t2 - t1
                if task == "reg":
                    sc.append(r2_score(y[te], p)); sc2.append(np.sqrt(mean_squared_error(y[te], p)))
                else:
                    sc.append(accuracy_score(y[te], p)); sc2.append(f1_score(y[te], p, average="macro"))
            rows.append({"dataset": dn, "algorithm": mn,
                         ("R2_mean" if task == "reg" else "Acc_mean"): np.mean(sc),
                         ("R2_std" if task == "reg" else "Acc_std"): np.std(sc),
                         ("RMSE_mean" if task == "reg" else "MacroF1_mean"): np.mean(sc2),
                         "fit_s_total": t_fit, "pred_s_total": t_pred})
    return pd.DataFrame(rows)


def run_clustering():
    rows = []
    for dn, (X, y) in cls_data().items():
        if dn == "Digits": continue
        Xs = StandardScaler().fit_transform(X)
        k = len(np.unique(y))
        algos = {
            "K-Means": KMeans(n_clusters=k, n_init=10, random_state=SEED),
            "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
            "Hierarchical (average)": AgglomerativeClustering(n_clusters=k, linkage="average"),
        }
        for an, a in algos.items():
            t0 = time.perf_counter(); lab = a.fit_predict(Xs); t = time.perf_counter() - t0
            n_cl = len(set(lab)) - (1 if -1 in lab else 0)
            sil = silhouette_score(Xs, lab) if n_cl > 1 else float("nan")
            rows.append({"dataset": dn, "algorithm": an, "n_clusters": n_cl, "silhouette": sil,
                         "ARI": adjusted_rand_score(y, lab), "fit_s": t})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "reg"):
        run_supervised(reg_data(), reg_models(), "reg").to_csv("results_sklearn_regression.csv", index=False)
    if which in ("all", "cls"):
        run_supervised(cls_data(), cls_models(), "cls").to_csv("results_sklearn_classification.csv", index=False)
    if which in ("all", "clu"):
        run_clustering().to_csv("results_sklearn_clustering.csv", index=False)
    print("done")
