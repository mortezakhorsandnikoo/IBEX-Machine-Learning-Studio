"""Exports the benchmark datasets to CSV for IBEX (target = LAST column, header row, comma separated).
Same generation parameters as benchmark_sklearn.py. Run:  python export_datasets.py data
"""
import sys, os, numpy as np, pandas as pd
from sklearn import datasets

out = sys.argv[1] if len(sys.argv) > 1 else "data"
os.makedirs(out, exist_ok=True)
SEED = 42

def clean(cols):
    return [str(c).replace(" ", "_").replace(",", "_").replace("(", "").replace(")", "").replace("/", "_") for c in cols]

def save_sup(name, X, y, cols, target="target", integer_target=False):
    df = pd.DataFrame(X, columns=clean(cols))
    df[target] = y.astype(int) if integer_target else y
    df.to_csv(os.path.join(out, name), index=False, float_format="%.10g")

d = datasets.load_diabetes();            save_sup("reg_diabetes.csv", d.data, d.target, d.feature_names)
X, y = datasets.make_friedman1(n_samples=500, noise=1.0, random_state=SEED)
save_sup("reg_friedman1.csv", X, y, [f"x{i+1}" for i in range(X.shape[1])])
X, y = datasets.make_regression(n_samples=400, n_features=8, n_informative=5, noise=10, random_state=SEED)
save_sup("reg_synthetic_linear.csv", X, y, [f"x{i+1}" for i in range(X.shape[1])])

for name, loader in [("iris", datasets.load_iris), ("wine", datasets.load_wine),
                     ("breast_cancer", datasets.load_breast_cancer), ("digits", datasets.load_digits)]:
    d = loader()
    save_sup(f"cls_{name}.csv", d.data, d.target, getattr(d, "feature_names", [f"p{i}" for i in range(d.data.shape[1])]),
             target="class", integer_target=True)
    if name != "digits":  # clustering: features only + separate label file (labels used only for ARI)
        pd.DataFrame(d.data, columns=clean(d.feature_names)).to_csv(os.path.join(out, f"clu_{name}.csv"), index=False, float_format="%.10g")
        pd.DataFrame({"label": d.target.astype(int)}).to_csv(os.path.join(out, f"clu_{name}_labels.csv"), index=False)

print("wrote", sorted(os.listdir(out)))
