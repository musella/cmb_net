import pandas as pd
import glob
import numpy as np

Xs = []
ys = []
for fn in glob.glob("samples/parton/*.csv")[:10]:
    data = pd.read_csv(fn, delim_whitespace=True)
    cols = data.columns
    feature_cols = cols[:-3]
    print feature_cols
    target_cols = cols[-3:]
    print target_cols
    X = data[feature_cols].as_matrix().astype("float32")
    y = data[target_cols].as_matrix().astype("float32")
    Xs += [X]
    ys += [y]
    print X.shape, y.shape

X = np.vstack(Xs)
y = np.vstack(ys)

of = open("jlr_data.npz", "wb")
print X.shape, y.shape
np.savez(of, X=X, y=y)
of.close()
