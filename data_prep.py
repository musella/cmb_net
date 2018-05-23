import pandas as pd
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output", type=str,
    default="jlr_data.npz", action="store",
    help="output file"
)
parser.add_argument(
    "--maxfiles", type=int,
    default=10, action="store",
    help="max files to process"
)
args = parser.parse_args()

Xs = []
ys = []
for fn in glob.glob("samples/parton/*.csv")[:args.maxfiles]:
    data = pd.read_csv(fn, delim_whitespace=True)
    cols = data.columns
    feature_cols = cols[:-3]
    print feature_cols
    target_cols = cols[-3:]
    print target_cols
    X = data[feature_cols].as_matrix().astype("float32")
    y = data[target_cols].as_matrix().astype("float32")
    #y = X[:, 10:11] + X[:, 11:12]
    Xs += [X]
    ys += [y]
    print X.shape, y.shape

X = np.vstack(Xs)
y = np.vstack(ys)

of = open(args.output, "wb")
print X.shape, y.shape
np.savez(of, X=X, y=y)
of.close()
