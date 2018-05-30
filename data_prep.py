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
parser.add_argument(
    "--cut", type=str,
    default="", action="store",
    help="cut to apply"
)
parser.add_argument(
    "--type", type=str,
    default="reco", choices=["reco", "parton", "gen"],
    help="type of input, reco level, genjet level or parton level (hard ME)"
)
args = parser.parse_args()

Xs = []
ys = []
for fn in glob.glob("samples/combined/*.csv".format(args.type))[:args.maxfiles]:
    data = pd.read_csv(fn, delim_whitespace=True)
    cols = list(data.columns)
    ic1 = cols.index("gen_num_leptons")
    ic2 = cols.index("top_pt")
    reco_cols = cols[:ic1]
    gen_cols = cols[ic1:ic2]
    parton_cols = cols[ic2:-3]

    print "precut", data.shape
    if len(args.cut) > 0:
        data = data[data.eval(args.cut)]
    print "postcut", data.shape
    if args.type == "parton":
        feature_cols = parton_cols
        target_cols = cols[-3:]
    elif args.type == "reco":
        feature_cols = reco_cols
        target_cols = cols[-3:]
    elif args.type == "gen":
        feature_cols = gen_cols
        target_cols = cols[-3:]
    print feature_cols
    print target_cols
    X = data[feature_cols].as_matrix().astype("float32")
    y = data[target_cols].as_matrix().astype("float32")
    Xs += [X]
    ys += [y]
    print X.shape, y.shape

X = np.vstack(Xs)
y = np.vstack(ys)

of = open(args.output, "wb")
print X.shape, y.shape
np.savez(of, X=X, y=y)
of.close()
