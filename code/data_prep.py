from __future__ import print_function
import pandas as pd
import glob
import numpy as np
import argparse
import ROOT
import os

from pyjlr.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str,
        default="jlr_data.npz", action="store",
        help="output file"
    )
    parser.add_argument(
        "--input", type=str,
        required=True, action="store",
        help="input folder"
    )
    parser.add_argument(
        "--maxfiles", type=int,
        default=-1, action="store",
        help="max files to process"
    )
    parser.add_argument(
        "--make-p4",dest="make_p4",
        default=False, action="store_true",
        help="compute 4-vectors"
    )
    parser.add_argument(
        "--no-make-p4",dest="make_p4",
        action="store_false",
        help="do not compute 4-vectors"
    )
    
    args = parser.parse_args()

    df = load_df(args.input, args.maxfiles)
    if os.path.isdir(args.input + "/mem"):
        df_mem = load_mem_df(args.input, args.maxfiles)
        assert(df.shape[0] == df_mem.shape[0])
        df = pd.concat([df, df_mem], axis=1)
    else:
        print("could not find the {0}/mem directory, setting to 0".format(args.input))
        df["mem_ratio"] = 0.0
        df["mem_ttbb"] = 0.0
        df["mem_tth"] = 0.0
    print(df.columns)

    if args.make_p4:
        for ilep in range(2):
            make_p4(df,'leptons',ilep)
            
        for ijet in range(10):
            make_p4(df,'jets',ijet)
            
        for parton in ["top","atop","bottom","abottom"]:
            make_p4(df,parton,None)
        
        make_m2(df,"top",None,"atop",None)
        make_m2(df,"top",None,"bottom",None)
        make_m2(df,"top",None,"abottom",None)
        make_m2(df,"atop",None,"bottom",None)
        make_m2(df,"atop",None,"abottom",None)
        make_m2(df,"bottom",None,"abottom",None)
    
    print("saving {0} to {1}".format(df.shape, args.output))
    df.to_hdf(args.output, key='df', format='t', mode='w')
