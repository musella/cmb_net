from __future__ import print_function
import pandas as pd
import glob
import numpy as np
import argparse
import ROOT
import os
import root_numpy

from pyjlr.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=str,
        default="data.h5", action="store",
        help="output file"
    )
    parser.add_argument(
        "--input", type=str,
        required=True, action="store", nargs='+',
        help="input folder or list of files"
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

    df = load_df(args.input)
    print(df.columns)

    if args.make_p4:
        for ilep in range(2):
            make_p4(df,'leps',ilep)

        for ijet in range(10):
            make_p4(df,'jets',ijet)

        #partons currently missing from tree
        for parton in ["jlr_top","jlr_atop","jlr_bottom","jlr_abottom"]:
            make_p4(df,parton,None)
        
        make_m2(df,"jlr_top",None,"jlr_atop",None)
        make_m2(df,"jlr_top",None,"jlr_bottom",None)
        make_m2(df,"jlr_top",None,"jlr_abottom",None)
        make_m2(df,"jlr_atop",None,"jlr_bottom",None)
        make_m2(df,"jlr_atop",None,"jlr_abottom",None)
        make_m2(df,"jlr_bottom",None,"jlr_abottom",None)
    
    print("saving {0} to {1}".format(df.shape, args.output))
    print(list(df.columns))
    df.to_hdf(args.output, key='df', format='t', mode='w')
