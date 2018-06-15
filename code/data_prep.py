from __future__ import print_function
import pandas as pd
import glob
import numpy as np
import argparse
import ROOT
import os
import root_numpy

def load_df(folder):
    if isinstance(folder, str) and os.path.isdir(folder):
        files = sorted(glob.glob(folder+'/*flat*.root'))
    elif isinstance(folder, list):
        files = list(folder)

    #in case we are trying to load from T3, add prefix
    files = [fi.replace("/pnfs/psi.ch", "root://t3dcachedb.psi.ch/pnfs/psi.ch") for fi in files]
    for fi in files:
        print(fi)
    df = pd.DataFrame(root_numpy.root2array(files, treename="tree"))
    df["JointLikelihoodRatioLog"] = np.log10(df["JointLikelihoodRatio"])
    return df

def make_p4(df,collection,iob):
    iob = "" if iob is None else "_%d" % iob
    pt   =  df['%s_pt%s'  % (collection,iob)]
    eta  = df['%s_eta%s' % (collection,iob)]
    phi  = df['%s_phi%s' % (collection,iob)]
    mass = df['%s_mass%s' % (collection,iob)]
    df["%s_px%s" % (collection,iob)] = pt * np.cos(phi)
    df["%s_py%s" % (collection,iob)] = pt * np.sin(phi)
    df["%s_pz%s" % (collection,iob)] = pt * np.sinh(eta)
    df["%s_en%s" % (collection,iob)] = np.sqrt(mass**2 + (1+np.sinh(eta)**2)*pt**2)
    
    
def make_m2(df,coll1,iob1,coll2,iob2):
    
    im = ""
    if iob1 is not None:
        iob1 = "_%d" % iob1
        im += iob1
    else:
        iob1 = ""
    if iob2 is not None:
        if im.startswith("_"):
            im += "%d" % iob2
        else:
            im += "_%d" % iob2
        iob2 = "_%d" % iob2
    else:
        iob2 = ""
    
    px = df[ "%s_px%s" % (coll1,iob1) ] + df[ "%s_px%s" % (coll2,iob2) ]
    py = df[ "%s_py%s" % (coll1,iob1) ] + df[ "%s_py%s" % (coll2,iob2) ]
    pz = df[ "%s_pz%s" % (coll1,iob1) ] + df[ "%s_pz%s" % (coll2,iob2) ]
    en = df[ "%s_en%s" % (coll1,iob1) ] + df[ "%s_en%s" % (coll2,iob2) ]
    
    df["%s_%s_m2%s" %(coll1,coll2,im)] = en*en - px*px - py*py - pz*pz

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
    
    args = parser.parse_args()

    df = load_df(args.input)
    print(df.columns)

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
