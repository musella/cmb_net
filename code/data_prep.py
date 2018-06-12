from __future__ import print_function
import pandas as pd
import glob
import numpy as np
import argparse
import ROOT

def load_df(folder, maxfiles):
    files = glob.glob(folder+'/*.csv')[:maxfiles]
    print("loading", files)
    df = pd.concat([pd.read_csv(x,sep=" ",index_col=False) for x in files])
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
    
    args = parser.parse_args()

    df = load_df(args.input, args.maxfiles)

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
