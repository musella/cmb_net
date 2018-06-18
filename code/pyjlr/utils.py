import numpy as np
import pandas as pd
import root_numpy as rnp

import glob

try:
    from tqdm import tqdm
except:
    tqdm = None

# -------------------------------------------------------------------------------------------
## def load_df(folder, maxfiles):
##     files = sorted(glob.glob(folder+'/*.csv'))[:maxfiles]
##     print("loading {0} files from {1}".format(len(files),folder) )
##     if tqdm is not None:
##         files = tqdm(files)
##     df = pd.concat([pd.read_csv(x,sep=" ",index_col=False) for x in files]).reset_index()
##     return df

def load_df(folder):
    if len(folder) == 1 and os.path.isdir(folder[0]):
        print("loading files from folder {0}".format(folder[0]))
        files = sorted(glob.glob(folder[0] + '/*flat*.root'))
    elif isinstance(folder, list):
        files = list(folder)

    #in case we are trying to load from T3, add prefix
    new_files = []
    for fi in files:
        if fi.startswith("/pnfs/psi.ch"):
            fi = "root://t3dcachedb.psi.ch/" + fi
        new_files += [fi]
    files = new_files

    for fi in files:
        print(fi)
    df = pd.DataFrame(rnp.root2array(files, treename="tree"))
    df["JointLikelihoodRatioLog"] = np.log10(df["JointLikelihoodRatio"])
    return df


# -------------------------------------------------------------------------------------------
def load_mem_df(folder, maxfiles):
    files = sorted(glob.glob(folder+'/mem/*.csv'))[:maxfiles]
    print("loading {0} mem files from {1}".format(len(files),folder) )
    if tqdm is not None:
        files = tqdm(files)
    df = pd.concat([pd.read_csv(x,sep=",",index_col=0) for x in files]).reset_index()
    return df


# -------------------------------------------------------------------------------------------
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
    
# -------------------------------------------------------------------------------------------
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
    
# -------------------------------------------------------------------------------------------
def make_pseudo_top(df,ilep,ijet):
    df['ptop_px_%d%d' % (ilep,ijet)] = df['jets_px_%d' % ijet] + df['leptons_px_%d' % ilep] + df['met_pt'] * np.cos(df['met_phi'])
    df['ptop_py_%d%d' % (ilep,ijet)] = df['jets_py_%d' % ijet] + df['leptons_py_%d' % ilep] + df['met_pt'] * np.sin(df['met_phi'])
    df['ptop_pz_%d%d' % (ilep,ijet)] = df['jets_pz_%d' % ijet] + df['leptons_pz_%d' % ilep]
    df['ptop_en_%d%d' % (ilep,ijet)] = df['jets_en_%d' % ijet] + df['leptons_en_%d' % ilep] + df['met_pt']
    df['ptop_pt_%d%d' % (ilep,ijet)] = np.sqrt(  df['ptop_px_%d%d' % (ilep,ijet)] **2 + df['ptop_py_%d%d' % (ilep,ijet)] **2  )
    df['ptop_mass_%d%d' % (ilep,ijet)] = np.sqrt( df['ptop_en_%d%d' % (ilep,ijet)]**2 - df['ptop_px_%d%d' % (ilep,ijet)] **2 - df['ptop_py_%d%d' % (ilep,ijet)] **2  - df['ptop_pz_%d%d' % (ilep,ijet)] **2  )
    df['ptop_eta_%d%d' % (ilep,ijet)] = np.arcsinh(  df['ptop_pt_%d%d' % (ilep,ijet)] / df['ptop_pz_%d%d' % (ilep,ijet)] )
