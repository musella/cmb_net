import numpy as np

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
    
def make_pseudo_top(df,ilep,ijet):
    df['ptop_px_%d%d' % (ilep,ijet)] = df['jets_px_%d' % ijet] + df['leptons_px_%d' % ilep] + df['met_pt'] * np.cos(df['met_phi'])
    df['ptop_py_%d%d' % (ilep,ijet)] = df['jets_py_%d' % ijet] + df['leptons_py_%d' % ilep] + df['met_pt'] * np.sin(df['met_phi'])
    df['ptop_pz_%d%d' % (ilep,ijet)] = df['jets_pz_%d' % ijet] + df['leptons_pz_%d' % ilep]
    df['ptop_en_%d%d' % (ilep,ijet)] = df['jets_en_%d' % ijet] + df['leptons_en_%d' % ilep] + df['met_pt']
    df['ptop_pt_%d%d' % (ilep,ijet)] = np.sqrt(  df['ptop_px_%d%d' % (ilep,ijet)] **2 + df['ptop_py_%d%d' % (ilep,ijet)] **2  )
    df['ptop_mass_%d%d' % (ilep,ijet)] = np.sqrt( df['ptop_en_%d%d' % (ilep,ijet)]**2 - df['ptop_px_%d%d' % (ilep,ijet)] **2 - df['ptop_py_%d%d' % (ilep,ijet)] **2  - df['ptop_pz_%d%d' % (ilep,ijet)] **2  )
    df['ptop_eta_%d%d' % (ilep,ijet)] = np.arcsinh(  df['ptop_pt_%d%d' % (ilep,ijet)] / df['ptop_pz_%d%d' % (ilep,ijet)] )
