from pyjlr.utils import make_p4
import argparse
import os
import pandas as pd
import numpy as np

if __name__ == "__main__":  
# default options
    opts = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","btag"],
        njets = 10,
        lep_feats = ["pt","eta","phi","en","px","py","pz"],
        nleps = 2,
        met_feats = ["phi","pt","sumEt","px","py"],
        truth_feats = ["pt","eta","phi","en","px","py","pz"],
        evdesc_feats = ["nleps", "njets"],
    )
    
    # options for delphes files, full hadronic selection
    delphes_had = dict(
        selection = 'nleps == 0',
        met_feats = None,
    )
    
    # options for delphes files, 1l selection
    delphes_1l = dict(
        selection = 'nleps == 1',
        nleps = 1,
        met_feats = ["phi","pt","px","py"],
    )
    delphes_2l = dict(
        selection = 'nleps > 1',
        met_feats = ["phi","pt","px","py"],
    )

    # options for cms files, 1l selection
    cms_1l = dict(
        selection = 'nleps == 1',
        nleps = 1,
        met_feats = ["phi","pt","sumEt","px","py"],
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        evdesc_feats = ["nleps", "njets", "nBDeepCSVM", "nMatch_wq", "nMatch_tb", "nMatch_hb"],
    )
    
    cms_2l = dict(
        selection = 'nleps > 1',
        nleps = 2,
        met_feats = ["phi","pt","sumEt","px","py"],
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        evdesc_feats = ["nleps", "njets", "nBDeepCSVM", "nMatch_wq", "nMatch_tb", "nMatch_hb"],
    )
    
    cms_0l = dict(
        selection = 'nleps == 0',
        nleps = 0,
        met_feats = ["phi","pt","sumEt","px","py"],
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        evdesc_feats = ["nleps", "njets", "nBDeepCSVM", "nMatch_wq", "nMatch_tb", "nMatch_hb"],
    )

    datatype_choices = {
        "cms_0l": cms_0l,
        "cms_1l": cms_1l,
        "cms_2l": cms_2l,
        "delphes_1l": delphes_1l,
        "delphes_2l": delphes_2l,
        "delphes_had": delphes_had
    }


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", type=str,
        default="/scratch/{0}/jlr/".format(os.environ["USER"]), action="store",
        help="output directory"
    )
    parser.add_argument(
        "--infile", type=str,
        required=True, action="store",
        help="input folder"
    )
    parser.add_argument(
        "--datatype", type=str,
        choices=datatype_choices.keys(),
        required=True,
        help="datatype choice"
    )
    
    args = parser.parse_args()

    choose = datatype_choices[args.datatype]
    
    # copy specific options
    opts.update(choose)
    
    # copy default values to globals
    globals().update(opts)
    
    print("loading hdf file {0}".format(args.infile))
    df = pd.read_hdf(args.infile)
    print("df.shape={0}".format(df.shape))

    if selection is not None:
        df = df.query(selection)
    
    jetsa = None
    hcanda = None
    lepsa = None
    meta = None
    trutha = None
    kina = None

    os.makedirs(args.outdir)
    
    # make dijet higgs candidate combination
    def hcand(X):    
        cmb = X[0]
        # print(cmb)
        if type(cmb[0]) == list:
            return np.zeros( (2,njf),np.float32 )
        jets = X[1:].values.reshape(1,-1,njf)
        return jets[:,cmb[0,0]].astype(np.float32)
    
    
    # pad top kinematic fit solutions
    def pad(X,npad=6):
        if len(X.shape) < 4:
            X = np.zeros((npad,8,4,2))
        elif X.shape[0] < npad:
            X = np.vstack([X,np.zeros((6-X.shape[0],8,4,2))])
        elif X.shape[0] > npad:
            X = X[:npad]
        return X.reshape(-1,*X.shape)
    
    
    flats = []
    
    # --------------------------------------------------------------------------------------------------------------
    # jets
    if jet_feats is not None:
        print('formatting jets...')
        onejet = list(range(njets))
        for ijet in onejet:
            make_p4(df,'jets',ijet)
        njf = len(jet_feats)
        jet_feat_cols = ["jets_%s_%d" % (feat,jet) for jet in onejet for feat in jet_feats  ]
        jetsa = df[jet_feat_cols].values
        flats.append(jetsa)
        jetsa = jetsa.reshape(-1,njets,njf)
        np.save(args.outdir+"/jets",jetsa)
        print('done',jetsa.shape)
        
    # --------------------------------------------------------------------------------------------------------------
    # leptons
    if lep_feats is not None and nleps>0:
        print('formatting leptons...')
        nlf = len(lep_feats)
        for ilep in range(nleps):
            make_p4(df,'leps',ilep)
        lepsa = df[ ["leps_%s_%d" % (feat,lep) for feat in lep_feats for lep in range(nleps)  ]  ].values
        flats.append(lepsa)
        lepsa = lepsa.reshape(-1,nleps,nlf) 
        np.save(args.outdir+"/leps",lepsa)
        print('done',lepsa.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # met
    if met_feats is not None:
        print('formatting met...')
        df["met_px"] = df["met_"+met_feats[1]]*np.cos(df["met_"+met_feats[0]])
        df["met_py"] = df["met_"+met_feats[1]]*np.sin(df["met_"+met_feats[0]])
        meta = df[ ["met_%s" % feat for feat in met_feats  ]  ].values 
        flats.append(meta)
        np.save(args.outdir+"/met",meta)
        print('done',meta.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # flat array with all above
    print('making flat (nokin) features...')
    flata = np.hstack(flats)
    np.save(args.outdir+"/flat_nokin",flata)
    print('done',flata.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # jet combinations: higgs candidates and top kin fit solutions
    if jet_feats is not None and "jet_cmb" in df.columns:
        print('formatting jet combinations...')
        twojets = list(itertools.combinations(onejet,2))
    
        twojets2ind ={  cmb:icomb for icomb,cmb in enumerate(twojets)  }
        jet_cols = ["jets_cmb"]+jet_feat_cols+["jets_jets_m2_%d%d" % x for x in twojets]
        
        df["kin_sols"] = df["kin_sols"].apply(pad)#.apply(lambda x: pad_sequences(x,6,value=np.zeros() ).shape)
        
        hcanda = np.vstack( df[["jets_cmb"]+jet_feat_cols].apply(hcand,axis=1,raw=True).tolist() )
        kina = np.vstack(df["kin_sols"].tolist())
    
        flats.append(hcanda.reshape(hcanda.shape[0],-1))
        flats.append(kina.reshape(kina.shape[0],-1))
        np.save(args.outdir+"/hcand",hcanda)
        np.save(args.outdir+"/kinsols",kina)
        print('done',hcanda.shape,kina.shape)    
        
    # --------------------------------------------------------------------------------------------------------------
    # flat arrat with all above
    print('making flat features...')
    flata = np.hstack(flats)
    np.save(args.outdir+"/flat",flata)
    print('done',flata.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # target
    print('making target...')
    jlra = df["JointLikelihoodRatioLog"].values
    np.save(args.outdir+"/target",jlra)
    print('done',jlra.shape)
    
    
    # --------------------------------------------------------------------------------------------------------------
    # MEM
    print('making MEM...')
    arr = df[["mem_tth_SL_2w2h2t_p", "mem_ttbb_SL_2w2h2t_p"]].values
    np.save(args.outdir+"/mem",arr)
    print('done', arr.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # Event description features
    print('making evdesc...')
    arr = df[evdesc_feats].values
    np.save(args.outdir+"/evdesc",arr)
    print('done', arr.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # truth level info
    if truth_feats is not None:
        print('formatting truth...')
        ntf = len(truth_feats)
        trutha = df[ ["%s_%s" % (part,feat) for feat in truth_feats for part in ["jlr_top","jlr_atop","jlr_bottom","jlr_abottom"]  ]  ].values 
        trutha = trutha.reshape(-1,4,ntf)
        np.save(args.outdir+"/truth",trutha)    
        print('done')
