import pandas as pd
import glob
import numpy as np
import argparse
import ROOT

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

Xsreco = []
Xsparton = []
ys = []

def p4_spherical_to_cartesian(pt, eta, phi, mass):
    v = ROOT.TLorentzVector()
    v.SetPtEtaPhiM(pt, eta, phi, mass)
    return v.Px(), v.Py(), v.Pz(), v.M()

reco_cols = ['num_leptons', 'leptons_pt_0', 'leptons_pt_1', 'leptons_eta_0', 'leptons_eta_1', 'leptons_phi_0', 'leptons_phi_1', 'leptons_mass_0', 'leptons_mass_1', 'num_jets', 'jets_pt_0', 'jets_pt_1', 'jets_pt_2', 'jets_pt_3', 'jets_pt_4', 'jets_pt_5', 'jets_pt_6', 'jets_pt_7', 'jets_pt_8', 'jets_pt_9', 'jets_eta_0', 'jets_eta_1', 'jets_eta_2', 'jets_eta_3', 'jets_eta_4', 'jets_eta_5', 'jets_eta_6', 'jets_eta_7', 'jets_eta_8', 'jets_eta_9', 'jets_phi_0', 'jets_phi_1', 'jets_phi_2', 'jets_phi_3',
'jets_phi_4', 'jets_phi_5', 'jets_phi_6', 'jets_phi_7', 'jets_phi_8', 'jets_phi_9', 'jets_mass_0', 'jets_mass_1', 'jets_mass_2', 'jets_mass_3', 'jets_mass_4', 'jets_mass_5', 'jets_mass_6', 'jets_mass_7', 'jets_mass_8', 'jets_mass_9', 'jets_btagDeepCSV_0', 'jets_btagDeepCSV_1', 'jets_btagDeepCSV_2', 'jets_btagDeepCSV_3', 'jets_btagDeepCSV_4', 'jets_btagDeepCSV_5', 'jets_btagDeepCSV_6', 'jets_btagDeepCSV_7', 'jets_btagDeepCSV_8', 'jets_btagDeepCSV_9', 'met_pt', 'met_phi', 'met_sumEt',
'nBDeepCSVM', 'mbb_closest', 'ht30']
gen_cols = ['gen_num_leptons', 'gen_leptons_pt_0', 'gen_leptons_pt_1', 'gen_leptons_eta_0', 'gen_leptons_eta_1', 'gen_leptons_phi_0', 'gen_leptons_phi_1', 'gen_leptons_mass_0', 'gen_leptons_mass_1', 'gen_num_jets', 'gen_jets_pt_0',
'gen_jets_pt_1', 'gen_jets_pt_2', 'gen_jets_pt_3', 'gen_jets_pt_4', 'gen_jets_pt_5', 'gen_jets_pt_6', 'gen_jets_pt_7', 'gen_jets_pt_8', 'gen_jets_pt_9', 'gen_jets_eta_0', 'gen_jets_eta_1', 'gen_jets_eta_2', 'gen_jets_eta_3', 'gen_jets_eta_4', 'gen_jets_eta_5', 'gen_jets_eta_6', 'gen_jets_eta_7', 'gen_jets_eta_8', 'gen_jets_eta_9', 'gen_jets_phi_0', 'gen_jets_phi_1', 'gen_jets_phi_2', 'gen_jets_phi_3', 'gen_jets_phi_4', 'gen_jets_phi_5', 'gen_jets_phi_6', 'gen_jets_phi_7', 'gen_jets_phi_8',
'gen_jets_phi_9', 'gen_jets_mass_0', 'gen_jets_mass_1', 'gen_jets_mass_2', 'gen_jets_mass_3', 'gen_jets_mass_4', 'gen_jets_mass_5', 'gen_jets_mass_6', 'gen_jets_mass_7', 'gen_jets_mass_8', 'gen_jets_mass_9', 'gen_jets_matchFlag_0', 'gen_jets_matchFlag_1', 'gen_jets_matchFlag_2', 'gen_jets_matchFlag_3', 'gen_jets_matchFlag_4', 'gen_jets_matchFlag_5', 'gen_jets_matchFlag_6', 'gen_jets_matchFlag_7', 'gen_jets_matchFlag_8', 'gen_jets_matchFlag_9', 'gen_num_nu', 'gen_nu_pt_0', 'gen_nu_pt_1',
'gen_nu_eta_0', 'gen_nu_eta_1', 'gen_nu_phi_0', 'gen_nu_phi_1']
parton_cols = ['top_pt', 'top_eta', 'top_phi', 'top_mass', 'atop_pt', 'atop_eta', 'atop_phi', 'atop_mass', 'bottom_pt', 'bottom_eta', 'bottom_phi', 'bottom_mass', 'abottom_pt', 'abottom_eta', 'abottom_phi', 'abottom_mass']
target_cols = ['prob_ttH', 'prob_ttbb', 'JLR']

#reco_cols += ['jets_matchFlag_0', 'jets_matchFlag_1', 'jets_matchFlag_2', 'jets_matchFlag_3', 'jets_matchFlag_4', 'jets_matchFlag_5', 'jets_matchFlag_6', 'jets_matchFlag_7', 'jets_matchFlag_8', 'jets_matchFlag_9']
#['nMatch_wq', 'nMatch_tb', 'nMatch_hb',]


for fn in glob.glob("data/Jun5/*.csv".format(args.type))[:args.maxfiles]:
    data = pd.read_csv(fn, delim_whitespace=True)
    cols = list(data.columns)

    print "precut", data.shape
    if len(args.cut) > 0:
        data = data[data.eval(args.cut)]
    print "postcut", data.shape
    if args.type == "parton":
        feature_cols = parton_cols
    elif args.type == "reco":
        feature_cols = reco_cols
    elif args.type == "gen":
        feature_cols = gen_cols
    print feature_cols
    print target_cols
    Xreco = data[reco_cols].as_matrix().astype("float32")
    jets = [["jets_pt_{0}".format(ij), "jets_eta_{0}".format(ij), "jets_phi_{0}".format(ij), "jets_mass_{0}".format(ij)] for ij in range(0,9)]
    leps = [["leptons_pt_{0}".format(ij), "leptons_eta_{0}".format(ij), "leptons_phi_{0}".format(ij), "leptons_mass_{0}".format(ij)] for ij in range(0,2)]

    lvs_jet = [] 
    for jet in jets:
        mm = data[jet].as_matrix()
        lvs = np.array([p4_spherical_to_cartesian(*mm[i, :]) for i in range(mm.shape[0])])
        lvs_jet += [lvs]
    jets_btag = ["jets_btagDeepCSV_{0}".format(ij) for ij in range(0,9)]
    csvs = data[jets_btag].as_matrix()
    lvs_jet = np.hstack(lvs_jet + [csvs])
    
    lvs_lep = [] 
    for lep in leps:
        mm = data[lep].as_matrix()
        lvs = np.array([p4_spherical_to_cartesian(*mm[i, :]) for i in range(mm.shape[0])])
        lvs_lep += [lvs]
    lvs_lep = np.hstack(lvs_lep)

    lvs_met = np.zeros((mm.shape[0], 2))
    lvs_met[:, 0] = data["met_pt"]*np.cos(data["met_phi"])
    lvs_met[:, 1] = data["met_pt"]*np.sin(data["met_phi"])
    
    lvs_jet_lep_met = np.hstack([lvs_lep, lvs_jet, lvs_met])

    partons = [
        ['top_pt', 'top_eta', 'top_phi', 'top_mass'],
        ['atop_pt', 'atop_eta', 'atop_phi', 'atop_mass'],
        ['bottom_pt', 'bottom_eta', 'bottom_phi', 'bottom_mass'],
        ['abottom_pt', 'abottom_eta', 'abottom_phi', 'abottom_mass']
    ]

    lvs_parton = []
    for parton in partons:
        mm = data[parton].as_matrix()
        lvs = np.array([p4_spherical_to_cartesian(*mm[i, :]) for i in range(mm.shape[0])])
        lvs_parton += [lvs]
    lvs_parton = np.hstack(lvs_parton)

    Xparton = data[parton_cols].as_matrix().astype("float32")
    y = data[target_cols].as_matrix().astype("float32")
    Xsreco += [lvs_jet_lep_met]
    Xsparton += [lvs_parton]
    ys += [y]
    #print Xreco.shape, Xparton.shape, y.shape

Xreco = np.vstack(Xsreco)
Xparton = np.vstack(Xsparton)
y = np.vstack(ys)

of = open(args.output, "wb")
print Xreco.shape, Xparton.shape, y.shape
np.savez(of, Xreco=Xreco, Xparton=Xparton, y=y)
of.close()
