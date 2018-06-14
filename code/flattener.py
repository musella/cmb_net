#!/usr/bin/env python
"""Flattens a structured ROOT TTree with branches like jets_pt, jets_eta
into a dataframe-like TTree with scalar branches like jets_pt_0, jets_pt_1, ...

Usage example:

flattener.py --infile input.root --intree tree \
    --outfile test.root --outtree tree \
    -f jets:njets:pt,eta,phi:10 \
    -f leps:nleps:pt,eta,phi,mass,pdgId:2 \
    -b met_pt -b met_phi
"""
from __future__ import print_function

import ROOT
import numpy as np
from collections import OrderedDict
import argparse
import time

class Flatten:

    """Flattens an input collection like jets_pt[njets], jets_phi[njets]
    to jets_pt_0...jets_pt_MAXLEN, jets_phi_0...jets_phi_MAXLEN
    
    Attributes:
        collection (str): Base name of the collection, e.g. "jets"
        maxlen (int): Maximum number of objects to get from the collection
        subvars (list of str): List of subbranches, e.g. ["pt", "eta", ...]
        sizekey (str): The branch that indexes the number of objects, e.g. "njets"
        target_columns (OrderedDict): the numpy arrays used to hold the current values,
            indexed by [subbranch][obj_index]
    """
    
    def __init__(self, tree, collection, sizekey, subvars, maxlen):
        """Given an output tree and the collection data, creates the
        branch values and branches
        
        Args:
            tree (TYPE): Description
            collection (str): Base name of the collection, e.g. "jets"
            sizekey (str): The branch that indexes the number of objects, e.g. "njets"
            subvars (list of str): List of subbranches, e.g. ["pt", "eta", ...]
            maxlen (int): Maximum number of objects to get from the collection
        """
        self.collection = str(collection)
        self.sizekey = str(sizekey)
        self.subvars = subvars
        self.maxlen = int(maxlen)

        self.target_columns = OrderedDict()

        self.target_columns[self.sizekey] = np.zeros(1, dtype=np.float32)

        #only create the branch for the size counter in case we expect more than one object
        if self.maxlen > 1:
            tree.Branch(self.sizekey, self.target_columns[self.sizekey], "{0}/F".format(self.sizekey))

        for subvar in self.subvars:
            self.target_columns[subvar] = []
            for i in range(self.maxlen):
                
                if self.maxlen > 1:
                    brname = "{0}_{1}_{2}".format(collection, subvar, i)
                else:
                    brname = "{0}_{1}".format(collection, subvar)

                #create the numpy array bucket used to fill the output
                self.target_columns[subvar].append(np.zeros(1, dtype=np.float32))

                #create the output branch
                tree.Branch(brname, self.target_columns[subvar][i], "{0}/F".format(brname))

    def set_default(self):
        """Clears the data arrays
        """
        self.target_columns[self.sizekey][0] = 0.0
        for subvar in self.subvars:
            for icol in range(len(self.target_columns[subvar])):
                self.target_columns[subvar][icol][0] = 0.0

    def process(self, event):
        """Given a TTree at a specific event, fills the data arrays
        
        Args:
            event (ROOT.TTree): The input TTree
        """
        self.set_default()

        size = getattr(event, self.sizekey)
        self.target_columns[self.sizekey][0] = size
        for subvar in self.subvars:
            brname = "{0}_{1}".format(self.collection, subvar)
            brcont = getattr(event, brname)

            for iobj in range(min(size, self.maxlen)):
                self.target_columns[subvar][iobj][0] = brcont[iobj]

class Scalar:

    """Copies a single branch from the input tree to the output tree
    
    Attributes:
        branch_name (str): The input branch name
        branch_val (np.array): The data array used to hold the branch value
    """
    
    def __init__(self, tree, branch_name):
        """Given an output tree, creates the output branch
        
        Args:
            tree (ROOT.TTree): The output tree name
            branch_name (str): The output branch name
        """
        self.branch_name = branch_name
        self.branch_val = np.zeros(1, dtype=np.float32)
        tree.Branch(self.branch_name, self.branch_val, "{0}/F".format(self.branch_name))

    def process(self, event):
        """Given a TTree at a specific event, fills the data arrays
        
        Args:
            event (ROOT.TTree): The input TTree
        """
        self.branch_val[0] = getattr(event, self.branch_name)

def flatten_tree(intree, outfile, outname, flatten_commands, scalar_commands):
    """Converts an input tree which can contain array branches and scalar branches
        into a completely flat tree.
    
    Args:
        intree (ROOT.TTree): The input tree
        outfile (str): The output file name
        outname (str): The output tree name
        flatten_commands (list): List of instructions to flatten array branches.
            ["jets", "njets", ["pt", "eta"], 10] transforms 'jets_pt[njets]'
            to 'jets_pt_0...jets_pt_10'
        scalar_commands (list): List of simple scalar/number branches to copy
            from the input tree to the output
    """
    tf = ROOT.TFile(outfile, "RECREATE")
    outtree = ROOT.TTree(outname, "tree")

    flatten_objects = [Flatten(outtree, *fl) for fl in flatten_commands]
    scalar_objects = [Scalar(outtree, br) for br in scalar_commands]

    total_bytes = 0

    t0 = time.time()
    for iev, ev in enumerate(intree):

        #process the arrays
        for fl in flatten_objects:
            fl.process(ev)

        #process the scalar branches
        for sc in scalar_objects:
            sc.process(ev)
        
        if (iev % 1000 == 0):
            print("{0}/{1}".format(iev, intree.GetEntries()))

        total_bytes += outtree.Fill()
   
    #0-based indexing
    iev += 1

    t1 = time.time()
    dt = t1 - t0
    print("Filled {0} entries, {1:.2f} Hz, {2:.2f} MB".format(
        iev, iev/float(dt), total_bytes/1024.0/1024.0)
    )
    tf.Write()
    tf.Close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=
"""
Flattens a structured ROOT TTree with branches like jets_pt, jets_eta
into a dataframe-like TTree with scalar branches like jets_pt_0, jets_pt_1, ...

Usage example:

flattener.py --infile input.root --intree tree \
    --outfile test.root --outtree tree \
    -f jets:njets:pt,eta,phi:10 \
    -f leps:nleps:pt,eta,phi,mass,pdgId:2 \
    -b met_pt -b met_phi
""")
    parser.add_argument(
        "--infile", type=str,
        required=True, action="store",
        help="Input root file"
    )
    parser.add_argument(
        "--intree", type=str,
        required=True, action="store",
        help="Input tree name"
    )
    parser.add_argument(
        "--outfile", type=str,
        required=True, action="store",
        help="Output file name"
    )
    parser.add_argument(
        "--outtree", type=str,
        required=True, action="store",
        help="Output tree name"
    )
    parser.add_argument('--flatten',
        '-f', action='append', type=str,
        help="coll:ncoll:v1,v2,v3:maxlen"
    )
    parser.add_argument('--branch', '-b',
        action='append', type=str,
        help="scalar branch name to copy"
    )
    args = parser.parse_args()

    flatten_commands = []

    #parse the colon-separated flatten commands
    for fl_cmd in args.flatten:
        collection, sizekey, subvars, maxlen = fl_cmd.split(":")
        maxlen = int(maxlen)
        subvars = subvars.split(",")
        cmd = [collection, sizekey, subvars, maxlen]
        print("flatten branches", cmd)
        flatten_commands.append(cmd)
    print("scalar branches", args.branch)

    infile = ROOT.TFile(args.infile)
    intree = infile.Get(args.intree)

    flatten_tree(intree, args.outfile, args.outtree, flatten_commands, args.branch)
