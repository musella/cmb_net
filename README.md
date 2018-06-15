# Samples

See the `Makefile` for the sample generation. Roughly, it consists of the following steps:

1. We create the CMS training samples using the usual [tthbb13](https://gitlab.cern.ch/Zurich_ttH/tthbb13) workflow, i.e. with `MEAnalysis_heppy.py` or `Delphes_Analysis_heppy.py` in batch jobs.
2. The outputs are flattened ntuples in /pnfs, produced using the `flattener.py` script.
2. The flat root ntuples are converted to a pandas dataframe in HDF5 using `code/data_prep.py`.
3. The pandas dataframe is converted to numpy arrays using `code/format.py`.

The samples can be found in
~~~
#on t3ui02
:!ls -1 /scratch/jpata/jlr/numpy/
cms_tth_0l
cms_tth_1l
cms_tth_2l
cms_ttjets_0l
cms_ttjets_1l
cms_ttjets_2l

~~~

# Training

Python 3 and tensorflow are needed to run the training. A simple feedforward regression can be trained using
~~~
KERAS_BACKEND=tensorflow python code/train_model.py --inp-dir /scratch/jpata/jlr/numpy/cms_tth_0l --architecture ffwd --dropout 0.2 --layers 128,128,128,128,128 --out-dir results/tth_0l_ffwd --epochs 100 --features flat_nokin
~~~

# Validation
