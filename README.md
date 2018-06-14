# Samples

1. We create the CMS training samples using the usual tthbb13 workflow, i.e. with `MEAnalysis_heppy.py` run using batch jobs.
2. The outputs are flattened ntuples in /pnfs `ls /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/meanalysis/GCcb1d8c0cc835/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/*flat*.root`.
2. The flat root ntuples are converted to a pandas dataframe in HDF5 using `code/data_prep.py`.
3. The pandas dataframe is converted to numpy arrays using `code/format.py`.

# Training

Python3 and tensorflow are needed to run the training. A simple feedforward regression can be trained using
~~~
KERAS_BACKEND=tensorflow python code/train_model.py --inp-dir data/numpy/cms_tth_0l --architecture ffwd --dropout 0.2 --layers 128,128,128,128,128 --out-dir results/tth_0l_ffwd --epochs 100 --features flat_nokin
~~~

# Validation