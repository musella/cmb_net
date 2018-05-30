data: data_gen data_reco data_parton

data_gen:
	python data_prep.py --output gen.npz --type gen --maxfiles -1  --cut="(num_leptons==1) & (num_jets>=4)"

data_reco:
	python data_prep.py --output reco.npz --type reco --maxfiles -1  --cut="(num_leptons==1) & (num_jets>=4)"

data_parton:
	python data_prep.py --output parton.npz --type parton --maxfiles -1  --cut="(num_leptons==1) & (num_jets>=4)"

train: train_reco train_parton train_gen

train_reco:
	OMP_NUM_THREADS=1 python jlr.py --input reco.npz --ntrain 700000 --ntest 100000 --batchnorm --activation tanh --layersize 256 --verbosity 1

train_parton:
	OMP_NUM_THREADS=1 python jlr.py --input parton.npz --ntrain 700000 --ntest 100000 --batchnorm --activation tanh --layersize 256 --verbosity 1

train_gen:
	OMP_NUM_THREADS=1 python jlr.py --input gen.npz --ntrain 700000 --ntest 100000 --batchnorm --activation tanh --layersize 256 --verbosity 1
