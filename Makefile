data:
	python data_prep.py --output gen.npz --type gen --maxfiles -1  --cut="(num_leptons==1) & (num_jets>=6)"
	python data_prep.py --output reco.npz --type reco --maxfiles -1  --cut="(num_leptons==1) & (num_jets>=6)"
	python data_prep.py --output parton.npz --type parton --maxfiles -1  --cut="(num_leptons==1) & (num_jets>=6)"

train: train_reco train_parton train_gen

train_reco:
	OMP_NUM_THREADS=1 python jlr.py --input reco.npz --ntrain 280000 --ntest 20000 --batchnorm --activation tanh

train_parton:
	OMP_NUM_THREADS=1 python jlr.py --input parton.npz --ntrain 280000 --ntest 20000 --batchnorm --activation tanh

train_gen:
	OMP_NUM_THREADS=1 python jlr.py --input gen.npz --ntrain 280000 --ntest 20000 --batchnorm --activation tanh
