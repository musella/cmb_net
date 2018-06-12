
data_dl:
	python code/data_prep.py --maxfiles -1 --output /scratch/jpata2/data_dl.npz --cut "(gen_num_leptons==2) & (gen_num_jets>=4)"

data_dl_match:
	python code/data_prep.py --maxfiles -1 --output /scratch/jpata2/data_dl_match.npz --cut "(gen_num_leptons==2) & (gen_num_jets>=4) & (nMatch_tb==2) & (nMatch_hb==2)"

data_sl:
	python code/data_prep.py --maxfiles -1 --output /scratch/jpata2/data_sl.npz --cut "(gen_num_leptons==1) & (gen_num_jets>=6)"

data_all:
	python code/data_prep.py --maxfiles -1 --output /scratch/jpata2/data_all.npz

.PHONY: data_dl data_sl data_all
