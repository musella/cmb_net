data:
	mkdir -p /scratch/${USER}/jlr
	python code/data_prep.py --input data/cmssw/tth/ --output /scratch/${USER}/jlr/tth_cms.npz --maxfiles 10
	python code/data_prep.py --input data/cmssw/ttjets_sl/ --output /scratch/${USER}/jlr/ttjets_sl_cms.npz --maxfiles 10

.PHONY: data
