data:
	mkdir -p /scratch/${USER}/jlr
	python code/data_prep.py --input data/cmssw/tth/ --output /scratch/${USER}/jlr/tth_cms.h5 --maxfiles 100
	python code/data_prep.py --input data/cmssw/ttjets_sl/ --output /scratch/${USER}/jlr/ttjets_sl_cms.h5 --maxfiles 100
	python code/data_prep.py --input data/delphes/tth/ --output /scratch/${USER}/jlr/tth_delphes.h5 --maxfiles 100
	python code/data_prep.py --input data/delphes/ttbb/ --output /scratch/${USER}/jlr/ttbb_delphes.h5 --maxfiles 100

numpy:
	mkdir -p /scratch/${USER}/jlr/numpy
	python code/format.py --infile /scratch/${USER}/jlr/tth_cms.h5 --outdir /scratch/${USER}/jlr/numpy

.PHONY: data
