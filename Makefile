
OUTPATH=/scratch/${USER}/jlr

ttjets_1l:
	#python code/data_prep.py --input data/cmssw/ttjets_sl/ --output /scratch/${USER}/jlr/ttjets_sl_cms.h5 --maxfiles 100
	rm -Rf ${OUTPATH}/numpy/cms_ttjets_1l
	python code/format.py --infile ${OUTPATH}/ttjets_sl_cms.h5 --outdir ${OUTPATH}/numpy --datatype cms_ttjets_1l

tth_1l:
	#python code/data_prep.py --input data/cmssw/tth/ --output /scratch/${USER}/jlr/tth_cms.h5 --maxfiles -1
	rm -Rf ${OUTPATH}/cms_tth_1l
	python code/format.py --infile ${OUTPATH}/tth_cms.h5 --outdir ${OUTPATH}/numpy --datatype cms_tth_1l

#data:
#	mkdir -p /scratch/${USER}/jlr
#	python code/data_prep.py --input data/cmssw/tth/ --output /scratch/${USER}/jlr/tth_cms.h5 --maxfiles -1
#	#python code/data_prep.py --input data/cmssw/ttjets_sl/ --output /scratch/${USER}/jlr/ttjets_sl_cms.h5 --maxfiles 100
#	#python code/data_prep.py --input data/delphes/tth/ --output /scratch/${USER}/jlr/tth_delphes.h5 --maxfiles 100
#	#python code/data_prep.py --input data/delphes/ttbb/ --output /scratch/${USER}/jlr/ttbb_delphes.h5 --maxfiles 100
#
#numpy: data
#	mkdir -p /scratch/${USER}/jlr/numpy
#	rm -Rf /scratch/${USER}/jlr/numpy/cms_tth_1l
#	python code/format.py --infile /scratch/${USER}/jlr/tth_cms.h5 --outdir /scratch/${USER}/jlr/numpy --datatype cms_tth_1l
#	#python code/format.py --infile /scratch/${USER}/jlr/tth_cms.h5 --outdir /scratch/${USER}/jlr/numpy --datatype cms_tth_2l
#	#python code/format.py --infile /scratch/${USER}/jlr/tth_cms.h5 --outdir /scratch/${USER}/jlr/numpy --datatype cms_tth_had

.PHONY: ttjets_sl tth_1l
