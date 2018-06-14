OUTPATH=/scratch/${USER}/jlr

tth:
	python code/data_prep.py --input /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/meanalysis/GC92b6146ce278/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/ --output ${OUTPATH}/tth.h5
	rm -Rf ${OUTPATH}/numpy/cms_tth_1l
	python code/format.py --infile ${OUTPATH}/tth.h5 --outdir ${OUTPATH}/numpy/cms_tth_1l --datatype cms_1l
	rm -Rf ${OUTPATH}/numpy/cms_tth_2l
	python code/format.py --infile ${OUTPATH}/tth.h5 --outdir ${OUTPATH}/numpy/cms_tth_2l --datatype cms_2l
	rm -Rf ${OUTPATH}/numpy/cms_tth_0l
	python code/format.py --infile ${OUTPATH}/tth.h5 --outdir ${OUTPATH}/numpy/cms_tth_0l --datatype cms_0l

.PHONY: tth
