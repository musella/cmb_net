OUTPATH=/scratch/${USER}/jlr

tth:
	python code/data_prep.py --input /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/meanalysis/GC92b6146ce278/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/ --output ${OUTPATH}/tth.h5
	rm -Rf ${OUTPATH}/numpy/cms_tth_1l
	python code/format.py --infile ${OUTPATH}/tth.h5 --outdir ${OUTPATH}/numpy/cms_tth_1l --datatype cms_1l
	rm -Rf ${OUTPATH}/numpy/cms_tth_2l
	python code/format.py --infile ${OUTPATH}/tth.h5 --outdir ${OUTPATH}/numpy/cms_tth_2l --datatype cms_2l
	rm -Rf ${OUTPATH}/numpy/cms_tth_0l
	python code/format.py --infile ${OUTPATH}/tth.h5 --outdir ${OUTPATH}/numpy/cms_tth_0l --datatype cms_0l

ttjets_sl:
	python code/data_prep.py --input /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/flatten/GC898817b12ac5/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/ --output ${OUTPATH}/ttjets_sl.h5
	rm -Rf ${OUTPATH}/numpy/cms_ttjets_1l
	python code/format.py --infile ${OUTPATH}/ttjets_sl.h5 --outdir ${OUTPATH}/numpy/cms_ttjets_1l --datatype cms_1l

.PHONY: tth ttjets_sl
