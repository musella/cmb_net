OUTPATH=/scratch/${USER}/jlr

tth:
	mkdir -p ${OUTPATH}
	python code/data_prep.py --input /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/meanalysis/GC92b6146ce278/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/ --output ${OUTPATH}/tth.h5
	rm -Rf ${OUTPATH}/numpy/cms_tth_1l
	python code/format.py --infile ${OUTPATH}/tth.h5 --outdir ${OUTPATH}/numpy/cms_tth_1l --datatype cms_1l
	rm -Rf ${OUTPATH}/numpy/cms_tth_2l
	python code/format.py --infile ${OUTPATH}/tth.h5 --outdir ${OUTPATH}/numpy/cms_tth_2l --datatype cms_2l
	rm -Rf ${OUTPATH}/numpy/cms_tth_0l
	python code/format.py --infile ${OUTPATH}/tth.h5 --outdir ${OUTPATH}/numpy/cms_tth_0l --datatype cms_0l

ttjets_sl:
	mkdir -p ${OUTPATH}
	rm -Rf ${OUTPATH}/ttjets_sl_*.h5
	ls /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/flatten/GC898817b12ac5/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/*.root | parallel --gnu -n10 -j5 python code/data_prep.py --input {} --output ${OUTPATH}/ttjets_sl_{#}.h5
	rm -Rf ${OUTPATH}/numpy/cms_ttjets_1l
	\ls -1 ${OUTPATH}/ttjets_sl_*.h5 | parallel --gnu -n1 -j5 python code/format.py --infile {} --outdir ${OUTPATH}/numpy/cms_ttjets_1l/{#} --datatype cms_1l

ttjets_dl:
	mkdir -p ${OUTPATH}
	rm -Rf ${OUTPATH}/ttjets_dl_*.h5
	ls /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/flatten/GC898817b12ac5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/*.root | parallel --gnu -n10 -j5 python code/data_prep.py --input {} --output ${OUTPATH}/ttjets_dl_{#}.h5
	rm -Rf ${OUTPATH}/numpy/cms_ttjets_2l
	\ls -1 ${OUTPATH}/ttjets_dl_*.h5 | parallel --gnu -n1 -j5 python code/format.py --infile {} --outdir ${OUTPATH}/numpy/cms_ttjets_2l/{#} --datatype cms_2l

ttjets_fh:
	mkdir -p ${OUTPATH}
	rm -Rf ${OUTPATH}/ttjets_fh_*.h5
	ls /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/flatten/GC898817b12ac5/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/*.root | parallel --gnu -n10 -j5 python code/data_prep.py --input {} --output ${OUTPATH}/ttjets_fh_{#}.h5
	rm -Rf ${OUTPATH}/numpy/cms_ttjets_0l
	\ls -1 ${OUTPATH}/ttjets_fh_*.h5 | parallel --gnu -n1 -j5 python code/format.py --infile {} --outdir ${OUTPATH}/numpy/cms_ttjets_0l/{#} --datatype cms_0l

ttbb_delphes:
	mkdir -p ${OUTPATH}
	rm -Rf ${OUTPATH}/ttbb_delphes_*.h5
	ls  /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/delphes/GC48647908c2c2/Delphes_ttbb_inclusive/*.root | parallel --gnu -n35 -j5 python code/data_prep.py --input {} --output ${OUTPATH}/ttbb_delphes_{#}.h5
	rm -Rf ${OUTPATH}/numpy/delphes_ttbb_1l
	\ls -1 ${OUTPATH}/ttbb_delphes_*.h5 | parallel --gnu -n1 -j5 python code/format.py --infile {} --outdir ${OUTPATH}/numpy/delphes_ttbb_1l/{#} --datatype delphes_1l

tth_delphes:
	mkdir -p ${OUTPATH}
	rm -Rf ${OUTPATH}/tth_delphes_*.h5
	ls  /pnfs/psi.ch/cms/trivcat/store/user/jpata/tth/delphes/GC48647908c2c2/Delphes_ttH/*.root | parallel --gnu -n70 -j5 python code/data_prep.py --input {} --output ${OUTPATH}/tth_delphes_{#}.h5
	rm -Rf ${OUTPATH}/numpy/delphes_tth_1l
	\ls -1 ${OUTPATH}/tth_delphes_*.h5 | parallel --gnu -n1 -j5 python code/format.py --infile {} --outdir ${OUTPATH}/numpy/delphes_tth_1l/{#} --datatype delphes_1l

.PHONY: tth ttjets_sl
