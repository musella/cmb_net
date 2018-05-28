#!/bin/bash

sbatch sub.sh python jlr.py --target jlr --dropout 0.0 --layers 4 --layersize 128 --activation tanh --lr 0.001 --epochs 1000 --earlystop 100 --input jlr_data_reco.npz --ntrain 20000 --ntest 20000 
sbatch sub.sh python jlr.py --target jlr --dropout 0.0 --layers 4 --layersize 128 --activation tanh --lr 0.001 --epochs 1000 --earlystop 100 --input jlr_data_reco.npz --ntrain 40000 --ntest 20000 
sbatch sub.sh python jlr.py --target jlr --dropout 0.0 --layers 4 --layersize 128 --activation tanh --lr 0.001 --epochs 1000 --earlystop 100 --input jlr_data_reco.npz --ntrain 80000 --ntest 20000 
sbatch sub.sh python jlr.py --target jlr --dropout 0.0 --layers 4 --layersize 128 --activation tanh --lr 0.001 --epochs 1000 --earlystop 100 --input jlr_data_reco.npz --ntrain 100000 --ntest 20000
sbatch sub.sh python jlr.py --target jlr --dropout 0.0 --layers 4 --layersize 128 --activation tanh --lr 0.001 --epochs 1000 --earlystop 100 --input jlr_data_reco.npz --ntrain 150000 --ntest 20000
sbatch sub.sh python jlr.py --target jlr --dropout 0.0 --layers 4 --layersize 128 --activation tanh --lr 0.001 --epochs 1000 --earlystop 100 --input jlr_data_reco.npz --ntrain 200000 --ntest 20000
sbatch sub.sh python jlr.py --target jlr --dropout 0.0 --layers 4 --layersize 128 --activation tanh --lr 0.001 --epochs 1000 --earlystop 100 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000
