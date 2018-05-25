#!/bin/bash

sbatch sub.sh python jlr.py --target jlr --batchnorm --layers 3 --layersize 512 --activation leakyrelu --dropout 0.1 --lr 0.001 --epochs 1000 --earlystop 1000 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000
sbatch sub.sh python jlr.py --target jlr --batchnorm --layers 4 --layersize 512 --activation leakyrelu --dropout 0.1 --lr 0.001 --epochs 1000 --earlystop 1000 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000
sbatch sub.sh python jlr.py --target jlr --batchnorm --layers 5 --layersize 512 --activation leakyrelu --dropout 0.1 --lr 0.001 --epochs 1000 --earlystop 1000 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000

sbatch sub.sh python jlr.py --target jlr --batchnorm --layers 3 --layersize 256 --activation leakyrelu --dropout 0.1 --lr 0.001 --epochs 1000 --earlystop 1000 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000
sbatch sub.sh python jlr.py --target jlr --batchnorm --layers 4 --layersize 256 --activation leakyrelu --dropout 0.1 --lr 0.001 --epochs 1000 --earlystop 1000 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000
sbatch sub.sh python jlr.py --target jlr --batchnorm --layers 5 --layersize 256 --activation leakyrelu --dropout 0.1 --lr 0.001 --epochs 1000 --earlystop 1000 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000

sbatch sub.sh python jlr.py --target jlr --batchnorm --layers 3 --layersize 128 --activation leakyrelu --dropout 0.1 --lr 0.001 --epochs 1000 --earlystop 1000 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000
sbatch sub.sh python jlr.py --target jlr --batchnorm --layers 4 --layersize 128 --activation leakyrelu --dropout 0.1 --lr 0.001 --epochs 1000 --earlystop 1000 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000
sbatch sub.sh python jlr.py --target jlr --batchnorm --layers 5 --layersize 128 --activation leakyrelu --dropout 0.1 --lr 0.001 --epochs 1000 --earlystop 1000 --input jlr_data_reco.npz --ntrain 300000 --ntest 20000
