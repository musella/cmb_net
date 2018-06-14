
# SL
sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_exp_ffwd_delphes_1l --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_1l --features flat
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb --out-dir mse_exp_cmb_delphes_1l --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_1l --features jets,leps,met --dijet-dropout 0.,0.2,0.2 --trijet-dropout 0.,0.2,0.2 --attention-dropout 0.2,0.2 --fc-dropout 0.2,0.2 --lep-dropout 0.,0.2,0.2


# HAD
### kfolds=5
### for ifold in $(seq 0 $(($kfolds-1)) ); do
###     opts="--kfolds $kfolds --ifold $ifold"
###     sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir xval/mse_exp_cmb_delphes_had  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had $opts
###     sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir xval/mse_exp_ffwd_delphes_had --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had $opts
###     ## sbatch train_model.sh python ./train_model.py --epochs 30 --architecture cmb  --out-dir xval/mse_exp_cmb_noatt_delphes_had --no-do-attention --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had $opts
###     ## sbatch train_model.sh python ./train_model.py --epochs 30 --architecture cmb  --out-dir xval/mse_exp_cmb_delphes_had_doa1  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --attention-dropout 0.2,0.2 --fc-dropout 0.2,0.2 $opts
### done


## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_noatt_delphes_had --no-do-attention --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had 

## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_rnn_delphes_had  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --do-rnn 2 --dijet-rnn-units 16,16 --trijet-rnn-units 32,32 

## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_rnn_delphes_had_do  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --do-rnn 2 --dijet-rnn-units 16,16 --trijet-rnn-units 32,32 --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --attention-dropout 0.2,0.2 --fc-dropout 0.2,0.2

## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_doa1  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --attention-dropout 0.2,0.2 --fc-dropout 0.2,0.2
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_doa2  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --attention-dropout 0.3,0.3 --fc-dropout 0.2,0.2
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_doa3  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --attention-dropout 0.5,0.5 --fc-dropout 0.2,0.2


## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_no1  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.,0.2  --trijet-dropout 0.,0.,0.2 --trijet-noise 0.,0.05,0.05  --dijet-noise 0.,0.05,0.05 --fc-dropout 0.2,0.2
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_no2  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.,0.2  --trijet-dropout 0.,0.,0.2 --trijet-noise 0.,0.1,0.1  --dijet-noise 0.,0.1,0.1 --fc-dropout 0.2,0.2
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_no3  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.,0.2  --trijet-dropout 0.,0.,0.2 --trijet-noise 0.,0.15,0.15  --dijet-noise 0.,0.15,0.15 --fc-dropout 0.2,0.2
 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_no4  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --attention-noise 0.05,0.05 --fc-dropout 0.2,0.2
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_no5  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --attention-noise 0.1,0.1 --fc-dropout 0.2,0.2
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_no6  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --attention-noise 0.15,0.15 --fc-dropout 0.2,0.2

## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_doj1  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --jets-dropout 0.1 --fc-dropout 0.2,0.2
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_doj2  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --jets-dropout 0.2 --fc-dropout 0.2,0.2
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_doj3  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --jets-dropout 0.3 --fc-dropout 0.2,0.2

## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_do1  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.5  --trijet-dropout 0.,0.2,0.5 --fc-dropout 0.5,0.5 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_do2  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --fc-dropout 0.5,0.5 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_do3  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --dijet-dropout 0.,0.2,0.2  --trijet-dropout 0.,0.2,0.2 --fc-dropout 0.2,0.2 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_do4  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --fc-dropout 0.5,0.5
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had_do5  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had --fc-dropout 0.2,0.2


## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_cmb_delphes_had  --loss mse --inp-dir $SCRATCH/delphes_tth_had
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_ffwd_delphes_had --loss mse --inp-dir $SCRATCH/delphes_tth_had
## 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir huber_cmb_delphes_had  --loss HuberLoss --inp-dir $SCRATCH/delphes_tth_had
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir huber_ffwd_delphes_had --loss HuberLoss --inp-dir $SCRATCH/delphes_tth_had
## 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_delphes_had  --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_exp_ffwd_delphes_had --exp-target --loss mse --inp-dir $SCRATCH/delphes_tth_had
## 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_unnorm_cmb_delphes_had  --no-normalize-target --loss mse --inp-dir $SCRATCH/delphes_tth_had
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_unnorm_ffwd_delphes_had --no-normalize-target --loss mse --inp-dir $SCRATCH/delphes_tth_had


## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_cmb_cms_had  --loss mse --inp-dir $SCRATCH/cms_tth_had
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_ffwd_cms_had --loss mse --inp-dir $SCRATCH/cms_tth_had
## 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir huber_cmb_cms_had  --loss HuberLoss --inp-dir $SCRATCH/cms_tth_had
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir huber_ffwd_cms_had --loss HuberLoss --inp-dir $SCRATCH/cms_tth_had
## 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_cms_had  --exp-target --loss mse --inp-dir $SCRATCH/cms_tth_had
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_exp_ffwd_cms_had --exp-target --loss mse --inp-dir $SCRATCH/cms_tth_had
## 
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_unnorm_cmb_cms_had  --no-normalize-target --loss mse --inp-dir $SCRATCH/cms_tth_had
## sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_unnorm_ffwd_cms_had --no-normalize-target --loss mse --inp-dir $SCRATCH/cms_tth_had
