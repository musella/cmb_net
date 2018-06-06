
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


sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_cmb_cms_had  --loss mse --inp-dir $SCRATCH/cms_tth_had
sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_ffwd_cms_had --loss mse --inp-dir $SCRATCH/cms_tth_had

sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir huber_cmb_cms_had  --loss HuberLoss --inp-dir $SCRATCH/cms_tth_had
sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir huber_ffwd_cms_had --loss HuberLoss --inp-dir $SCRATCH/cms_tth_had

sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_exp_cmb_cms_had  --exp-target --loss mse --inp-dir $SCRATCH/cms_tth_had
sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_exp_ffwd_cms_had --exp-target --loss mse --inp-dir $SCRATCH/cms_tth_had

sbatch train_model.sh python ./train_model.py --epochs 50 --architecture cmb  --out-dir mse_unnorm_cmb_cms_had  --no-normalize-target --loss mse --inp-dir $SCRATCH/cms_tth_had
sbatch train_model.sh python ./train_model.py --epochs 50 --architecture ffwd --out-dir mse_unnorm_ffwd_cms_had --no-normalize-target --loss mse --inp-dir $SCRATCH/cms_tth_had
