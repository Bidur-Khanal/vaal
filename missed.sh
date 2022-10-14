#! bin bash -l

job_File="sbatch_run_classification.sh" 
dir="sbatch_log/classification_dataset_liver_seg_gallbladder_filtered"
adversary_param=$"10"
num_vae_steps=$"2"
seed=$"1000"
mse_gamma=$"0.8"

EXPT=classification_dataset_liver_seg_gallbladder_removed_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_Seed"$seed"
STD=$dir/STD_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_Seed"$seed".out
ERR=$dir/ERR_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_Seed"$seed".err
METHOD="multimodal_VAAL"
ADVER_PARAM=$adversary_param
MSE_GAMMA=$mse_gamma
NUM_VAE=$num_vae_steps
RAND_SAM_SEED=$"0"
SEED=$seed

export EXPT;
export METHOD;
export ADVER_PARAM;
export MSE_GAMMA;
export NUM_VAE;
export RAND_SAM_SEED;
export SEED;

sbatch -J $EXPT -o $STD -t 04-05:00:00 -e $ERR $job_File   

