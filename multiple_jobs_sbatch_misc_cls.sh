#! bin bash -l

job_File="sbatch_misc_cls.sh" 
# dir="sbatch_log/classification_dataset_liver_seg_gallbladder_filtered_small"

# adversary_param=$"10"
# num_vae_steps=$"2"
# seed=$"0"

# EXPT=small_classification_dataset_liver_seg_gallbladder_removed_full_train_seed_"$seed"
# STD=$dir/STD_full_train_seed_"$seed".out
# ERR=$dir/ERR_full_train_seed_"$seed".err
# METHOD="RandomSampling"
# ADVER_PARAM=$adversary_param
# NUM_VAE=$num_vae_steps
# RAND_SAM_SEED=$seed
# SEED=$seed


# export EXPT;
# export METHOD;
# export ADVER_PARAM;
# export NUM_VAE;
# export RAND_SAM_SEED;
# export SEED;


# sbatch -J $EXPT -o $STD -t 00-05:00:00 -e $ERR $job_File


dir="sbatch_log/classification_dataset_liver_seg_gallbladder_filtered_small"
adversary_param=$"10"
num_vae_steps=$"2"
mse_gamma=$"1"
for seed in 0 255 1000 2550 100
do
    EXPT=final_small_classification_dataset_liver_seg_gallbladder_removed_EXPT_MULTI_VAAL_adver"$adversary_param"_adaptive_mse_Seed"$seed"
    STD=$dir/STD_MULTI_VAAL_adver"$adversary_param"_adaptive_mse_Seed"$seed".out
    ERR=$dir/ERR_MULTI_VAAL_adver"$adversary_param"_adaptive_mse_Seed"$seed".err
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

    sbatch -J $EXPT -o $STD -t 02-00:00:00 -e $ERR $job_File   
    
done;