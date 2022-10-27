#! bin bash -l

job_File="sbatch_run.sh" 


dir="sbatch_log/dataset_liver_seg_gallbladder_removed_class_no_less_than_3_small"
adversary_param=$"25"
num_vae_steps=$"2"
mse_gamma=$"1"
for seed in 0 255 1000 2550 100
do
    EXPT=final_small_dataset_liver_seg_gallbladder_removed_class_no_less_than_3_EXPT_MULTI_VAAL2_adver"$adversary_param"_Seed"$seed"
    STD=$dir/STD_MULTI_VAAL2_adver"$adversary_param"_Seed"$seed".out
    ERR=$dir/ERR_MULTI_VAAL2_adver"$adversary_param"_Seed"$seed".err
    METHOD="multimodal_VAAL2"
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

    sbatch -J $EXPT -o $STD -t 03-10:00:00 -e $ERR $job_File   
    
done;