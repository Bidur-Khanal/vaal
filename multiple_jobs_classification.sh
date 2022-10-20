#! bin bash -l

job_File="sbatch_run_classification.sh" 
dir="sbatch_log/classification_dataset_liver_seg_gallbladder_filtered_small"
adversary_param=$"10"
num_vae_steps=$"2"
for seed in 0 255 1000
do
    for mse_gamma in 0.2 0.4 0.8 1
    do 
        EXPT=small_classification_dataset_liver_seg_gallbladder_removed_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_Seed"$seed"
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

        sbatch -J $EXPT -o $STD -t 02-00:00:00 -e $ERR $job_File   
    done;
done;
