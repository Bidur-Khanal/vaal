#! bin bash -l

job_File="sbatch_vaal_and_randomsampling.sh" 
# dir="sbatch_log/small_dataset_liver_seg"
# dir="sbatch_log/dataset_liver_seg_mse_2"
dir="sbatch_log/dataset_liver_seg_gallbladder_removed_class_no_less_than_3_small"
adversary_param=$"25"
num_vae_steps=$"2"
for seed in 0 255 1000 2550 100
do
    EXPT=final_small_dataset_liver_seg_gallbladder_removed_class_no_less_than_3_EXPT_depth_VAAL_seed_"$seed"
    STD=$dir/STD_depth_VAAL_seed_"$seed".out
    ERR=$dir/ERR_depth_VAAL_seed_"$seed".err
    METHOD="depth_VAAL"
    ADVER_PARAM=$adversary_param
    NUM_VAE=$num_vae_steps
    RAND_SAM_SEED=$seed
    SEED=$seed


    export EXPT;
    export METHOD;
    export ADVER_PARAM;
    export NUM_VAE;
    export RAND_SAM_SEED;
    export SEED;


    sbatch -J $EXPT -o $STD -t 02-00:00:00 -e $ERR $job_File
done;
           
        