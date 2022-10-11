#! bin bash -l

job_File="rendered_sbatch_vaal_and_randomsampling.sh" 
dir="sbatch_log/rendered_classification_dataset_liver_seg_gallbladder_filtered"

adversary_param=$"10"
num_vae_steps=$"2"
for seed in 0 255 1000
do
    EXPT=rendered_classification_dataset_liver_seg_gallbladder_removed_VAAL_seed_"$seed"
    STD=$dir/STD_VAAL_seed_"$seed".out
    ERR=$dir/ERR_VAAL_seed_"$seed".err
    METHOD="VAAL"
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


    sbatch -J $EXPT -o $STD -t 05-00:00:00 -e $ERR $job_File
done;
           
        