#! bin bash -l

job_File="sbatch_run.sh" 
dir="sbatch_log"
for adversary_param in  1 10 25
do 
    for mse_gamma in 0.1 0.2 0.5 1
    do 
        for num_vae_steps in 1 2 3
        do 
            EXPT=EXPT_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_num_vae_steps"$num_vae_steps"
            STD=$dir/STD_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_num_vae_steps"$num_vae_steps".out
            ERR=$dir/ERR_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_num_vae_steps"$num_vae_steps".err
            METHOD="multimodal_VAAL"
            ADVER_PARAM=$adversary_param
            MSE_GAMMA=$mse_gamma
            NUM_VAE=$num_vae_steps
            RAND_SAM_SEED=$"0"

            export EXPT;
            export METHOD;
            export ADVER_PARAM;
            export MSE_GAMMA;
            export NUM_VAE;
            export RAND_SAM_SEED;

            sbatch -J $EXPT -o $STD -t 03-05:00:00 -e $ERR $job_File
           
        done;
    done;
done