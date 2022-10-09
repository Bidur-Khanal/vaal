#! bin bash -l

# job_File="sbatch_run.sh" 
# # dir="sbatch_log/small_dataset_liver_seg"
# # dir="sbatch_log/dataset_liver_seg_mse_2"
# dir="sbatch_log/dataset_liver_seg_gallbladder_removed_class_no_less_than_3"
# for adversary_param in  1 10 25
# do 
#     for mse_gamma in 0.1 0.2 0.5 1
#     do 
#         for num_vae_steps in 1 2 3
#         do 
#             EXPT=full_dataset_liver_seg_gallbladder_filtered_10_percent_gap_EXPT_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_num_vae_steps"$num_vae_steps"
#             STD=$dir/STD_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_num_vae_steps"$num_vae_steps".out
#             ERR=$dir/ERR_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_num_vae_steps"$num_vae_steps".err
#             METHOD="multimodal_VAAL"
#             ADVER_PARAM=$adversary_param
#             MSE_GAMMA=$mse_gamma
#             NUM_VAE=$num_vae_steps
#             RAND_SAM_SEED=$"0"

#             export EXPT;
#             export METHOD;
#             export ADVER_PARAM;
#             export MSE_GAMMA;
#             export NUM_VAE;
#             export RAND_SAM_SEED;

#             sbatch -J $EXPT -o $STD -t 05-00:00:00 -e $ERR $job_File
           
#         done;
#     done;
# done

job_File="sbatch_run.sh" 
# dir="sbatch_log/small_dataset_liver_seg"
# dir="sbatch_log/dataset_liver_seg_mse_2"
dir="sbatch_log/dataset_liver_seg_gallbladder_removed_class_no_less_than_3"
adversary_param=$"25"
num_vae_steps=$"2"
for seed in 0 255 1000
do
    for mse_gamma in 0.2 0.4 0.8 1
    do 
        EXPT=dataset_liver_seg_gallbladder_removed_class_no_less_than_3_EXPT_MULTI_VAAL_adver"$adversary_param"_mse"$mse_gamma"_Seed"$seed"
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
    done;
done;
