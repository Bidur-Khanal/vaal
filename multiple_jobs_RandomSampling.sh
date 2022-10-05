#! bin bash -l

job_File="sbatch_vaal_and_randomsampling.sh" 
# dir="sbatch_log/small_dataset_liver_seg"
# dir="sbatch_log/dataset_liver_seg_mse_2"
dir="sbatch_log/dataset_liver_seg_gallbladder_filtered"
#dir="sbatch_log/dataset_liver_seg_gallbladder_filtered_10_percent_gap"

RAND_SAM_SEED=$"1000"
# EXPT=full_dataset_liver_seg_gallbladder_filtered_10_percent_gap_EXPT_RandomSampling_seed_"$RAND_SAM_SEED"
EXPT=full_dataset_liver_seg_gallbladder_filtered_EXPT_RandomSampling_seed_"$RAND_SAM_SEED"
STD=$dir/STD_RandomSampling_seed_"$RAND_SAM_SEED".out
ERR=$dir/ERR_RandomSampling_seed_"$RAND_SAM_SEED".err
METHOD="RandomSampling"

export EXPT;
export METHOD;
export RAND_SAM_SEED;

sbatch -J $EXPT -o $STD -t 05-00:00:00 -e $ERR $job_File
           
        