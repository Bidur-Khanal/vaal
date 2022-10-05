#! bin bash -l

job_File="sbatch_vaal_and_randomsampling.sh" 
# dir="sbatch_log/small_dataset_liver_seg"
# dir="sbatch_log/dataset_liver_seg_mse_2"
dir="sbatch_log/dataset_liver_seg_gallbladder_filtered"
# dir="sbatch_log/dataset_liver_seg_gallbladder_filtered_10_percent_gap"

# EXPT=full_dataset_liver_seg_gallbladder_filtered_10_percent_gap_EXPT_VAAL
EXPT=full_dataset_liver_seg_gallbladder_filtered_10_percent_gap_EXPT_VAAL
STD=$dir/STD_VAAL.out
ERR=$dir/ERR_VAAL.err
METHOD="VAAL"
RAND_SAM_SEED=$"0"

export EXPT;
export METHOD;
export RAND_SAM_SEED;

sbatch -J $EXPT -o $STD -t 05-00:00:00 -e $ERR $job_File
           
        