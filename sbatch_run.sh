#!/bin/bash -l


#SBATCH --account airetreat22 --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:p4:1
#SBATCH --mem=24g
module purge
conda activate dplearning
Scale=$"0,0"
Resize=$"256"
Dataset=$"liver-seg"
srun -n 1 python3 main.py --scale $Scale --resize $Resize --expt $EXPT --dataset $Dataset --method $METHOD --adversary_param $ADVER_PARAM --random_sampling_seed $RAND_SAM_SEED --mse_gamma1 $MSE_GAMMA --num_vae_steps $NUM_VAE --query_train_epochs 30

# echo "$EXPT"
# echo "$METHOD"
# echo "$ADVER_PARAM"
# echo "$MSE_GAMMA"
# echo "$NUM_VAE"
# echo "$RAND_SAM_SEED"

