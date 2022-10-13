#!/bin/bash -l


#SBATCH --account airetreat22 --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
module purge
conda activate dplearning
Scale=$"0,0"
Resize=$"256"

# Dataset=$"liver-seg"
# srun -n 1 python3 main.py --scale $Scale --resize $Resize --expt $EXPT --dataset $Dataset --method $METHOD --adversary_param $ADVER_PARAM --random_sampling_seed $RAND_SAM_SEED --mse_gamma2 $MSE_GAMMA --num_vae_steps $NUM_VAE --query_train_epochs 30

# Dataset=$"liver-seg-small"
# srun -n 1 python3 main.py --scale $Scale --resize $Resize --expt $EXPT --dataset $Dataset --method $METHOD --adversary_param $ADVER_PARAM --random_sampling_seed $RAND_SAM_SEED --mse_gamma1 $MSE_GAMMA --num_vae_steps $NUM_VAE --query_train_epochs 100

# Dataset=$"liver-seg-gallbladder-removed"
# srun -n 1 python3 main.py --scale $Scale --resize $Resize --expt $EXPT --dataset $Dataset --method $METHOD --adversary_param $ADVER_PARAM --random_sampling_seed $RAND_SAM_SEED --mse_gamma2 $MSE_GAMMA --num_vae_steps $NUM_VAE --query_train_epochs 30

Dataset=$"liver-seg-gallbladder-removed-class-no-less-than-3"
srun -n 1 python3 main.py --scale $Scale --resize $Resize --expt $EXPT --dataset $Dataset --method $METHOD --adversary_param $ADVER_PARAM --random_sampling_seed $RAND_SAM_SEED --mse_gamma2 $MSE_GAMMA --num_vae_steps $NUM_VAE --seed $SEED --query_train_epochs 50

# echo "$EXPT"
# echo "$METHOD"
# echo "$ADVER_PARAM"
# echo "$MSE_GAMMA"
# echo "$NUM_VAE"
# echo "$RAND_SAM_SEED"

