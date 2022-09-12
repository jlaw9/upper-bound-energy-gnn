#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=2-00
#SBATCH --job-name=learning_curves
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/jlaw/gpu_model_lc.%j.out

source ~/.bashrc
module load cudnn/8.1.1/cuda-11.2
module load gcc
conda activate /home/jlaw/.conda-envs/crystals_nfp0_3

run_id=1

for ((i = 0 ; i < 10 ; i++)); do
    srun --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --exclusive \
        python train_model.py $run_id $i &
done

wait
