#!/bin/bash
#SBATCH -p gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 48:00:00
#SBATCH -J image_deblur
#SBATCH -A cwr109
#SBATCH --export=ALL

export SLURM_EXPORT_ENV=ALL

module purge
module load gpu
module load slurm

source /home/zhen1997/anaconda3/etc/profile.d/conda.sh
conda activate py39

cd "/home/zhen1997/image_deblurring"
# python train.py
# wandb agent oilab/image_deblurring/ztpb3mm6
python train_nafnet.py