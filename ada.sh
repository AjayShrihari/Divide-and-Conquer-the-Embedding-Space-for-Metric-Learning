#!/bin/bash
#SBATCH -A $USER
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END

module add cuda/8.0
module add cudnn/7-cuda-8.0

# add pyth0pn training file script in next line
# python .....



# Running Instructions:
# Before running, ensure that the bash.sh runs else it will crash the batch job
# Run : Sbatch ada.sh