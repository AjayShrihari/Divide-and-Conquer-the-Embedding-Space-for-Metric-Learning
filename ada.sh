#!/bin/bash
#SBATCH -A research	
#SBATCH -n 32
#SBATCH --mincpus=32
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=ajay.shrihari@students.iiit.ac.in
#SBATCH --mail-type=ALL
module add cuda/10.2
module add cudnn/7.6.5-cuda-10.2

# add pyth0pn training file script in next line
python3 init.py



# Running Instructions:
# Before running, ensure that the bash.sh runs else it will crash the batch job
# Run : Sbatch ada.sh


# Running Instructions:
# Before running, ensure that the bash.sh runs else it will crash the batch job
# Run : Sbatch ada.sh
