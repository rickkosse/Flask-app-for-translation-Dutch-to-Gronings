#!/bin/bash

#SBATCH --time=0-20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --job-name=main
#SBATCH --mail-type=ALL
#SBATCH --mail-user=r.kosse@st.rug.nl

echo "Running on Peregrine"

module load  TensorFlow/1.10.1-fosscuda-2018a-Python-3.6.4

python3 ./main.py 