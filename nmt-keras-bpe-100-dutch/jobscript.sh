#!/bin/bash

#SBATCH --time=0-24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=10
#SBATCH --job-name=main
#SBATCH --mem=8000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=r.kosse@st.rug.nl

echo "Running on Peregrine"

module load TensorFlow/1.12.0-foss-2018a-Python-3.6.4
export KERAS_BACKEND=tensorflow
python3 ./main.py 
