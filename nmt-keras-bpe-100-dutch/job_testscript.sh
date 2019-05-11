#!/bin/bash


#SBATCH --time=0-05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=main
#SBATCH --mail-type=ALL
#SBATCH --mail-user=r.kosse@st.rug.nl

module load  TensorFlow/1.10.1-fosscuda-2018a-Python-3.6.4


python3 ./sample_ensemble.py -m trained_models/EuTrans_gronl_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_58 -ds datasets/Dataset_EuTrans_gronl.pkl  --text examples/EuTrans/twenty_mono_bpe.gro

