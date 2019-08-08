#!/bin/sh


subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L1 --vocabulary-threshold 100 < ./output_bpe.txt > ./output_bpe_encoded.txt
