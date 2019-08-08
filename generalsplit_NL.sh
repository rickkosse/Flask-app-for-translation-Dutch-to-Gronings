#!/bin/sh

subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L2 --vocabulary-threshold 100 < ./output_bpe_NL.txt > ./output_bpe_nl_encoded.txt
