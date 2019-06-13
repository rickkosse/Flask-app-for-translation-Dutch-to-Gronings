#!/bin/sh

# cat ../nmt_keras_training/test_Martha.nl ../nmt_keras_training/test_Martha.gro | subword-nmt learn-bpe -s 100 -o {codes_file}
# subword-nmt apply-bpe -c {codes_file} < ../nmt_keras_training/test_Martha.gro | subword-nmt get-vocab > /100/{vocab_file}.L1
# subword-nmt apply-bpe -c {codes_file} < ../nmt_keras_training/test_Martha.nl | subword-nmt get-vocab > /100/{vocab_file}.L2


# subword-nmt learn-joint-bpe-and-vocab --input ./vocab_collection.gro ./vocab_collection.nl -s 100 -o {codes_file} --write-vocabulary ./100/{vocab_file}.L1 ./100/{vocab_file}.L2

# subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L1 --vocabulary-threshold 100 < ./100/training.gro > ./100/nmt/training.gro
# subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L2 --vocabulary-threshold 100 < ./100/training.nl > ./100/nmt/training.nl

subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L1 --vocabulary-threshold 100 < ./output_bpe.txt > ./output_bpe_encoded.txt
# subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L1 --vocabulary-threshold 100 < ./100/nmt/fourty_mono.gro > ./100/nmt/fourty_mono_bpe.gro
# subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L2 --vocabulary-threshold 100 < ./100/dev.nl > ./100/nmt/dev.nl



# cat ../nmt_keras_training/test_Martha.nl ../nmt_keras_training/test_Martha.gro | subword-nmt learn-bpe -s 100 -o {codes_file}
# subword-nmt apply-bpe -c {codes_file} < ../nmt_keras_training/test_Martha.gro | subword-nmt get-vocab > /100/{vocab_file}.L1
# subword-nmt apply-bpe -c {codes_file} < ../nmt_keras_training/test_Martha.nl | subword-nmt get-vocab > /100/{vocab_file}.L2


# subword-nmt learn-joint-bpe-and-vocab --input ./vocab_collection.nl ./vocab_collection.gro  -s 100 -o {codes_file} --write-vocabulary ./100/{vocab_file}.L1 ./100/{vocab_file}.L2

# # subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L1 --vocabulary-threshold 100 < ./100/training.gro > ./100/nmt/training.gro
# # subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L2 --vocabulary-threshold 100 < ./100/training.nl > ./100/nmt/training.nl

# subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L2 --vocabulary-threshold 100 < ./100/nmt/twenty_ted.nl > ./100/nmt/twenty_ted_bpe.nl
# subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L2 --vocabulary-threshold 100 < ./100/nmt/fourty_ted.nl > ./100/nmt/fourty_ted_bpe.nl
# # subword-nmt apply-bpe -c {codes_file} --vocabulary ./100/{vocab_file}.L2 --vocabulary-threshold 100 < ./100/dev.nl > ./100/nmt/dev.nl
