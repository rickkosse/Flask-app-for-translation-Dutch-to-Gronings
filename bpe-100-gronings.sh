subword-nmt apply-bpe -c {codes_file} --vocabulary {vocab_file}.L1 --vocabulary-threshold 100 < {train_file}.L1 > {train_file}.BPE.L1

