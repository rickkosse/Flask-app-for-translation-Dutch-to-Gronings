cat output_bpe_encoded_translated.txt | gsed -r 's/(@@ )|(@@ ?$)//g'
