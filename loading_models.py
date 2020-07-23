from nmtkeras.sample_ensemble import load_in

char_args, char_params, char_models, char_dataset = load_in("char", "NL_GRO")
char_args_nl, char_params_nl, char_models_nl, char_dataset_nl = load_in("char", "GRO_NL")

bpe_args, bpe_params, bpe_models, bpe_dataset = load_in("BPE", "NL_GRO")
bpe_args_nl, bpe_params_nl, bpe_models_nl, bpe_dataset_nl = load_in("BPE", "GRO_NL")