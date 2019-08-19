#!/usr/bin/env python3

# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import logging
import ast
import os
from nmt_keras import check_params
from nmt_keras.apply_model import sample_ensemble, char_loading, bpe_loading
from keras_wrapper.extra.read_write import pkl2dict


logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S")
logger = logging.getLogger(__name__)


def loadmodel(encoding, args):
    if encoding == "char":
        model, dataset = char_loading(args)

    else:
        model, dataset = bpe_loading(args)

    return model, dataset


def parse_args_char(direction):
    if direction =="NL_GRO":
        parser = argparse.ArgumentParser("Use several translation models for obtaining predictions from a source text file.")
        parser.add_argument("-ds", "--dataset", required=False, default=os.getcwd()+"/nmtkeras/datasets/char/Dataset_EuTrans_nlgro.pkl", help="Dataset instance with data")
        parser.add_argument("-t", "--text", required=False, default=os.getcwd()+"/Output.txt", help="Text file with source sentences")
        parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'], help="Splits to sample. "
                                                                                               "Should be already included"
                                                                                               "into the dataset object.")
        parser.add_argument("-d", "--dest", required=False, help="File to save translations in. If not specified, "
                                                                 "translations are outputted in STDOUT.")
        parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
        parser.add_argument("-c", "--config", required=False, default=os.getcwd()+"/nmtkeras/char_config.pkl" ,help="Config pkl for loading the model configuration. "
                                                                   "If not specified, hyperparameters "
                                                                   "are read from config.py")
        parser.add_argument("-n", "--n-best", action="store_true", default=False, help="Write n-best list (n = beam size)")
        parser.add_argument("-w", "--weights", nargs="*", help="Weight given to each model in the ensemble. You should provide the same number of weights than models."
                                                               "By default, it applies the same weight to each model (1/N).", default=[])
        parser.add_argument("-g", "--glossary", required=False, help="Glossary file for overwriting translations.")
        parser.add_argument("-m", "--models", nargs="+", required=False, default=[os.getcwd()+"/nmtkeras/trained_models_char/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_66"],help="Path to the models")
        parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                            default="")
        return parser.parse_args()

    else:
        parser = argparse.ArgumentParser(
            "Use several translation models for obtaining predictions from a source text file.")
        parser.add_argument("-ds", "--dataset", required=False,
                            default=os.getcwd()+"/nmtkeras/datasets/char/Dataset_EuTrans_gronl.pkl",
                            help="Dataset instance with data")
        parser.add_argument("-t", "--text", required=False,
                            default=os.getcwd()+"/Output_NL.txt",
                            help="Text file with source sentences")
        parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'], help="Splits to sample. "
                                                                                               "Should be already included"
                                                                                               "into the dataset object.")
        parser.add_argument("-d", "--dest", required=False, help="File to save translations in. If not specified, "
                                                                 "translations are outputted in STDOUT.")
        parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
        parser.add_argument("-c", "--config", required=False,
                            default=os.getcwd()+"/nmtkeras/char_config_nl.pkl",
                            help="Config pkl for loading the model configuration. "
                                 "If not specified, hyperparameters "
                                 "are read from config.py")
        parser.add_argument("-n", "--n-best", action="store_true", default=False, help="Write n-best list (n = beam size)")
        parser.add_argument("-w", "--weights", nargs="*",
                            help="Weight given to each model in the ensemble. You should provide the same number of weights than models."
                                 "By default, it applies the same weight to each model (1/N).", default=[])
        parser.add_argument("-g", "--glossary", required=False, help="Glossary file for overwriting translations.")
        parser.add_argument("-m", "--models", nargs="+", required=False, default=[os.getcwd()+"/nmtkeras/trained_models_char/EuTrans_gronl_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_22"], help="Path to the models")
        parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                            default="")
        return parser.parse_args()



def parse_args_bpe(direction):
    if direction == "NL_GRO":
        parser = argparse.ArgumentParser("Use several translation models for obtaining predictions from a source text file.")
        parser.add_argument("-ds", "--dataset", required=False, default=os.getcwd()+"/nmtkeras/datasets/bpe/Dataset_EuTrans_nlgro.pkl", help="Dataset instance with data")
        parser.add_argument("-t", "--text", required=False, default=os.getcwd()+"/output_bpe_encoded.txt", help="Text file with source sentences")
        parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'], help="Splits to sample. "
                                                                                               "Should be already included"
                                                                                               "into the dataset object.")
        parser.add_argument("-d", "--dest", default="./output_bpe_encoded_translated.txt",  help="File to save translations in. If not specified, "
                                                                 "translations are outputted in STDOUT.")
        parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
        parser.add_argument("-c", "--config", required=False, default=os.getcwd()+"/nmtkeras/bpe_config.pkl", help="Config pkl for loading the model configuration. "
                                                                   "If not specified, hyperparameters "
                                                                   "are read from config.py")
        parser.add_argument("-n", "--n-best", action="store_true", default=False, help="Write n-best list (n = beam size)")
        parser.add_argument("-w", "--weights", nargs="*", help="Weight given to each model in the ensemble. You should provide the same number of weights than models."
                                                               "By default, it applies the same weight to each model (1/N).", default=[])
        parser.add_argument("-g", "--glossary", required=False, help="Glossary file for overwriting translations.")
        parser.add_argument("-m", "--models", nargs="+", required=False,default=[os.getcwd()+"/nmtkeras/trained_models_bpe/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_43"],  help="Path to the models")
        parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                            default="")
        return parser.parse_args()
    else:
        parser = argparse.ArgumentParser(
            "Use several translation models for obtaining predictions from a source text file.")
        parser.add_argument("-ds", "--dataset", required=False, default=os.getcwd()+"/nmtkeras/datasets/bpe/Dataset_EuTrans_gronl.pkl",
                            help="Dataset instance with data")
        parser.add_argument("-t", "--text", required=False,
                            default=os.getcwd()+"/output_bpe_nl_encoded.txt",
                            help="Text file with source sentences")
        parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'], help="Splits to sample. "
                                                                                               "Should be already included"
                                                                                               "into the dataset object.")
        parser.add_argument("-d", "--dest", default="./output_bpe_encoded_translated_NL.txt",
                            help="File to save translations in. If not specified, "
                                 "translations are outputted in STDOUT.")
        parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
        parser.add_argument("-c", "--config", required=False,
                            default=os.getcwd()+"/nmtkeras/bpe_config_nl.pkl",
                            help="Config pkl for loading the model configuration. "
                                 "If not specified, hyperparameters "
                                 "are read from config.py")
        parser.add_argument("-n", "--n-best", action="store_true", default=False,
                            help="Write n-best list (n = beam size)")
        parser.add_argument("-w", "--weights", nargs="*",
                            help="Weight given to each model in the ensemble. You should provide the same number of weights than models."
                                 "By default, it applies the same weight to each model (1/N).", default=[])
        parser.add_argument("-g", "--glossary", required=False, help="Glossary file for overwriting translations.")
        parser.add_argument("-m", "--models", nargs="+", required=False, default=[os.getcwd()+"/nmtkeras/trained_models_bpe/EuTrans_gronl_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_17"],
                            help="Path to the models")
        parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                            default="")
        return parser.parse_args()

def predict(args, params, models, dataset):

    result = sample_ensemble(args, params, models, dataset)

    return result

def load_in(encoding, direction):
    if encoding == "char":
        args = parse_args_char(direction)
    else:
        args = parse_args_bpe(direction)
    if args.config is None:
        logging.info("Reading parameters from config.py")
        from config import load_parameters
        params = load_parameters()
    else:
        logging.info("Loading parameters from %s" % str(args.config))
        params = pkl2dict(args.config)
    try:
        for arg in args.changes:
            try:
                k, v = arg.split('=')
            except ValueError:
                print ('Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args.changes))
                exit(1)
            try:
                params[k] = ast.literal_eval(v)
            except ValueError:
                params[k] = v
    except ValueError:
        print ('Error processing arguments: (', k, ",", v, ")")
        exit(2)
    params = check_params(params)
    model, dataset = loadmodel(encoding, args)

    # dit moet een functie worden
    # sample_ensemble(args, params, models, dataset)
    return args, params, model, dataset


if __name__ == "__main__":
    load_in()
