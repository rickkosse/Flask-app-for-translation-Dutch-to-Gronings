# -*- coding: utf-8 -*-
from __future__ import print_function

import logging
import ast
from keras_wrapper.extra.read_write import pkl2dict
from nmtkeras.nmt_keras import check_params
from nmtkeras.nmt_keras.apply_model import sample_ensemble, char_loading, bpe_loading
import os

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S")
logger = logging.getLogger(__name__)


def loadmodel(encoding, args):
    if encoding == "char":
        model, dataset = char_loading(args)

    else:
        model, dataset = bpe_loading(args)

    return model, dataset


def parse_args_char(direction):
    if direction == "NL_GRO":
        argument_dict = {"config": os.getcwd() + "/nmtkeras/char_config.pkl", "verbose": 0, "dest": None,
                         "splits": ['val'], "text": os.getcwd() + "/Output.txt",
                         "dataset": os.getcwd() + "/nmtkeras/datasets/char/Dataset_EuTrans_nlgro.pkl", "n_best": None,
                         "weights": [], "glossary": None, "models": [
                os.getcwd() + "/nmtkeras/trained_models_char/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_66"],
                         "changes": ""}

        return argument_dict

    else:
        argument_dict = {"config": os.getcwd() + "/nmtkeras/char_config_nl.pkl", "verbose": 0, "dest": None,
                         "splits": ['val'], "text": (os.getcwd() + "/Output_NL.txt",),
                         "dataset": os.getcwd() + "/nmtkeras/datasets/char/Dataset_EuTrans_gronl.pkl", "n_best": None,
                         "weights": [], "glossary": None, "models": [
                os.getcwd() + "/nmtkeras/trained_models_char"
                              "/EuTrans_gronl_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_22"],
                         "changes": ""}

        return argument_dict


def parse_args_bpe(direction):

    if direction == "NL_GRO":
        argument_dict = {"config": os.getcwd() + "/nmtkeras/bpe_config.pkl", "verbose": 0,
                         "dest": "./output_bpe_encoded_translated.txt", "splits": ['val'],
                         "text": os.getcwd() + "/output_bpe_encoded.txt",
                         "dataset": os.getcwd() + "/nmtkeras/datasets/bpe/Dataset_EuTrans_nlgro.pkl", "n_best": None,
                         "weights": [], "glossary": None, "models": [os.getcwd() + "/nmtkeras/trained_models_bpe"
                                                                                   "/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_43"],
                         "changes": ""}

        return argument_dict
    else:
        argument_dict = {"config": os.getcwd() + "/nmtkeras/bpe_config_nl.pkl", "verbose": 0,
                         "dest": "./output_bpe_encoded_translated_NL.txt", "splits": ['val'],
                         "text": os.getcwd() + "/output_bpe_nl_encoded.txt",
                         "dataset": os.getcwd() + "/nmtkeras/datasets/bpe/Dataset_EuTrans_gronl.pkl", "n_best": None,
                         "weights": [], "glossary": None, "models": [os.getcwd() + "/nmtkeras/trained_models_bpe"
                                                                                   "/EuTrans_gronl_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_17"],
                         "changes": ""}

        return argument_dict


def predict(text, args, params, models, dataset):
    result = sample_ensemble(text, args, params, models, dataset)

    return result


def load_in(encoding, direction):
    if encoding == "char":
        print(encoding)
        args = parse_args_char(direction)
    else:
        args = parse_args_bpe(direction)
    print(args)
    if args["config"] is None:
        logging.info("Reading parameters from config.py")
        from config import load_parameters
        params = load_parameters()
    else:
        logging.info("Loading parameters from %s" % str(args["config"]))
        params = pkl2dict(args["config"])
    try:
        for arg in args["changes"]:
            try:
                k, v = arg["split"]('=')
            except ValueError:
                print('Overwritten arguments must have the form key=Value. \n Currently are: %s' % str(args["changes"]))
                exit(1)
            try:
                params[k] = ast.literal_eval(v)
            except ValueError:
                params[k] = v
    except ValueError:
        print('Error processing arguments: (', k, ",", v, ")")
        exit(2)
    params = check_params(params)
    model, dataset = loadmodel(encoding, args)

    # dit moet een functie worden
    # sample_ensemble(args, params, models, dataset)
    return args, params, model, dataset
