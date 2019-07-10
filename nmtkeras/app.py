#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import logging
import ast
from keras_wrapper.extra.read_write import pkl2dict
from nmt_keras import check_params
from nmt_keras.apply_model import sample_ensemble
from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
from proces_and_convert_to_char import process, convert_char
import tensorflow as tf
import sys
import importlib.util
import subprocess
import os 



logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Use several translation models for obtaining predictions from a source text file.")
    parser.add_argument("-ds", "--dataset", required=True, help="Dataset instance with data")
    parser.add_argument("-t", "--text", required=True, help="Text file with source sentences")
    parser.add_argument("-s", "--splits", nargs='+', required=False, default=['val'], help="Splits to sample. "
                                                                                           "Should be already included"
                                                                                           "into the dataset object.")
    parser.add_argument("-d", "--dest", required=False, help="File to save translations in. If not specified, "
                                                             "translations are outputted in STDOUT.")
    parser.add_argument("-v", "--verbose", required=False, default=0, type=int, help="Verbosity level")
    parser.add_argument("-c", "--config", required=False, help="Config pkl for loading the model configuration. "
                                                               "If not specified, hyperparameters "
                                                               "are read from config.py")
    parser.add_argument("-n", "--n-best", action="store_true", default=False, help="Write n-best list (n = beam size)")
    parser.add_argument("-w", "--weights", nargs="*", help="Weight given to each model in the ensemble. You should provide the same number of weights than models."
                                                           "By default, it applies the same weight to each model (1/N).", default=[])
    parser.add_argument("-g", "--glossary", required=False, help="Glossary file for overwriting translations.")
    parser.add_argument("-m", "--models", nargs="+", required=True, help="Path to the models")
    parser.add_argument("-ch", "--changes", nargs="*", help="Changes to the config. Following the syntax Key=Value",
                        default="")
    return parser.parse_args()



def get_predictions():
    p= subprocess.Popen("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras/sample_ensemble.py -m trained_models/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_16 -ds datasets/Dataset_EuTrans_nlgro.pkl  --text ../Output.txt", shell=True)
    output = p.communicate()




def write_to_file(comments):
    with open("Output.txt", "w") as text_file:
        text_file.write(comments)

app = Flask(__name__)
Bootstrap(app)

global graph
graph = tf.get_default_graph()
# model = loadModel('/models','53.h5')


@app.route('/')
def home():
    return render_template('../home.html')

@app.route('/predict',methods=['POST'])
def predict():

    #Alternative Usage of Saved Model
    # ytb_model = open("naivebayes_spam_model.pkl","rb")
    # clf = joblib.load(ytb_model)

    if request.method == 'POST':

        namequery = request.form['namequery']

        processed = process(namequery)
        char_encoding= convert_char(processed)
        create_file = write_to_file(char_encoding)
        prediction = get_predictions()
   

    return render_template('result.html',prediction = char_encoding)



if __name__ == '__main__':
    app.run(debug=True)


    args = parse_args()
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
    sample_ensemble(args, params)

