#!/usr/bin/env python3
from __future__ import print_function
import sys
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/subword-nmt/subword_nmt")
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/")
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras_bpe_100_dutch")
# # sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/nmt_keras/src/keras-wrapper/keras-wrapper")
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras_bpe_100/")
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras_bpe_100/nmt_keras")
# sys.path.remove("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras_bpe_100/src/keras-wrapper/keras-wrapper")
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras_bpe_100_dutch/src/keras-wrapper/keras_wrapper/")
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras_bpe_100_dutch/")

from flask import Flask,render_template,url_for,request, jsonify, abort
from flask_bootstrap import Bootstrap
from proces_and_convert_to_char import process, convert_char, restore
# from proces_and_convert_to_bpe import  restore_bpe

import tensorflow as tf
import sys
import importlib.util
import subprocess, pathlib
from nmtkeras import pred_try
from nmt_keras_bpe_100 import bpe_pred_try
import re
import os 
from subtokenizer import SubTokenizer
import time
from nmt_keras_bpe_100_dutch import bpe_pred_try_nl


# initialize the Flask application and the Keras model
app = Flask(__name__)
Bootstrap(app)

def init():
    # load the pre-trained Keras model
    global bpe_models, bpe_dataset, bpe_args, bpe_params, graph, char_models, char_dataset, char_args, char_params,bpe_models_NL, bpe_dataset_NL, bpe_args_NL, bpe_params_NL

    char_models, char_dataset, char_args, char_params = pred_try.load_all()

    bpe_models, bpe_dataset, bpe_args, bpe_params = bpe_pred_try.bpe_load_all()

    bpe_models_NL, bpe_dataset_NL, bpe_args_NL, bpe_params_NL = bpe_pred_try_nl.bpe_load_all_NL()

    graph = tf.get_default_graph()



def bpe_get_predictions():
    pred= bpe_pred_try.bpe_predicted(bpe_models, bpe_dataset, bpe_args, bpe_params)
    return pred


def bpe_NL_get_predictions():
    pred= bpe_pred_try_nl.bpe_NL_predicted(bpe_models_NL, bpe_dataset_NL, bpe_args_NL, bpe_params_NL)
    return pred

def char_get_predictions():
    pred= pred_try.char_predicted(char_models, char_dataset, char_args, char_params)
    return pred

def write_to_file(comments):
    with open("Output.txt", "w") as text_file:
        text_file.write(comments)


def bpe_write_to_file(comments):
    print("writing to files",comments)
    with open("output_bpe.txt", "w") as text_file:
        text_file.write(comments)


def bpe_NL_write_to_file(comments):
    print("writing to files",comments)
    with open("output_bpe_NL.txt", "w") as text_NL_file:
        text_NL_file.write(comments)


"""Flask env below"""

@app.route('/')
def home():
    return render_template('form.html')

"""Char NL"""
@app.route('/predict_CHAR_nl-gro',methods=['POST'])
def predict_predict_nl_gro():
    print("received nl-gro")
    if request.method == 'POST':
        namequery = request.form['name']
        processed = process(namequery)
        char_encoding= convert_char(processed)
        create_file = write_to_file(char_encoding)
        with graph.as_default():
            output = char_get_predictions()
        output_sen = restore(output)

        return jsonify({'name' : output})
    else:
        return abort(404)

"""Char gro nl"""
@app.route('/predict_CHAR_gro-nl',methods=['POST'])
def predict_gro_nl():
    print("received gro-nl")

    if request.method == 'POST':
        namequery = request.form['name']
        processed = process(namequery)
        char_encoding= convert_char(processed)
        create_file = write_to_file(char_encoding)
        with graph.as_default():
            output = char_get_predictions()
        output_sen = restore(output)

        return jsonify({'name' : output})

    else:
        return abort(404)

"""BPE NL GRO""" 
@app.route('/predict_BPE_nl-gro',methods=['POST'])
def predict_nl_gro_bpe():
    if request.method == 'POST':
        namequery = request.form['name']
        print("request received",namequery)
        processed = process(namequery)
        create_file = bpe_write_to_file(processed)
        tokenize = subprocess.Popen('bash generalsplit.sh',shell=True,stdout=subprocess.PIPE ,stdin=subprocess.PIPE,cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/")
        time.sleep(0.4)
        with graph.as_default():
            output = bpe_get_predictions()
        p = subprocess.Popen('bash restore.sh',shell=True,cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/", stdout=subprocess.PIPE)
        detokenized = p.communicate()[0].decode("utf-8")
        print("predicted translation:", detokenized)
        output_sen = detokenized

        return jsonify({'name' : output_sen})

    else:
        return abort(404)

"""BPE GRO NL""" 
@app.route('/predict_BPE_gro-nl',methods=['POST'])
def predict_gro_nl_bpe():
    if request.method == 'POST':
        print("received GRO-NL")
        namequery = request.form['name']
        print("request received",namequery)
        processed = process(namequery)
        create_file = bpe_NL_write_to_file(processed)
        tokenize = subprocess.Popen('bash generalsplit_NL.sh',shell=True,stdout=subprocess.PIPE ,stdin=subprocess.PIPE,cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/")
        time.sleep(0.4)
        with graph.as_default():
            output = bpe_NL_get_predictions()
        p = subprocess.Popen('bash restore_nl.sh',shell=True,cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/", stdout=subprocess.PIPE)
        detokenized = p.communicate()[0].decode("utf-8")
        print("predicted translation:", detokenized)
        output_sen = detokenized

        return jsonify({'name' : output_sen})

    else:
        return abort(404)


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(debug=True)