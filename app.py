#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/subword-nmt/subword_nmt")
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/")
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/nmt_keras")
# sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/nmt_keras/src/keras-wrapper/keras-wrapper")
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras_bpe_100/")
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras_bpe_100/nmt_keras")
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras_bpe_100/src/keras-wrapper/keras-wrapper")

from flask import Flask,render_template,url_for,request, jsonify, abort
from flask_bootstrap import Bootstrap
from proces_and_convert_to_char import process, convert_char, restore
# from proces_and_convert_to_bpe import  restore_bpe

import tensorflow as tf
import sys
import importlib.util
import subprocess, pathlib
# from nmtkeras import sample_ensemble
from nmtkeras import pred_try
from nmt_keras_bpe_100 import bpe_pred_try
import re
import os 
from subtokenizer import SubTokenizer
import time



# initialize the Flask application and the Keras model
app = Flask(__name__)
Bootstrap(app)

def init():
    # load the pre-trained Keras model
    global bpe_models, bpe_dataset, bpe_args, bpe_params, graph, char_models, char_dataset, char_args, char_params

    char_models, char_dataset, char_args, char_params = pred_try.load_all()

    bpe_models, bpe_dataset, bpe_args, bpe_params = bpe_pred_try.bpe_load_all()

    graph = tf.get_default_graph()



def bpe_get_predictions():
    pred= bpe_pred_try.bpe_predicted(bpe_models, bpe_dataset, bpe_args, bpe_params)
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
    print("Done writing")

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/get_toggled_status') 
def toggled_status():
    global current_status
    current_status = request.args.get('status')

    print("deze loopt achter:",current_status)

    return 'nl_gro' if current_status == 'gro_nl' else 'gro_nl'

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
        print("wachten")
        time.sleep(0.5)
        print("klaar met wachten")
        with graph.as_default():
            output = bpe_get_predictions()
        p = subprocess.Popen('bash restore.sh',shell=True,cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/", stdout=subprocess.PIPE)
        detokenized = p.communicate()[0].decode("utf-8")
        print(detokenized)
        output_sen = detokenized

        return jsonify({'name' : output_sen})

    else:
        return abort(404)


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(debug=True)