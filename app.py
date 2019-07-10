#!/usr/bin/env python3
from __future__ import print_function
import sys
from flask import Flask,render_template,url_for,request, jsonify, abort
from flask_bootstrap import Bootstrap
from proces_and_convert_to_char import process, convert_char, restore
import tensorflow as tf
import sys
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
    # load the pre-trained Keras models
    global bpe_models, bpe_dataset, bpe_args, bpe_params, graph, char_models, \
            char_dataset, char_args, char_params,bpe_models_NL, bpe_dataset_NL, \
            bpe_args_NL, bpe_params_NL

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


"""Flask env """
@app.route('/')
def home():
    return render_template('form.html')

"""Char NL"""
@app.route('/predict_CHAR_nl-gro',methods=['POST'])
def predict_predict_nl_gro():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        #Preprocess
        processed = process(translation_query)
        # Tokenize to Char
        char_encoding= convert_char(processed)
        create_file = write_to_file(char_encoding)
        #Translate
        with graph.as_default():
            output = char_get_predictions()
        # Detokenize and restore
        output_sen = restore(output)
        return jsonify({'name' : output})
    else:
        return abort(404)

"""Char gro nl"""
@app.route('/predict_CHAR_gro-nl',methods=['POST'])
def predict_gro_nl():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        #Preprocess
        processed = process(translation_query)
        # Tokenize to Char
        char_encoding= convert_char(processed)
        create_file = write_to_file(char_encoding)
        #Translate
        with graph.as_default():
            output = char_get_predictions()
        # Detokenize and restore
        output_sen = restore(output)

        return jsonify({'translation' : output})

    else:
        return abort(404)

"""BPE NL GRO""" 
@app.route('/predict_BPE_nl-gro',methods=['POST'])
def predict_nl_gro_bpe():
    # Validation
    if request.method == 'POST': 
        translation_query = request.form['translation']
        #Preprocess
        processed = process(translation_query)
        create_file = bpe_write_to_file(processed)
        #Tokenize to BPE
        tokenize = subprocess.Popen('bash generalsplit.sh',shell=True,stdout=subprocess.PIPE ,stdin=subprocess.PIPE,cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/")
        # Break to extend time for writing to files
        time.sleep(0.5)
        #Translate
        with graph.as_default():
            output = bpe_get_predictions()
        # Detokenize and restore
        p = subprocess.Popen('bash restore.sh',shell=True,cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/", stdout=subprocess.PIPE)
        detokenized = p.communicate()[0].decode("utf-8")
        print("predicted translation:", detokenized)
        output_sen = detokenized

        return jsonify({'translation' : output_sen})

    else:
        return abort(404)

"""BPE GRO NL""" 
@app.route('/predict_BPE_gro-nl',methods=['POST'])
def predict_gro_nl_bpe():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        #Preprocess
        processed = process(translation_query)
        create_file = bpe_NL_write_to_file(processed)
        #Tokenize to BPE
        tokenize = subprocess.Popen('bash generalsplit_NL.sh',shell=True,stdout=subprocess.PIPE ,stdin=subprocess.PIPE,cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/")
        # Break to extend time for writing to files
        time.sleep(0.5)
        #Translate
        with graph.as_default():
            output = bpe_NL_get_predictions()
        # Detokenize and restore
        p = subprocess.Popen('bash restore_nl.sh',shell=True,cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/", stdout=subprocess.PIPE)
        detokenized = p.communicate()[0].decode("utf-8")
        print("predicted translation:", detokenized)
        output_sen = detokenized

        return jsonify({'translation' : output_sen})

    else:
        return abort(404)

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."\
            "please wait until server has fully started"))
    init()
    app.run(debug=True)