#!/usr/bin/env python3
from __future__ import print_function
import sys
import os
<<<<<<< HEAD
sys.path.append(os.getcwd()+"/nmtkeras/nmt_keras")
from flask import Flask,render_template,url_for,request, jsonify, abort
from flask_bootstrap import Bootstrap
from proces_and_convert_to_char import process, convert_char, restore
import tensorflow as tf
import subprocess, pathlib
=======
sys.path.append(os.getcwd()+'/nmtkeras')
print(os.getcwd())
# sys.path.append('/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras')
from flask import Flask,render_template,request, jsonify, abort
from flask_bootstrap import Bootstrap
from proces_and_convert_to_char import process, convert_char, restore
import tensorflow as tf
import subprocess
>>>>>>> temp-branch
import time
from nmtkeras import sample_ensemble

# initialize the Flask application and the Keras model
app = Flask(__name__)
Bootstrap(app)

def init():
    # load the pre-trained Keras models
    global graph, char_args, char_params, char_models, char_dataset, \
        char_args_nl, char_params_nl, char_models_nl, char_dataset_nl, \
        bpe_args, bpe_params, bpe_models, bpe_dataset, \
        bpe_args_nl, bpe_params_nl, bpe_models_nl, bpe_dataset_nl

    char_args, char_params, char_models, char_dataset = sample_ensemble.load_in("char", "NL_GRO")
    char_args_nl, char_params_nl, char_models_nl, char_dataset_nl = sample_ensemble.load_in("char","GRO_NL")

    bpe_args, bpe_params, bpe_models, bpe_dataset = sample_ensemble.load_in("BPE","NL_GRO")
    bpe_args_nl, bpe_params_nl, bpe_models_nl, bpe_dataset_nl = sample_ensemble.load_in("BPE", "GRO_NL")


    graph = tf.get_default_graph()


def get_predictions(args, params, models, dataset):
    pred= sample_ensemble.predict(args, params, models, dataset)
    return pred

#

def write_to_file(comments):
    with open("Output.txt", "w") as text_file:
        text_file.write(comments)


def write_to_file_NL(comments):
    with open("Output_NL.txt", "w") as text_file:
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
            output = get_predictions(char_args, char_params, char_models, char_dataset)
        # Detokenize and restore
        output_sen = restore(output)
        return jsonify({'translation': output_sen})
    else:
        return abort(404)
#
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
        create_file = write_to_file_NL(char_encoding)
        #Translate
        with graph.as_default():
            output = get_predictions(char_args_nl, char_params_nl, char_models_nl, char_dataset_nl)
        # Detokenize and restore
        output_sen = restore(output)

        return jsonify({'translation': output_sen})

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
        tokenize = subprocess.Popen('bash generalsplit.sh',shell=True,stdout=subprocess.PIPE ,stdin=subprocess.PIPE,cwd=os.getcwd())
        # Break to extend time for writing to files
        time.sleep(0.5)
        #Translate
        with graph.as_default():
            output = get_predictions(bpe_args, bpe_params, bpe_models, bpe_dataset)
        # Detokenize and restore
        print("translated",output)
        p = subprocess.Popen('bash restore.sh',shell=True,cwd=os.getcwd(), stdout=subprocess.PIPE)
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
        time.sleep(1.5)
        #Translate
        with graph.as_default():
            output = get_predictions(bpe_args_nl, bpe_params_nl, bpe_models_nl, bpe_dataset_nl)
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