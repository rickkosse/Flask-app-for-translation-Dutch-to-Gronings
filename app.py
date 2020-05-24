#!/usr/bin/env python3
from __future__ import print_function
import sys, random, string
import os

sys.path.append(os.getcwd() + "/nmtkeras/nmt_keras")
sys.path.append(os.getcwd() + '/nmtkeras')
from flask import Flask, render_template, request, jsonify, abort, session, redirect, url_for
from flask_bootstrap import Bootstrap
from proces_and_convert_to_char import process, convert_char, restore
import tensorflow as tf
import subprocess
import time
from nmtkeras import sample_ensemble
from mongo_db import get_annotation, get_all_annotation, store_anno_in_mongo, get_all_validations
import json
from _collections import defaultdict

# initialize the Flask application and the Keras model
app = Flask(__name__)
Bootstrap(app)

app.secret_key = ''.join(random.choice(string.printable) for _ in range(20))


# to use flask.session, a secret key must be passed to the app instance


def init():
    # load the pre-trained Keras models
    global graph, char_args, char_params, char_models, char_dataset, \
        char_args_nl, char_params_nl, char_models_nl, char_dataset_nl, \
        bpe_args, bpe_params, bpe_models, bpe_dataset, \
        bpe_args_nl, bpe_params_nl, bpe_models_nl, bpe_dataset_nl

    char_args, char_params, char_models, char_dataset = sample_ensemble.load_in("char", "NL_GRO")
    char_args_nl, char_params_nl, char_models_nl, char_dataset_nl = sample_ensemble.load_in("char", "GRO_NL")

    bpe_args, bpe_params, bpe_models, bpe_dataset = sample_ensemble.load_in("BPE", "NL_GRO")
    bpe_args_nl, bpe_params_nl, bpe_models_nl, bpe_dataset_nl = sample_ensemble.load_in("BPE", "GRO_NL")

    graph = tf.get_default_graph()


def get_predictions(args, params, models, dataset):
    pred = sample_ensemble.predict(args, params, models, dataset)
    return pred


def write_to_file(comments):
    with open("Output.txt", "w") as text_file:
        text_file.write(comments)


def write_to_file_NL(comments):
    with open("Output_NL.txt", "w") as text_file:
        text_file.write(comments)


def bpe_write_to_file(comments):
    print("writing to files", comments)
    with open("output_bpe.txt", "w") as text_file:
        text_file.write(comments)


def bpe_NL_write_to_file(comments):
    print("writing to files", comments)
    with open("output_bpe_NL.txt", "w") as text_NL_file:
        text_NL_file.write(comments)


"""Flask env """


@app.route('/')
def home():
    return render_template('vertaal.html')


@app.route('/help', methods=['GET'])
def display_sent():
    """function to return the HTML page to display the sentence"""
    session['count'] = 0
    files = get_all_annotation()
    print("Begin pagina")

    if "read_items" in session:
        read_items = session.get('read_items', None)
        session['read_items'] = read_items
        session["all"] = [(str(instance._id), instance.orginal_gronings) for instance in files if
                          str(instance._id) not in session["read_items"]]
        all = session["all"]

    else:
        all = [(str(instance._id), instance.orginal_gronings) for instance in files]
        session["all"] = all

    return render_template('help.html', all=all, count=session['count'])


@app.route('/get_anno', methods=['GET'])
def get_anno():
    _direction = request.args.get('direction')
    count = session.get('count', None)
    session['count'] = count
    session['count'] = session['count'] + (1 if _direction == 'f' else - 1)
    files = get_all_annotation()
    if 'read_items' in session:

        print("read in anno", session['read_items'])
        read_items = session.get('read_items', None)
        session['read_items'] = read_items
        session['all'] = [(str(instance._id), instance.orginal_gronings) for instance in files if
                          str(instance._id) not in session['read_items']]
    else:
        print("creating session")
        session['all'] = [(str(instance._id), instance.orginal_gronings) for instance in files]

        return jsonify(
            {'forward': str(session['count'] + 1 < len(session["all"])),
             'back': str(bool(session['count'])), "count": session['count'], "all": session['all']})


@app.route('/store_in_mongo', methods=['POST'])
def store_in_mongo():
    if request.method == 'POST':
        anno = request.form['annotation']
        original_id = request.form['original_id']
        storing = store_anno_in_mongo(anno, original_id)
        if "read_items" in session:
            read_items = session.get('read_items', None)
            read_items.append(original_id)
            session['read_items'] = read_items
        else:
            session['read_items'] = [original_id]

        if "all" in session:
            all_items = session.get('all', None)
            session['all'] = [i for i in all_items if i[0] not in session['read_items']]
        count = session.get('count', None)
        session['count'] = count
        print(count, len(session['all']))

        return jsonify({"count": session['count'], "all": session['all']})


@app.route('/validation', methods=['GET'])
def display_validation():
    """function to return the HTML page to display the sentence"""
    session['validation_count'] = 0
    files = get_all_validations()

    if "read_validations" in session:
        read_items = session.get('read_validations', None)
        session['read_validations'] = read_items
        session['all_validations'] = [(str(instance._id), instance.annotated_gronings, instance.orginal_gronings) for instance in files if
                                      str(instance._id) not in session["read_validations"]]
        all = session["all_validations"]

    else:
        all = [(str(instance._id), instance.annotated_gronings, instance.orginal_gronings) for instance in files]
        session["all_validations"] = all

    return render_template('validation.html', all=all, count=session['validation_count'])


@app.route('/get_validations', methods=['GET'])
def get_validations():
    _direction = request.args.get('direction')
    val_count = session.get('validation_count', None)
    session['validation_count'] = val_count
    session['validation_count'] = session['validation_count'] + (1 if _direction == 'f' else - 1)

    files = get_all_validations()

    if 'read_validation' in session:

        read_items = session.get('read_validation', None)
        session['read_validation'] = read_items
        session['all_validations'] = [(str(instance._id), instance.annotated_gronings, instance.orginal_gronings) for
                                      instance in files if
                                      str(instance._id) not in session['read_validation']]
    else:
        session['all_validations'] = [(str(instance._id), instance.annotated_gronings, instance.orginal_gronings) for
                                      instance in files]

    return jsonify(
        {'forward': str(session['validation_count'] + 1 < len(session["all_validations"])),
         'back': str(bool(session['validation_count'])), "count": session['validation_count'],
         "all": session['all_validations']})


"""Char NL"""


@app.route('/predict_CHAR_nl-gro', methods=['POST'])
def predict_predict_nl_gro():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        # Preprocess
        processed = process(translation_query)
        # Tokenize to Char
        char_encoding = convert_char(processed)
        create_file = write_to_file(char_encoding)
        # Translate
        with graph.as_default():
            output = get_predictions(char_args, char_params, char_models, char_dataset)
        # Detokenize and restore
        output_sen = restore(output)
        return jsonify({'translation': output_sen})
    else:
        return abort(404)


"""Char gro nl"""


@app.route('/predict_CHAR_gro-nl', methods=['POST'])
def predict_gro_nl():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        # Preprocess
        processed = process(translation_query)
        # Tokenize to Char
        char_encoding = convert_char(processed)
        create_file = write_to_file_NL(char_encoding)
        # Translate
        with graph.as_default():
            output = get_predictions(char_args_nl, char_params_nl, char_models_nl, char_dataset_nl)
        # Detokenize and restore
        output_sen = restore(output)

        return jsonify({'translation': output_sen})

    else:
        return abort(404)


"""BPE NL GRO"""


@app.route('/predict_BPE_nl-gro', methods=['POST'])
def predict_nl_gro_bpe():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        # Preprocess
        processed = process(translation_query)
        create_file = bpe_write_to_file(processed)
        # Tokenize to BPE
        tokenize = subprocess.Popen('bash generalsplit.sh', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                    cwd=os.getcwd())
        # Break to extend time for writing to files
        time.sleep(0.5)
        # Translate
        with graph.as_default():
            output = get_predictions(bpe_args, bpe_params, bpe_models, bpe_dataset)
        # Detokenize and restore
        p = subprocess.Popen('bash restore.sh', shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE)
        detokenized = p.communicate()[0].decode("utf-8")
        print("predicted translation:", detokenized)
        output_sen = detokenized

        return jsonify({'translation': output_sen})

    else:
        return abort(404)


"""BPE GRO NL"""


@app.route('/predict_BPE_gro-nl', methods=['POST'])
def predict_gro_nl_bpe():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        # Preprocess
        processed = process(translation_query)
        create_file = bpe_NL_write_to_file(processed)
        # Tokenize to BPE
        tokenize = subprocess.Popen('bash generalsplit_NL.sh', shell=True, stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE, cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/")
        # Break to extend time for writing to files
        time.sleep(1.5)
        # Translate
        with graph.as_default():
            output = get_predictions(bpe_args_nl, bpe_params_nl, bpe_models_nl, bpe_dataset_nl)
        # Detokenize and restore
        p = subprocess.Popen('bash restore_nl.sh', shell=True,
                             cwd="/Users/rickkosse/Documents/RUG/flask_translation_env/", stdout=subprocess.PIPE)
        detokenized = p.communicate()[0].decode("utf-8")
        print("predicted translation:", detokenized)
        output_sen = detokenized

        return jsonify({'translation': output_sen})

    else:
        return abort(404)


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..." \
           "please wait until server has fully started"))
    init()
    app.run(debug=True, use_reloader=False)
