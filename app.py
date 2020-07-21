#!/usr/bin/env python3
from __future__ import print_function

import random
import string
import subprocess
from random import shuffle
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, abort, session
from mongo_db import store_valid_in_mongo, replete_valid_db, store_anno_in_mongo, replete_anno_db
from nmtkeras.sample_ensemble import *
from proces_and_convert_to_char import process, convert_char, restore
from byte_pair_loading import bpe, bpe_nl, graph
from loading_models import *
# initialize the Flask application and the Keras model
app = Flask(__name__)

app.secret_key = ''.join(random.choice(string.printable) for _ in range(20))
# to use flask.session, a secret key must be passed to the app instance


def bpe_encode(input):
    return "".join(bpe.process_line(input))


def bpe_encode_nl(input):
    return "".join(bpe_nl.process_line(input))


def get_predictions(text, args, params, models, dataset):
    pred = predict(text, args, params, models, dataset)
    return pred

"""Flask env """


@app.route('/')
def home():
    return render_template('vertaal.html')


def update_anno(read_items=None):
    if read_items is None:
        read_items = []
    files = replete_anno_db(read_items)
    all = [(str(instance._id), instance.orginal_gronings) for instance in files if
           str(instance._id) not in read_items]
    return all


def update_valid(read_items=None):
    if read_items is None:
        read_items = []
    files = replete_valid_db(read_items)
    all = [(str(instance._id), instance.annotated_gronings, instance.orginal_gronings) for
           instance in files if
           str(instance._id) not in read_items]
    return all


@app.route('/help', methods=['GET'])
def display_sent():
    """function to return the HTML page to display the sentence"""
    session['count'] = 0

    if "read_items" in session:
        read_items = session.get('read_items', None)
        session['read_items'] = read_items
        all = update_anno(read_items)

    else:
        all = update_anno()
    shuffle(all)

    return render_template('help.html', all=all, count=session['count'])


@app.route('/get_anno', methods=['GET'])
def get_anno():
    _direction = request.args.get('direction')
    count = session.get('count', None)
    session['count'] = count
    session['count'] = session['count'] + (1 if _direction == 'f' else - 1)
    if "read_items" in session:
        read_items = session.get('read_items', None)
        session['read_items'] = read_items
        all = update_anno(read_items)

    else:
        all = update_anno()

    return jsonify(
        {'forward': str(session['count'] + 1 < len(all)),
         'back': str(bool(session['count'])), "count": session['count'], "all": all})


@app.route('/store_in_mongo', methods=['POST'])
def store_in_mongo():
    if request.method == 'POST':
        anno = request.form['annotation']
        original_id = request.form['original_id']
        store_anno_in_mongo(anno, original_id)
        if 'read_items' in session:
            read_items = session.get('read_items', None)
            read_items.append(str(original_id))
            session['read_items'] = read_items
        else:
            session['read_items'] = [str(original_id)]

        if "read_items" in session:
            read_items = session.get('read_items', None)
            session['read_items'] = read_items
            all = update_anno(read_items)

        else:
            all = update_anno()
        count = session.get('count', None)

        if int(count) != 0:
            session['count'] = count - 1

        return jsonify({"count": session['count'], "all": all})


@app.route('/validation', methods=['GET'])
def display_validation():
    """function to return the HTML page to display the sentence"""
    session['validation_count'] = 0

    if "read_validations" in session:
        read_items = session.get('read_validations', None)
        session['read_validations'] = read_items
        all = update_valid(read_items)

    else:
        all = update_valid()

    shuffle(all)

    return render_template('validation.html', all=all, count=session['validation_count'])


@app.route('/get_validations', methods=['GET'])
def get_validations():
    _direction = request.args.get('direction')
    val_count = session.get('validation_count', None)
    session['validation_count'] = val_count
    session['validation_count'] = session['validation_count'] + (1 if _direction == 'f' else - 1)

    if "read_validations" in session:
        read_items = session.get('read_validations', None)
        session['read_validations'] = read_items
        all = update_valid(read_items)

    else:
        all = update_valid()

    return jsonify(
        {'forward': str(session['validation_count'] + 1 < len(session['all_validations'])),
         'back': str(bool(session['validation_count'])), "count": session['validation_count'],
         'all': all})


@app.route('/store_validation_in_mongo', methods=['POST'])
def store_validation_in_mongo():
    if request.method == 'POST':
        original_id = request.form['original_id']
        best = request.form['best_pick']
        store_valid_in_mongo(best, original_id)
        if 'read_validations' in session:
            read_items = session.get('read_validations', None)
            read_items.append(str(original_id))
            session['read_validations'] = read_items
        else:
            session['read_validations'] = [str(original_id)]

        if "read_validations" in session:
            read_items = session.get('read_validations', None)
            session['read_validations'] = read_items
            all = update_valid(read_items)

        else:
            all = update_valid()

        count = session.get('validation_count', None)
        if int(count) != 0:
            session['validation_count'] = count - 1

        return jsonify({'count': session['validation_count'],
                        'all_validations': all,
                        'data': render_template('response.html', count=session['validation_count'],
                                                all_validations=all)})


"""Char NL"""

@app.route('/predict_CHAR_nl-gro', methods=['POST'])
def predict_predict_nl_gro():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        translation_query = translation_query.strip()

        if translation_query[-1] not in string.punctuation:
            translation_query = translation_query + "."
            punctuation_alert = True
        else:
            punctuation_alert = False
        # Preprocess
        processed = process(translation_query)
        # Tokenize to Char
        char_encoding = convert_char(processed)
        # Translate
        with graph.as_default():
            output = get_predictions(char_encoding, char_args, char_params, char_models, char_dataset)
        # Detokenize and restore
        output_sen = restore(output)
        if punctuation_alert:
            output_sen = output_sen[0:-1]
        return jsonify({'translation': output_sen})
    else:
        return abort(404)


"""Char gro nl"""


@app.route('/predict_CHAR_gro-nl', methods=['POST'])
def predict_gro_nl():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        translation_query = translation_query.strip()

        if translation_query[-1] not in string.punctuation:
            translation_query = translation_query + "."
            punctuation_alert = True
        else:
            punctuation_alert = False
        # Preprocess
        processed = process(translation_query)
        # Tokenize to Char
        char_encoding = convert_char(processed)
        # Translate
        with graph.as_default():
            output = get_predictions(char_encoding, char_args_nl, char_params_nl, char_models_nl, char_dataset_nl)
        # Detokenize and restore
        output_sen = restore(output)
        if punctuation_alert:
            output_sen = output_sen[0:-1]

        return jsonify({'translation': output_sen})

    else:
        return abort(404)


"""BPE NL GRO"""


@app.route('/predict_BPE_nl-gro', methods=['POST'])
def predict_nl_gro_bpe():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        translation_query = translation_query.strip()

        if translation_query[-1] not in string.punctuation:
            translation_query = translation_query + "."
            punctuation_alert = True
        else:
            punctuation_alert = False
        # Preprocess
        encoded_text = bpe_encode(translation_query)

        # Translate
        with graph.as_default():
            output = get_predictions(encoded_text, bpe_args, bpe_params, bpe_models, bpe_dataset)
        # Detokenize and restore
        output_string = "".join(output)

        decode_string = subprocess.run(['bash', 'restore.sh', output_string],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                                       check=True,
                                       text=True)
        output_sen = decode_string.stdout
        if punctuation_alert:
            output_sen = output_sen[0:-1]

        return jsonify({'translation': output_sen})

    else:
        return abort(404)


"""BPE GRO NL"""


@app.route('/predict_BPE_gro-nl', methods=['POST'])
def predict_gro_nl_bpe():
    # Validation
    if request.method == 'POST':
        translation_query = request.form['translation']
        translation_query = translation_query.strip()

        if translation_query[-1] not in string.punctuation:
            translation_query = translation_query + "."
            punctuation_alert = True
        else:
            punctuation_alert = False
        # Preprocess
        encoded_text = bpe_encode_nl(translation_query)

        with graph.as_default():

            output = get_predictions(encoded_text, bpe_args_nl, bpe_params_nl, bpe_models_nl, bpe_dataset_nl)
        # Detokenize and restore
        output_string = "".join(output)

        decode_string = subprocess.run(['bash', 'restore.sh', output_string],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                                       check=True,
                                       text=True)
        output_sen = decode_string.stdout
        if punctuation_alert:
            output_sen = output_sen[0:-1]

        return jsonify({'translation': output_sen})

    else:
        return abort(404)


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..." \
           "please wait until server has fully started"))
    # init()




    app.run(debug=True, use_reloader=False)
