#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/")
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/nmt_keras")
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/nmt_keras/src/keras-wrapper/keras-wrapper")

from flask import Flask,render_template,url_for,request, jsonify
from flask_bootstrap import Bootstrap
from proces_and_convert_to_char import process, convert_char, restore
import tensorflow as tf
import sys
import importlib.util
import subprocess
from nmtkeras import sample_ensemble
from nmtkeras import pred_try

import os 

# initialize our Flask application and the Keras model
app = Flask(__name__)
Bootstrap(app)

def init():
    global models, dataset, args, params,graph, current_status
    # load the pre-trained Keras model
    models, dataset, args, params = pred_try.load_all()
    graph = tf.get_default_graph()
    current_status = ""

    return models, dataset, args, params

def get_predictions():
    pred= pred_try.predicted(models, dataset, args, params)
    return pred

def write_to_file(comments):
    with open("Output.txt", "w") as text_file:
        text_file.write(comments)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/get_toggled_status') 
def toggled_status():
    global current_status
    current_status = request.args.get('status')

    print("deze loopt achter:",current_status)

    return 'nl_gro' if current_status == 'gro_nl' else 'gro_nl'

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        namequery = request.form['name']
        processed = process(namequery)
        char_encoding= convert_char(processed)
        create_file = write_to_file(char_encoding)
        print("the current_status=", current_status)
        with graph.as_default():
            output = get_predictions()
        output_sen = restore(output)

        return jsonify({'name' : output})


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(debug=True)