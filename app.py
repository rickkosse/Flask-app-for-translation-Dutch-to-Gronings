#!/usr/bin/env python3
from __future__ import print_function

import sys
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/")
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/nmt_keras")
sys.path.append("/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/nmt_keras/src/keras-wrapper/keras-wrapper")

from flask import Flask,render_template,url_for,request
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
    global models, dataset, args, params,graph
    # load the pre-trained Keras model
    models, dataset, args, params = pred_try.load_all()
    graph = tf.get_default_graph()

    return models, dataset, args, params
def get_predictions():
    # result= subprocess.Popen("/Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras/sample_ensemble.py -m /Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras/trained_models/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_16 -ds /Users/rickkosse/Documents/RUG/flask_translation_env/nmt_keras/datasets/Dataset_EuTrans_nlgro.pkl  --text /Users/rickkosse/Documents/RUG/flask_translation_env/Output.txt", shell=True, stdout=subprocess.PIPE)
    pred= pred_try.predicted(models, dataset, args, params)
    return pred

def write_to_file(comments):
    with open("Output.txt", "w") as text_file:
        text_file.write(comments)

@app.route('/')
def home():
    return render_template('home.html')

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
        with graph.as_default():
            output = get_predictions()
        output_sen = restore(output)
   

    return render_template('result.html',prediction = output_sen)



if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(debug=True)