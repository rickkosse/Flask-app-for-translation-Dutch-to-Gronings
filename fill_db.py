from mongo_db import *
import os

path = os.path.dirname("/Users/KosseR/Documents/translation_data/")
with open(path+"/mono_kinder_biebel.txt", "r") as input_file:
    lines = input_file.readlines()

for sentence in lines:
    sen_without_trailing_newline= sentence.strip("\n")
    sen_without_trailing_space = sen_without_trailing_newline.strip()
    insert_to_db(sen_without_trailing_space)