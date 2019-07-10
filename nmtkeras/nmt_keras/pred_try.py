

# -*- coding: utf-8 -*-
from __future__ import print_function
try:
    import itertools.imap as map
except ImportError:
    pass
import logging
from keras_wrapper.extra.read_write import list2file, nbest2file, list2stdout, numpy2file, pkl2dict
import sys

def load_all():
    args= {"changes":'', "config":None, "dataset":'/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/datasets/Dataset_EuTrans_nlgro.pkl', "dest":"./translated.txt", "glossary":None, "models":['/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/trained_models/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_16'], "n_best":False, 'splits':['val'], 'text':'/Users/rickkosse/Documents/RUG/flask_translation_env/Output.txt', 'verbose':0, "weights":[]}
    params= {'FORCE_RELOAD_VOCABULARY': False, 'TRAIN_ON_TRAINVAL': False, 'MODE': 'training', 'REBUILD_DATASET': True, 'RELOAD_EPOCH': True, 'RELOAD': 0, 'VERBOSE': 1, 'PLOT_EVALUATION': False, 'SAMPLING_SAVE_MODE': 'list', 'WORD_EMBEDDINGS_LABELS': ['source_text', 'target_text'], 'LABEL_WORD_EMBEDDINGS_WITH_VOCAB': True, 'EMBEDDINGS_METADATA': None, 'EMBEDDINGS_LAYER_NAMES': ['source_word_embedding', 'target_word_embedding'], 'EMBEDDINGS_FREQ': 1, 'LOG_DIR': 'tensorboard_logs', 'TENSORBOARD': True, 'DATASET_STORE_PATH': 'datasets/', 'STORE_PATH': 'trained_models/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/', 'MODEL_NAME': 'EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001', 'EXTRA_NAME': '', 'DOUBLE_STOCHASTIC_ATTENTION_REG': 0.0, 'USE_L2': False, 'USE_L1': False, 'USE_PRELU': False, 'BATCH_NORMALIZATION_MODE': 1, 'USE_BATCH_NORMALIZATION': True, 'NOISE_AMOUNT': 0.01, 'USE_NOISE': False, 'ATTENTION_DROPOUT_P': 0.0, 'RECURRENT_DROPOUT_P': 0.0, 'RECURRENT_INPUT_DROPOUT_P': 0.0, 'DROPOUT_P': 0.0, 'RECURRENT_WEIGHT_DECAY': 0.0, 'WEIGHT_DECAY': 0.0001, 'REGULARIZATION_FN': 'L2', 'N_HEADS': 8, 'FF_SIZE': 128, 'MULTIHEAD_ATTENTION_ACTIVATION': 'linear', 'MODEL_SIZE': 32, 'SKIP_VECTORS_SHARED_ACTIVATION': 'tanh', 'ADDITIONAL_OUTPUT_MERGE_MODE': 'Add', 'SKIP_VECTORS_HIDDEN_SIZE': 32, 'ATTENTION_SIZE': 32, 'DECODER_HIDDEN_SIZE': 32, 'INIT_LAYERS': ['tanh'], 'BIDIRECTIONAL_MERGE_MODE': 'concat', 'BIDIRECTIONAL_DEEP_ENCODER': True, 'BIDIRECTIONAL_ENCODER': True, 'ENCODER_HIDDEN_SIZE': 32, 'ATTENTION_MODE': 'add', 'DECODER_RNN_TYPE': 'ConditionalLSTM', 'USE_CUDNN': False, 'ENCODER_RNN_TYPE': 'LSTM', 'DEEP_OUTPUT_LAYERS': [('linear', 32)], 'N_LAYERS_DECODER': 1, 'N_LAYERS_ENCODER': 1, 'TIE_EMBEDDINGS': False, 'SCALE_TARGET_WORD_EMBEDDINGS': False, 'SCALE_SOURCE_WORD_EMBEDDINGS': False, 'TRG_PRETRAINED_VECTORS_TRAINABLE': True, 'TRG_PRETRAINED_VECTORS': None, 'TARGET_TEXT_EMBEDDING_SIZE': 32, 'SRC_PRETRAINED_VECTORS_TRAINABLE': True, 'SRC_PRETRAINED_VECTORS': None, 'SOURCE_TEXT_EMBEDDING_SIZE': 32, 'INIT_ATT': 'glorot_uniform', 'INNER_INIT': 'orthogonal', 'INIT_FUNCTION': 'glorot_uniform', 'TRAINABLE_DECODER': True, 'TRAINABLE_ENCODER': True, 'MODEL_TYPE': 'AttentionRNNEncoderDecoder', 'STOP_METRIC': 'Bleu_4', 'PATIENCE': 10, 'EARLY_STOP': True, 'SAVE_EACH_EVALUATION': True, 'WRITE_VALID_SAMPLES': True, 'EPOCHS_FOR_SAVE': 1, 'PARALLEL_LOADERS': 1, 'JOINT_BATCHES': 4, 'HOMOGENEOUS_BATCHES': False, 'N_GPUS': 1, 'BATCH_SIZE': 50, 'MAX_EPOCH': 500, 'MIN_LR': 1e-09, 'WARMUP_EXP': -1.5, 'LR_HALF_LIFE': 100, 'LR_REDUCER_EXP_BASE': -0.5, 'LR_REDUCER_TYPE': 'exponential', 'LR_START_REDUCTION_ON_EPOCH': 0, 'LR_REDUCE_EACH_EPOCHS': False, 'LR_GAMMA': 0.8, 'LR_DECAY': None, 'ACCUMULATE_GRADIENTS': 1, 'EPSILON': 1e-08, 'AMSGRAD': False, 'BETA_2': 0.999, 'BETA_1': 0.9, 'RHO': 0.9, 'NESTEROV_MOMENTUM': False, 'MOMENTUM': 0.0, 'USE_TF_OPTIMIZER': True, 'CLIP_V': 0.0, 'CLIP_C': 5.0, 'LR': 0.001, 'OPTIMIZER': 'Adam', 'LABEL_SMOOTHING': 0.0, 'SAMPLE_WEIGHTS': True, 'CLASSIFIER_ACTIVATION': 'softmax', 'LOSS': 'categorical_crossentropy', 'MAX_OUTPUT_TEXT_LEN_TEST': 150, 'MAX_OUTPUT_TEXT_LEN': 50, 'MIN_OCCURRENCES_OUTPUT_VOCAB': 0, 'OUTPUT_VOCABULARY_SIZE': 0, 'MAX_INPUT_TEXT_LEN': 50, 'MIN_OCCURRENCES_INPUT_VOCAB': 0, 'INPUT_VOCABULARY_SIZE': 0, 'PAD_ON_BATCH': True, 'FILL': 'end', 'DATA_AUGMENTATION': False, 'TOKENIZE_REFERENCES': True, 'TOKENIZE_HYPOTHESES': True, 'APPLY_DETOKENIZATION': False, 'DETOKENIZATION_METHOD': 'detokenize_none', 'BPE_CODES_PATH': 'examples/EuTrans//training_codes.joint', 'TOKENIZATION_METHOD': 'tokenize_none', 'MAPPING': 'examples/EuTrans//mapping.nl_gro.pkl', 'ALIGN_FROM_RAW': True, 'HEURISTIC': 0, 'POS_UNK': True, 'SAMPLE_EACH_UPDATES': 300, 'START_SAMPLING_ON_EPOCH': 1, 'N_SAMPLES': 5, 'SAMPLE_ON_SETS': ['train', 'val'], 'ALPHA_FACTOR': 0.6, 'NORMALIZE_SAMPLING': False, 'COVERAGE_NORM_FACTOR': 0.2, 'COVERAGE_PENALTY': False, 'LENGTH_NORM_FACTOR': 0.2, 'LENGTH_PENALTY': False, 'MINLEN_GIVEN_X_FACTOR': 3, 'MINLEN_GIVEN_X': True, 'MAXLEN_GIVEN_X_FACTOR': 2, 'MAXLEN_GIVEN_X': True, 'SEARCH_PRUNING': False, 'OPTIMIZED_SEARCH': True, 'BEAM_SIZE': 6, 'BEAM_SEARCH': True, 'TEMPERATURE': 1, 'SAMPLING': 'max_likelihood', 'EVAL_EACH': 1, 'EVAL_EACH_EPOCHS': True, 'START_EVAL_ON_EPOCH': 1, 'EVAL_ON_SETS_KERAS': [], 'EVAL_ON_SETS': ['val'], 'METRICS': ['coco'], 'OUTPUTS_TYPES_DATASET': ['text-features'], 'INPUTS_TYPES_DATASET': ['text-features', 'text-features'], 'OUTPUTS_IDS_MODEL': ['target_text'], 'INPUTS_IDS_MODEL': ['source_text', 'state_below'], 'OUTPUTS_IDS_DATASET': ['target_text'], 'INPUTS_IDS_DATASET': ['source_text', 'state_below'], 'GLOSSARY': None, 'TEXT_FILES': {'train': 'training.', 'val': 'dev.', 'test': 'test.'}, 'DATA_ROOT_PATH': 'examples/EuTrans/', 'TRG_LAN': 'gro', 'SRC_LAN': 'nl', 'DATASET_NAME': 'EuTrans', 'TASK_NAME': 'EuTrans'}
    # from config import load_parameters
    # params = load_parameters()
    # params = check_params(params)

    """
    Use several translation models for obtaining predictions from a source text file.

    :param argparse.Namespace args: Arguments given to the method:

                      * dataset: Dataset instance with data.
                      * text: Text file with source sentences.
                      * splits: Splits to sample. Should be already included in the dataset object.
                      * dest: Output file to save scores.
                      * weights: Weight given to each model in the ensemble. You should provide the same number of weights than models. By default, it applies the same weight to each model (1/N).
                      * n_best: Write n-best list (n = beam size).
                      * config: Config .pkl for loading the model configuration. If not specified, hyperparameters are read from config.py.
                      * models: Path to the models.
                      * verbose: Be verbose or not.

    :param params: parameters of the translation model.
    """
    from data_engine.prepare_data import update_dataset_from_file
    from keras_wrapper.model_ensemble import BeamSearchEnsemble
    from keras_wrapper.cnn_model import loadModel
    from keras_wrapper.dataset import loadDataset
    from keras_wrapper.utils import decode_predictions_beam_search

    # logging.info("Using an ensemble of %d models" % len(args["models"]))
    models = [loadModel(m, -1, full_path=True) for m in args["models"]]
    dataset = loadDataset(args["dataset"])
    dataset = update_dataset_from_file(dataset, args["text"], params, splits=args["splits"], remove_outputs=True)



    return models, dataset, args, params


def predicted():

    args= {"changes":'', "config":None, "dataset":'/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/datasets/Dataset_EuTrans_nlgro.pkl', "dest":"./translated.txt", "glossary":None, "models":['/Users/rickkosse/Documents/RUG/flask_translation_env/nmtkeras/trained_models/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/epoch_16'], "n_best":False, 'splits':['val'], 'text':'/Users/rickkosse/Documents/RUG/flask_translation_env/Output.txt', 'verbose':0, "weights":[]}
    params= {'FORCE_RELOAD_VOCABULARY': False, 'TRAIN_ON_TRAINVAL': False, 'MODE': 'training', 'REBUILD_DATASET': True, 'RELOAD_EPOCH': True, 'RELOAD': 0, 'VERBOSE': 1, 'PLOT_EVALUATION': False, 'SAMPLING_SAVE_MODE': 'list', 'WORD_EMBEDDINGS_LABELS': ['source_text', 'target_text'], 'LABEL_WORD_EMBEDDINGS_WITH_VOCAB': True, 'EMBEDDINGS_METADATA': None, 'EMBEDDINGS_LAYER_NAMES': ['source_word_embedding', 'target_word_embedding'], 'EMBEDDINGS_FREQ': 1, 'LOG_DIR': 'tensorboard_logs', 'TENSORBOARD': True, 'DATASET_STORE_PATH': 'datasets/', 'STORE_PATH': 'trained_models/EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001/', 'MODEL_NAME': 'EuTrans_nlgro_AttentionRNNEncoderDecoder_src_emb_32_bidir_True_enc_LSTM_32_dec_ConditionalLSTM_32_deepout_linear_trg_emb_32_Adam_0.001', 'EXTRA_NAME': '', 'DOUBLE_STOCHASTIC_ATTENTION_REG': 0.0, 'USE_L2': False, 'USE_L1': False, 'USE_PRELU': False, 'BATCH_NORMALIZATION_MODE': 1, 'USE_BATCH_NORMALIZATION': True, 'NOISE_AMOUNT': 0.01, 'USE_NOISE': False, 'ATTENTION_DROPOUT_P': 0.0, 'RECURRENT_DROPOUT_P': 0.0, 'RECURRENT_INPUT_DROPOUT_P': 0.0, 'DROPOUT_P': 0.0, 'RECURRENT_WEIGHT_DECAY': 0.0, 'WEIGHT_DECAY': 0.0001, 'REGULARIZATION_FN': 'L2', 'N_HEADS': 8, 'FF_SIZE': 128, 'MULTIHEAD_ATTENTION_ACTIVATION': 'linear', 'MODEL_SIZE': 32, 'SKIP_VECTORS_SHARED_ACTIVATION': 'tanh', 'ADDITIONAL_OUTPUT_MERGE_MODE': 'Add', 'SKIP_VECTORS_HIDDEN_SIZE': 32, 'ATTENTION_SIZE': 32, 'DECODER_HIDDEN_SIZE': 32, 'INIT_LAYERS': ['tanh'], 'BIDIRECTIONAL_MERGE_MODE': 'concat', 'BIDIRECTIONAL_DEEP_ENCODER': True, 'BIDIRECTIONAL_ENCODER': True, 'ENCODER_HIDDEN_SIZE': 32, 'ATTENTION_MODE': 'add', 'DECODER_RNN_TYPE': 'ConditionalLSTM', 'USE_CUDNN': False, 'ENCODER_RNN_TYPE': 'LSTM', 'DEEP_OUTPUT_LAYERS': [('linear', 32)], 'N_LAYERS_DECODER': 1, 'N_LAYERS_ENCODER': 1, 'TIE_EMBEDDINGS': False, 'SCALE_TARGET_WORD_EMBEDDINGS': False, 'SCALE_SOURCE_WORD_EMBEDDINGS': False, 'TRG_PRETRAINED_VECTORS_TRAINABLE': True, 'TRG_PRETRAINED_VECTORS': None, 'TARGET_TEXT_EMBEDDING_SIZE': 32, 'SRC_PRETRAINED_VECTORS_TRAINABLE': True, 'SRC_PRETRAINED_VECTORS': None, 'SOURCE_TEXT_EMBEDDING_SIZE': 32, 'INIT_ATT': 'glorot_uniform', 'INNER_INIT': 'orthogonal', 'INIT_FUNCTION': 'glorot_uniform', 'TRAINABLE_DECODER': True, 'TRAINABLE_ENCODER': True, 'MODEL_TYPE': 'AttentionRNNEncoderDecoder', 'STOP_METRIC': 'Bleu_4', 'PATIENCE': 10, 'EARLY_STOP': True, 'SAVE_EACH_EVALUATION': True, 'WRITE_VALID_SAMPLES': True, 'EPOCHS_FOR_SAVE': 1, 'PARALLEL_LOADERS': 1, 'JOINT_BATCHES': 4, 'HOMOGENEOUS_BATCHES': False, 'N_GPUS': 1, 'BATCH_SIZE': 50, 'MAX_EPOCH': 500, 'MIN_LR': 1e-09, 'WARMUP_EXP': -1.5, 'LR_HALF_LIFE': 100, 'LR_REDUCER_EXP_BASE': -0.5, 'LR_REDUCER_TYPE': 'exponential', 'LR_START_REDUCTION_ON_EPOCH': 0, 'LR_REDUCE_EACH_EPOCHS': False, 'LR_GAMMA': 0.8, 'LR_DECAY': None, 'ACCUMULATE_GRADIENTS': 1, 'EPSILON': 1e-08, 'AMSGRAD': False, 'BETA_2': 0.999, 'BETA_1': 0.9, 'RHO': 0.9, 'NESTEROV_MOMENTUM': False, 'MOMENTUM': 0.0, 'USE_TF_OPTIMIZER': True, 'CLIP_V': 0.0, 'CLIP_C': 5.0, 'LR': 0.001, 'OPTIMIZER': 'Adam', 'LABEL_SMOOTHING': 0.0, 'SAMPLE_WEIGHTS': True, 'CLASSIFIER_ACTIVATION': 'softmax', 'LOSS': 'categorical_crossentropy', 'MAX_OUTPUT_TEXT_LEN_TEST': 150, 'MAX_OUTPUT_TEXT_LEN': 50, 'MIN_OCCURRENCES_OUTPUT_VOCAB': 0, 'OUTPUT_VOCABULARY_SIZE': 0, 'MAX_INPUT_TEXT_LEN': 50, 'MIN_OCCURRENCES_INPUT_VOCAB': 0, 'INPUT_VOCABULARY_SIZE': 0, 'PAD_ON_BATCH': True, 'FILL': 'end', 'DATA_AUGMENTATION': False, 'TOKENIZE_REFERENCES': True, 'TOKENIZE_HYPOTHESES': True, 'APPLY_DETOKENIZATION': False, 'DETOKENIZATION_METHOD': 'detokenize_none', 'BPE_CODES_PATH': 'examples/EuTrans//training_codes.joint', 'TOKENIZATION_METHOD': 'tokenize_none', 'MAPPING': 'examples/EuTrans//mapping.nl_gro.pkl', 'ALIGN_FROM_RAW': True, 'HEURISTIC': 0, 'POS_UNK': True, 'SAMPLE_EACH_UPDATES': 300, 'START_SAMPLING_ON_EPOCH': 1, 'N_SAMPLES': 5, 'SAMPLE_ON_SETS': ['train', 'val'], 'ALPHA_FACTOR': 0.6, 'NORMALIZE_SAMPLING': False, 'COVERAGE_NORM_FACTOR': 0.2, 'COVERAGE_PENALTY': False, 'LENGTH_NORM_FACTOR': 0.2, 'LENGTH_PENALTY': False, 'MINLEN_GIVEN_X_FACTOR': 3, 'MINLEN_GIVEN_X': True, 'MAXLEN_GIVEN_X_FACTOR': 2, 'MAXLEN_GIVEN_X': True, 'SEARCH_PRUNING': False, 'OPTIMIZED_SEARCH': True, 'BEAM_SIZE': 6, 'BEAM_SEARCH': True, 'TEMPERATURE': 1, 'SAMPLING': 'max_likelihood', 'EVAL_EACH': 1, 'EVAL_EACH_EPOCHS': True, 'START_EVAL_ON_EPOCH': 1, 'EVAL_ON_SETS_KERAS': [], 'EVAL_ON_SETS': ['val'], 'METRICS': ['coco'], 'OUTPUTS_TYPES_DATASET': ['text-features'], 'INPUTS_TYPES_DATASET': ['text-features', 'text-features'], 'OUTPUTS_IDS_MODEL': ['target_text'], 'INPUTS_IDS_MODEL': ['source_text', 'state_below'], 'OUTPUTS_IDS_DATASET': ['target_text'], 'INPUTS_IDS_DATASET': ['source_text', 'state_below'], 'GLOSSARY': None, 'TEXT_FILES': {'train': 'training.', 'val': 'dev.', 'test': 'test.'}, 'DATA_ROOT_PATH': 'examples/EuTrans/', 'TRG_LAN': 'gro', 'SRC_LAN': 'nl', 'DATASET_NAME': 'EuTrans', 'TASK_NAME': 'EuTrans'}

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    # For converting predictions into sentences
    index2word_y = dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words']

    if params.get('APPLY_DETOKENIZATION', False):
        detokenize_function = eval('dataset.' + params['DETOKENIZATION_METHOD'])

    params_prediction = dict()
    params_prediction['max_batch_size'] = params.get('BATCH_SIZE', 20)
    params_prediction['n_parallel_loaders'] = params.get('PARALLEL_LOADERS', 1)
    params_prediction['beam_size'] = params.get('BEAM_SIZE', 6)
    params_prediction['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 100)
    params_prediction['optimized_search'] = params['OPTIMIZED_SEARCH']
    params_prediction['model_inputs'] = params['INPUTS_IDS_MODEL']
    params_prediction['model_outputs'] = params['OUTPUTS_IDS_MODEL']
    params_prediction['dataset_inputs'] = params['INPUTS_IDS_DATASET']
    params_prediction['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
    params_prediction['search_pruning'] = params.get('SEARCH_PRUNING', False)
    params_prediction['normalize_probs'] = params.get('NORMALIZE_SAMPLING', False)
    params_prediction['alpha_factor'] = params.get('ALPHA_FACTOR', 1.0)
    params_prediction['coverage_penalty'] = params.get('COVERAGE_PENALTY', False)
    params_prediction['length_penalty'] = params.get('LENGTH_PENALTY', False)
    params_prediction['length_norm_factor'] = params.get('LENGTH_NORM_FACTOR', 0.0)
    params_prediction['coverage_norm_factor'] = params.get('COVERAGE_NORM_FACTOR', 0.0)
    params_prediction['pos_unk'] = params.get('POS_UNK', False)
    params_prediction['state_below_maxlen'] = -1 if params.get('PAD_ON_BATCH', True) \
        else params.get('MAX_OUTPUT_TEXT_LEN', 50)
    params_prediction['output_max_length_depending_on_x'] = params.get('MAXLEN_GIVEN_X', True)
    params_prediction['output_max_length_depending_on_x_factor'] = params.get('MAXLEN_GIVEN_X_FACTOR', 3)
    params_prediction['output_min_length_depending_on_x'] = params.get('MINLEN_GIVEN_X', True)
    params_prediction['output_min_length_depending_on_x_factor'] = params.get('MINLEN_GIVEN_X_FACTOR', 2)
    params_prediction['attend_on_output'] = params.get('ATTEND_ON_OUTPUT', 'transformer' in params['MODEL_TYPE'].lower())
    params_prediction['glossary'] = params.get('GLOSSARY', None)

    heuristic = params.get('HEURISTIC', 0)
    mapping = None if dataset.mapping == dict() else dataset.mapping
    model_weights = args["weights"]

    if args["glossary"] is not None:
        glossary = pkl2dict(args["glossary"])
    elif params_prediction['glossary'] is not None:
        glossary = pkl2dict(params_prediction['glossary'])
    else:
        glossary = None

    if model_weights is not None and model_weights != []:
        assert len(model_weights) == len(models), 'You should give a weight to each model. You gave %d models and %d weights.' % (len(models), len(model_weights))
        model_weights = map(float, model_weights)
        if len(model_weights) > 1:
            logger.info('Giving the following weights to each model: %s' % str(model_weights))
    for s in args["splits"]:
        # Apply model predictions
        params_prediction['predict_on_sets'] = [s]
        beam_searcher = BeamSearchEnsemble(models, dataset, params_prediction,
                                           model_weights=model_weights, n_best=args["n_best"], verbose=args["verbose"])
        if args["n_best"]:
            predictions, n_best = beam_searcher.predictBeamSearchNet()[s]
        else:
            predictions = beam_searcher.predictBeamSearchNet()[s]
            n_best = None
        if params_prediction['pos_unk']:
            samples = predictions[0]
            alphas = predictions[1]
            sources = [x.strip() for x in open(args["text"], 'r').read().split('\n')]
            sources = sources[:-1] if len(sources[-1]) == 0 else sources
        else:
            samples = predictions
            alphas = None
            heuristic = None
            sources = None

        predictions = decode_predictions_beam_search(samples,
                                                     index2word_y,
                                                     glossary=glossary,
                                                     alphas=alphas,
                                                     x_text=sources,
                                                     heuristic=heuristic,
                                                     mapping=mapping,
                                                     verbose=args["verbose"])
        # Apply detokenization function if needed
        if params.get('APPLY_DETOKENIZATION', False):
            predictions = map(detokenize_function, predictions)

        if args["n_best"]:
            n_best_predictions = []
            for i, (n_best_preds, n_best_scores, n_best_alphas) in enumerate(n_best):
                n_best_sample_score = []
                for n_best_pred, n_best_score, n_best_alpha in zip(n_best_preds, n_best_scores, n_best_alphas):
                    pred = decode_predictions_beam_search([n_best_pred],
                                                          index2word_y,
                                                          glossary=glossary,
                                                          alphas=[n_best_alpha] if params_prediction['pos_unk'] else None,
                                                          x_text=[sources[i]] if params_prediction['pos_unk'] else None,
                                                          heuristic=heuristic,
                                                          mapping=mapping,
                                                          verbose=args["verbose"])
                    # Apply detokenization function if needed
                    if params.get('APPLY_DETOKENIZATION', False):
                        pred = map(detokenize_function, pred)

                    n_best_sample_score.append([i, pred, n_best_score])
                n_best_predictions.append(n_best_sample_score)
        # Store result
        if args["dest"] is not None:
            filepath = args["dest"]  # results file
            if params.get('SAMPLING_SAVE_MODE', 'list'):
                list2file(filepath, predictions)
                if args["n_best"]:
                    nbest2file(filepath + '.nbest', n_best_predictions)
                    return n_best_predictions
            else:
                raise Exception('Only "list" is allowed in "SAMPLING_SAVE_MODE"')
        else:
            list2stdout(predictions)
            if args["n_best"]:
                # logging.info('Storing n-best sentences in ./' + s + '.nbest')
                nbest2file('./' + s + '.nbest', n_best_predictions)
        # logging.info('Sampling finished')

if __name__ == '__main__':
    predicted()

