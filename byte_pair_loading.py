import tensorflow as tf
import codecs
from subword_nmt.apply_bpe import BPE

codes = codecs.open("{codes_file}", encoding='utf-8')
# output = codecs.open(args.output.name, 'w', encoding='utf-8')
vocabulary = codecs.open("./100/{vocab_file}.L1", encoding='utf-8')
bpe = BPE(codes, 100, '@@', vocabulary)

# output = codecs.open(args.output.name, 'w', encoding='utf-8')
vocabulary_nl = codecs.open("./100/{vocab_file}.L2", encoding='utf-8')
bpe_nl = BPE(codes, 100, '@@', vocabulary_nl)

graph = tf.compat.v1.get_default_graph()
