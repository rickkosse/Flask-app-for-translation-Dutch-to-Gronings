import nltk
import re
import nltk.data


def restore(sent_list):

	words = [w.replace(' ', '') for w in sent_list]
	words_out = ["".join(w.replace('***', ' ')) for w in words]
	return ''.join(words_out)

def qoute_detector(sentence):
	keywords = ["!","?","."]
	words = sentence.split()
	pattern = re.compile("[!.?]+$")
	all_punt = [item for item in sentence if pattern.match(item)]
	indices = [i for i, x in enumerate(words) if x =="." or x=="!" or x =="?"]

def punkt_trainer(text):
	train_text = nltk.data.load('tokenizers/punkt/dutch.pickle')
	custom_sent_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
	custom_sent_tokenizer.train(train_text)
	tokenized = custom_sent_tokenizer.tokenize(text)

	return '\n'.join(tokenized)


def sent_detector(text):
	sent_detector = nltk.data.load('tokenizers/punkt/dutch.pickle')
	return('\n'.join(sent_detector.tokenize(text.strip())))

def remove_unnecessary_punct_in_sentences(sentence):
	"""Function that removes unnecessary punctuation """
	punctuation = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
	translator = str.maketrans('','', punctuation)
	sentence= sentence.translate(translator)
	return sentence


def remove_unnecessary_punct(words):
	"""Function that removes punctuation but keeps the paragraph characters"""
	punctuation = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
	translator = str.maketrans('','', punctuation)
	word= words.translate(translator)
	if re.match(r'^p$', word):
		word ="<p>"
	return word

def normalize_punctuation(sentence):
	"""Normalizes the punctuation"""
	text = sentence.replace('„', '\"')
	text = text.replace('“', '\"')
	text = text.replace('``', '\"')
	text = text.replace('”', '\"')
	text = text.replace('’', '\'')
	text = text.replace('–', '-')
	text = text.replace(' +', ' ')
	text = text.replace('‘', '\'')
	text = text.replace('‚', '\'')
	text = text.replace('«', '\"')
	text = text.replace('»', '\"')
	text = text.replace('»', '\"')
	text = text.replace('  ', ' ')
	# text = text.replace('...', '')
	# text = text.replace('....', '')
	# text = text.replace('.....', '')
	# text = text.replace('......', '')
	text = text.replace('`', '\'')
	text = text.replace('``', '\"')
	return text


def replace_tokenized_punct(sentence):
	"""Punctuation added by NLTK is removed"""
	text = sentence.replace('`', '')
	text = text.replace('``', '')
	text = text.replace('´´', '')
	text = text.replace('´', '')
	return text

def normalize_gronings(sentence):
	"""keep pronoun in Gronings"""

	text = sentence.replace('d \'', 'd\'')
	return text

def NLTK_tokenize_punt(sentence):
	tokenized = nltk.word_tokenize(sentence)
	line = " ".join(tokenized)
	return line


def tokenize(sentence):
	"""NLTK tokenizer and punctuation added by NLTK is removed"""

	# tokenized = nltk.word_tokenize(sentence)
	# line = " ".join(tokenized)
	text = sentence.replace('< p >', '<p>')
	text = text.replace('``', '\"')
	text = text.replace('`', '\"')
	text = text.replace('\'\'', '\"')
	text = text.replace('\'', '\'')
	text = text.replace('´´', '\"')
	text = text.replace('´', '\"')
	return text

def process(alignment_nl):
	final_list = []
	"""Dutch"""
	for _ in alignment_nl.split("\n"):
		text8=_.lstrip(' ')
		sen = normalize_punctuation(text8)
		tokenized = tokenize(sen)
		# # final_sent = qoute_detector(text8)
		for sent in nltk.tokenize.sent_tokenize(tokenized, "dutch"):
			los_puntuation = NLTK_tokenize_punt(sent)
			tokenized = tokenize(los_puntuation)

	return tokenized.lower()


def convert_char(sentences):


	replace_whitespace = sentences.replace(' ', '_')
	chars = " ".join(replace_whitespace)
	chars_based = chars.replace('< p >\n', '')
	nl_words = ' '.join(chars_based.split())
	replace_star_nl = nl_words.replace('_', '***')

	return replace_star_nl


def main():
	written_files = process(poems_list_nl)
	char= convert_char(written_files)





if __name__ == "__main__":
	main()