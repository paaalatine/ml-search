import re
import sys
import warnings

import numpy as np
import pandas as pd

from time import time
from math import exp

from joblib import dump, load

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class MLSearch():

	DIR = "/home/sonya/mlsearch/"
	INPUT_FILE = "product-mappings-dataset-full-part-100000.csv"
	ROWS_COUNT = 1000


	def __init__(self):
		self.vectorizer = CountVectorizer(token_pattern=r'(?u)\b[\w\w\d]+\b')
		self.tfidf = TfidfTransformer(norm='l2')

		self.clf = LinearSVC() # yes
		# self.clf = SGDClassifier() # yes
		# self.clf = RidgeClassifier() # ?
		# self.clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0) # ?


	def save(self):
		print("save model")
		dump(self.clf, self.DIR + 'saved/clf')
		dump(self.vectorizer, self.DIR + 'saved/vectorizer')
		dump(self.tfidf, self.DIR + 'saved/tfidf')


	def load(self):
		print("load saved model")
		self.clf = load(self.DIR + 'saved/clf')
		self.vectorizer = load(self.DIR + 'saved/vectorizer')
		self.tfidf = load(self.DIR + 'saved/tfidf')


	def evaluate(self, test_docs, my_products):
		test_vectors = self.transform(self.vectorizer, test_docs)
		test_vectors = self.transform(self.tfidf, test_vectors)

		predicted = self.clf.predict(test_vectors)
		print(np.mean(predicted == my_products))


	def check_all_important_words(self, doc, important_words):
		splited_doc = self.vectorizer.build_tokenizer()(doc.lower())

		i = 0
		j = 0
		k = 0
		
		for important_word in important_words:
			if re.search(r'\b{0}\b'.format(important_word), doc, re.IGNORECASE):
				i += 1
			elif re.search(r'(\b{0}\B|\B{0}\b)'.format(important_word), doc, re.IGNORECASE):
				j += 1
			else: 
				for word in splited_doc:
					if re.search(r'(\B{0}\b|\b{0}\B)'.format(word), important_word, re.IGNORECASE):
						k += 1

		# print(i, j, k)

		if i + j / 2 + k / 2 != len(important_words) or j % 2 != 0 or k % 2 != 0:
			return False

		return True


	def predict(self, input):	
		#input = input.replace('-', '')

		test_vectors = self.transform(self.vectorizer, [input])
		test_vectors = self.transform(self.tfidf, test_vectors)

		t0 = time()

		predicted = self.clf.predict(test_vectors)[0]
		# probas = clf.predict_proba(test_transformed_vectors)
		probas = self.clf.decision_function(test_vectors)[0]

		print("\nprediction done in %fs" % (time() - t0))
		
		proba = probas[np.where(self.clf.classes_ == predicted)]
		proba = 1 / (1 + exp(-proba)) # for decision_function

		input_important_words, output_important_words = self.make_input_important_words(input, predicted)

		input_have_all = self.check_all_important_words(input, output_important_words)
		output_have_all = self.check_all_important_words(predicted, input_important_words)

		if not input_have_all or not output_have_all:
			predicted = 'not found'
			proba = 0.0

		return predicted, proba


	def make_stopwords_list(self, vectors):
		stopwords = []
		words_sum = vectors.sum(axis=0)
		total = words_sum.sum()
		words_counts = [(word, words_sum[0, idx]) for word, idx in self.vectorizer.vocabulary_.items()]
		words_counts = sorted(words_counts, key = lambda x: x[1], reverse=True)
		for key, value in words_counts:
			if(value / total >= 0.03):
				stopwords.append(key)
		return stopwords


	def read_dataset(self):
		df = pd.read_csv(self.DIR + self.INPUT_FILE, header=0, nrows=self.ROWS_COUNT)
		return df["competitor_model"], df["product_model"]


	def fit_transformer(self, transformer, vectors):
		print('\nfit ' + transformer.__class__.__name__)
		t0 = time()
		transformer.fit(vectors)
		print('fitted in %fs' % (time() - t0,))


	def transform(self, transformer, vectors):
		# t0 = time()
		vectors = transformer.transform(vectors)
		# print('transformed in %fs' % (time() - t0,))
		return vectors


	def fit_model(self, X, y):
		print('\nstart fitting model, rows_count = ' + str(self.ROWS_COUNT))
		t0 = time()
		self.clf.fit(X, y)
		print("done in %fs" % (time() - t0))


	def train(self, X, y):

		# FIT & TRANSFORM TO VECTORS

		self.fit_transformer(self.vectorizer, X)
		vectors = self.transform(self.vectorizer, X)

		# stopwords = self.make_stopwords_list(vectors)

		# FIT & TRANSFORM VECTORS WITH FREQ

		self.fit_transformer(self.tfidf, vectors)
		vectors = self.transform(self.tfidf, vectors)

		# TRAINING

		self.fit_model(vectors, y)


	def make_input_important_words(self, input, output):
		coef = self.clf.coef_[np.where(self.clf.classes_ == output)][0].ravel()
		vocabulary_weights = dict(zip(self.vectorizer.get_feature_names(), coef))

		# print(vocabulary_weights)

		input_word_weights = {}
		output_word_weights = {}
		
		for word in self.vectorizer.build_tokenizer()(input.lower()):
			if word in vocabulary_weights:
				input_word_weights.update({word: vocabulary_weights[word]})
			else:
				input_word_weights.update({word: 2});

		for word in self.vectorizer.build_tokenizer()(output.lower()):
			if word in vocabulary_weights:
				output_word_weights.update({word: vocabulary_weights[word]});

		input_word_weights = sorted(input_word_weights.items(), key=lambda kv: kv[1])
		output_word_weights = sorted(output_word_weights.items(), key=lambda kv: kv[1])
		# print(input_word_weights)
		# print(output_word_weights)
		input_important_words = [i[0] for i in input_word_weights if i[1] > 1.5][-3:]
		output_important_words = [i[0] for i in output_word_weights if i[1] > 0.8 and re.search(r'\d', i[0])][-3:]
		# print(input_important_words)
		# print(output_important_words)

		return input_important_words, output_important_words


if __name__ == "__main__":
	np.set_printoptions(threshold=np.nan) # show full numpy array
	warnings.simplefilter(action='ignore', category=FutureWarning) # ignore FutureWarning

	# rows_count = int(sys.argv[1])
	
	search = MLSearch()

	competitor_products, my_products = search.read_dataset();
	
	search.train(competitor_products, my_products)

	inputs = ['Arctic Cooling Freezer i11',
	'Arctic Cooling Freezer i1',
	'Arctic Cooling Freezer',
	'Циркуляционный насос Насосы плюс оборудование BPS 25-6-130',
	'Циркуляционный насос Насосы плюс оборудование BS 25-6-130',
	'Пенал мягкий, 3 отделения, Goal',
	'Пенал школьный мягкий Yes Paul Frank 20*8*3см 531116',
	'Пенал 1 Вересня Paul Frank (530734)',
	'Весы электронные BabyOno 291',
	'Весы электронные BabyOno 293',
	'Весы электронные BabyOno',
	'Весы BabyOno 293',
	'электронные BabyOno 293',
	'Весы электронные 293',
	'электронные 293',
	'Весы 293',
	'Call of Duty: Black Ops 2 Uprising (DLC)',
	'Doom 4 [PC-Jewel]',
	'FIFA 16 (PS4) Русская Версия',
	'Mafia 3',
	'Телевизор Sharp LC-49CFF6002E',
	'Встраиваемая посудомоечная машина BEKO DIS15010',
	'Встраиваемая посудомоечная машина BEKO DIS 1501',
	'Румяна Ninelle Artist компактные № 122',
	'Ninelle Artist Blusher',
	'Кабель USB 2.0 AM to Mini 5P 0.1m Manhattan (328739)']

	data = []

	for input in inputs:
		output, proba = search.predict(input)
		print('{0} => {1} \nwith proba = {2}'.format(input, output, proba))
		data.append((input, output, proba))

	pd.DataFrame(data, columns = ['input', 'output', 'proba']).to_csv('results.csv', ';')

	# search.evaluate(competitor_products, my_products)
