import re
import sys
import warnings

import numpy as np
import pandas as pd

from time import time
from math import exp

from joblib import dump, load

# from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class MLSearch:

	DIR = "/home/sonya/mlsearch/"
	INPUT_FILE = "product-mappings-dataset-full-part-100000.csv"
	P = .000000

	def __init__(self, rows_count):
		self.rows_count = rows_count
		
		self.vectorizer = CountVectorizer(token_pattern=r'(?u)\b[\w\w\d]+\b')
		self.tfidf = TfidfTransformer(norm='l2')
		self.selector = VarianceThreshold(threshold=self.P * (1 - self.P))

		self.clf = LinearSVC() # yes
		# self.clf = SGDClassifier() # yes
		# self.clf = RidgeClassifier() # ?
		# self.clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0) # ?


	def save(self):
		print("save model")
		dump(self.clf, self.DIR + 'saved/clf')
		dump(self.vectorizer, self.DIR + 'saved/vectorizer')
		dump(self.tfidf, self.DIR + 'saved/tfidf')
		dump(self.selector, self.DIR + 'saved/selector')


	def load(self):
		print("load saved model")
		self.clf = load(self.DIR + 'saved/clf')
		self.vectorizer = load(self.DIR + 'saved/vectorizer')
		self.tfidf = load(self.DIR + 'saved/tfidf')
		self.selector = load(self.DIR + 'saved/selector')


	def evaluate(self, test_docs, my_products):
		test_vectors = self.transform(self.vectorizer, test_docs)
		test_vectors = self.transform(self.tfidf, test_vectors)
		test_vectors = self.transform(self.selector, test_vectors)

		predicted = self.clf.predict(test_vectors)
		print(np.mean(predicted == my_products))


	def predict(self, input):	
		test_vectors = self.transform(self.vectorizer, [input])
		test_vectors = self.transform(self.tfidf, test_vectors)
		test_vectors = self.transform(self.selector, test_vectors)

		t0 = time()

		predicted = self.clf.predict(test_vectors)[0]
		# probas = clf.predict_proba(test_transformed_vectors)
		probas = self.clf.decision_function(test_vectors)[0]

		print("\nprediction done in %fs" % (time() - t0))
		
		proba = probas[np.where(self.clf.classes_ == predicted)]
		proba = 1 / (1 + exp(-proba)) # for decision_function

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
		df = pd.read_csv(self.DIR + self.INPUT_FILE, header=0, nrows=self.rows_count)
		return df["competitor_model"], df["product_model"]


	def fit_transformer(self, transformer, vectors):
		print('\nfit ' + transformer.__class__.__name__)
		t0 = time()
		transformer.fit(vectors)
		print('fitted in %fs' % (time() - t0,))


	def transform(self, transformer, vectors):
		#print('\ntransform by ' + transformer.__class__.__name__)
		t0 = time()
		vectors = transformer.transform(vectors)
		# if(transformer.__class__.__name__ == 'VarianceThreshold'):
		# 	print('\ntransformed in %fs, shape: %s' % (time() - t0, vectors.shape,))
		return vectors


	def fit_model(self, X, y):
		print('\nstart fitting model, rows_count = ' + str(self.rows_count))
		t0 = time()
		self.clf.fit(X, y)
		print("done in %fs" % (time() - t0))


	def train(self, X, y):

		# --------------- FIT & TRANSFORM TO VECTORS ----------------

		self.fit_transformer(self.vectorizer, X)
		vectors = self.transform(self.vectorizer, X)

		# stopwords = self.make_stopwords_list(vectors)

		# ------------ FIT & TRANSFORM VECTORS WITH FREQ ------------

		self.fit_transformer(self.tfidf, vectors)
		vectors = self.transform(self.tfidf, vectors)

		# -------------------- FEATURES SELECTION -------------------

		self.fit_transformer(self.selector, vectors)
		vectors = self.transform(self.selector, vectors)

		# ------------------------ TRAINING -------------------------

		self.fit_model(vectors, y)


if __name__ == "__main__":
	np.set_printoptions(threshold=np.nan) # show full numpy array
	warnings.simplefilter(action='ignore', category=FutureWarning) # ignore FutureWarning

	rows_count = int(sys.argv[1])
	
	search = MLSearch(rows_count)

	competitor_products, my_products = search.read_dataset();
	
	search.train(competitor_products, my_products)

	search.save()

	new_search = MLSearch(rows_count)

	new_search.load()

	inputs = ['Arctic Cooling Freezer i11',
	'Arctic Cooling Freezer i1',
	'Циркуляционный насос Насосы плюс оборудование BPS 25-6-130',
	'Пенал мягкий, 3 отделения, Goal',
	'Весы электронные BabyOno 291',
	'Весы электронные BabyOno 293',
	'Весы электронные BabyOno',
	'Весы BabyOno 293',
	'электронные BabyOno 293',
	'Весы электронные 293',
	'электронные 293',
	'Весы 293',
	'Call of Duty: Black Ops 2 Uprising (DLC)',
	'Doom 4 [PC-Jewel]']

	for input in inputs:
		output, proba = new_search.predict(input)
		print('{0} => {1} \nwith proba = {2}'.format(input, output, proba))

	search.evaluate(competitor_products, my_products)
