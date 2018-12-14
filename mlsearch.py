import re
import sys

import numpy as np
import pandas as pd

from time import time
from math import exp

# from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# selection or stop words

class MLSearch:

	def __init__(self, input_file, rows_count):
		self.input_file = input_file
		self.rows_count = rows_count
		self.vectorizer = CountVectorizer(token_pattern=r'(?u)\b[\w\w\d]+\b')
		self.transformer = TfidfTransformer(use_idf=False)
		self.clf = LinearSVC(penalty="l2", dual=False)
		# self.clf = SGDClassifier(loss="log", penalty='elasticnet') #yes
		# self.clf = RidgeClassifier()
		# self.clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0)


	def select_features(self, vectors):
		print("selecting features")
		t0 = time()
		
		# selector = VarianceThreshold()
		# selector = SelectKBest(chi2, k=3000)
		# sum_words = vectors.sum(axis=0)
		# total = sum_words.sum()
		# words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
		# words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
		# print(words_freq)
		# print(total)

		print('selected in %fs, shape: %s' % (time() - t0, vectors.shape,))


	def predict(self, input):
		test = [input]
		test_vectors = self.vectorizer.transform(test)
		test_vectors = self.transformer.transform(test_vectors)

		t0 = time()

		print('\npredict...')

		predicted = self.clf.predict(test_vectors)[0]
		#probas = clf.predict_proba(test_transformed_vectors)
		probas = self.clf.decision_function(test_vectors)[0]

		print("done in %fs" % (time() - t0))

		proba_per_class_dict = dict(zip(self.clf.classes_, probas))
		
		proba = proba_per_class_dict[predicted]

		proba = 1 / (1 + exp(-proba)) # for decision_function

		return predicted, proba


	def read_dataset(self):
		df = pd.read_csv(input_file, header=0, nrows=self.rows_count)
		return df["competitor_model"], df["product_model"]


	def train(self):

		competitor_products, my_products = self.read_dataset();

		t0 = time()

		print('\nfit vectorizer')
		self.vectorizer.fit(competitor_products)

		print('transform docs to vectors')
		vectors = self.vectorizer.transform(competitor_products)

		print('\nfitted & transformed in %fs, shape: %s' % (time() - t0, vectors.shape,))

		t0 = time()

		print('\nfit tfidf transformer')
		self.transformer.fit(vectors)

		print('transform vectors to frequencies')
		vectors = self.transformer.transform(vectors)

		print('\nfitted & transformed in %fs, shape: %s' % (time() - t0, vectors.shape,))

		t0 = time()

		print('\nstart fitting model, rows_count = ' + str(self.rows_count))
		self.clf.fit(vectors, my_products)
		
		print("done in %fs" % (time() - t0))


if __name__ == "__main__":

	rows_count = int(sys.argv[1])
	input_file = "/home/sonya/zoomos/product-mappings-dataset-full-part-100000.csv"
	
	search = MLSearch(input_file, rows_count)
	
	search.train()

	inputs = ['Arctic Cooling Freezer i11',
	'Arctic Cooling Freezer i1',
	'Циркуляционный насос Насосы плюс оборудование BPS 25-6-130',
	'Пенал мягкий, 3 отделения, Goal',
	'Весы электронные BabyOno 291',
	'Весы BabyOno 293',
	'электронные BabyOno 293',
	'Весы электронные 293',
	'электронные 293',
	'Весы 293',
	'Call of Duty: Black Ops 2 Uprising (DLC)',
	'Doom 4 [PC-Jewel]']

	for input in inputs:
		output, proba = search.predict(input)
		print('\n{0} => {1} \nwith proba = {2}'.format(input, output, proba))
