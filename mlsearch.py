import re
import sys

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

	INPUT_FILE = "/home/sonya/mlsearch/dataset/product-mappings-dataset-full-part-100000.csv"
	ROWS_COUNT = 5000


	def __init__(self):
		self.vectorizer = CountVectorizer(token_pattern=r'(?u)\b[\w\w\d]+\b')
		self.tfidf = TfidfTransformer(norm='l2')

		self.clf = LinearSVC() # yes
		# self.clf = SGDClassifier() # yes
		# self.clf = RidgeClassifier() # ?
		# self.clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0) # ?


	def save(self):
		print("save model")
		dump(self.clf, 'saved/clf')
		dump(self.vectorizer, 'saved/vectorizer')
		dump(self.tfidf, 'saved/tfidf')


	def load(self):
		print("load saved model")
		self.clf = load('saved/clf')
		self.vectorizer = load('saved/vectorizer')
		self.tfidf = load('saved/tfidf')


	def evaluate(self, test_docs, my_products):
		test_vectors = self.transform(self.vectorizer, test_docs)
		test_vectors = self.transform(self.tfidf, test_vectors)

		predicted = self.clf.predict(test_vectors)
		print(np.mean(predicted == my_products))


	def predict(self, input):
		input_vectors = self.vectorizer.transform([input])
		input_vectors = self.tfidf.transform(input_vectors)

		# t0 = time()

		predicted = self.clf.predict(input_vectors)[0]

		# probas = clf.predict_proba(test_transformed_vectors)
		probas = self.clf.decision_function(input_vectors)[0]

		# print("\nprediction done in %fs" % (time() - t0))
		
		proba = probas[np.where(self.clf.classes_ == predicted)]
		proba = 1 / (1 + exp(-proba)) # for decision_function

		# input_important_words = self.make_important_words(input, predicted)

		if self.has_nonexistent_word(input, predicted):
			predicted = 'not found'
			proba = 0.0

		return predicted, proba


	# def check_important_words(self, doc, important_words):
	# 	# if not important_words:
	# 	# 	return False

	# 	doc = doc.lower()

	# 	splited_doc = self.vectorizer.build_tokenizer()(doc.lower())

	# 	i = 0

	# 	for important_word in important_words:
	# 		if important_word in doc:
	# 			i += 1
	# 			break

	# 		j = 0
	# 		for doc_word in splited_doc:
	# 			if doc_word in important_word:
	# 				j += 1

	# 		if j > 1:
	# 			i += 1

	# 	# print(i, j)

	# 	if i != len(important_words):
	# 		return False
		
	# 	return True


	def read_dataset(self):
		df = pd.read_csv(self.INPUT_FILE, header=0, nrows=self.ROWS_COUNT)
		return df["competitor_model"], df["product_model"]


	def train(self):

		X, y = search.read_dataset();

		# FIT & TRANSFORM TO VECTORS

		t0 = time()
		self.vectorizer.fit(X)
		print('vectorizer fitted in %fs' % (time() - t0,))
		
		X = self.vectorizer.transform(X)

		# FIT & TRANSFORM VECTORS WITH FREQ

		t0 = time()
		self.tfidf.fit(X)
		print('tfidf fitted in %fs' % (time() - t0,))
		
		X = self.tfidf.transform(X)

		# TRAINING

		print('start fitting model, rows_count = ' + str(self.ROWS_COUNT))
		t0 = time()
		self.clf.fit(X, y)
		print("done in %fs" % (time() - t0))


	def has_nonexistent_word(self, doc, predicted):
		coef = self.clf.coef_[np.where(self.clf.classes_ == predicted)][0].ravel()
		vocabulary_weights = dict(zip(self.vectorizer.get_feature_names(), coef))

		for word in self.vectorizer.build_tokenizer()(doc.lower()):
			if word not in vocabulary_weights:
				return True


	# def make_important_words(self, input, predicted):
	# 	coef = self.clf.coef_[np.where(self.clf.classes_ == predicted)][0].ravel()
	# 	vocabulary_weights = dict(zip(self.vectorizer.get_feature_names(), coef))

	# 	# print(vocabulary_weights)

	# 	input_word_weights = {}
	# 	predicted_word_weights = {}
		
	# 	for word in self.vectorizer.build_tokenizer()(input.lower()):
	# 		if word in vocabulary_weights:
	# 			input_word_weights.update({word: vocabulary_weights[word]})
	# 		else:
	# 			input_word_weights.update({word: 3});

	# 	input_word_weights = sorted(input_word_weights.items(), key=lambda kv: kv[1])
	# 	# print(input_word_weights)
	# 	input_important_words = [i[0] for i in input_word_weights if i[1] > 2.0][-1:]

	# 	return input_important_words


if __name__ == "__main__":	
	search = MLSearch()
	search.train()
	search.save()