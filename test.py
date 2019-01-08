import numpy as np
import pandas as pd

from mlsearch import MLSearch


def test_nonexistent_items():
	items = ['Arctic Cooling Freezer i1',
	'Arctic Cooling Freezer',
	'Циркуляционный насос Насосы плюс оборудование BS 25-6-130',
	'Весы электронные BabyOno 293',
	'Весы электронные BabyOno',
	'Весы BabyOno 293',
	'электронные BabyOno 293',
	'Весы электронные 293',
	'электронные 293',
	'Весы 293',
	'Встраиваемая посудомоечная машина BEKO DIS 1501']

	for item in items:
		predicted, proba = search.predict(item)
		print('{0} => {1}'.format(item, predicted))


def test_splited_model_items():
	items = ['Встраиваемая посудомоечная машина BEKO DIS15010',
	'Встраиваемая посудомоечная машина BEKO DIS 15010']

	for item in items:
		predicted, proba = search.predict(item)
		print('{0} => {1}'.format(item, predicted))


def test_all_items():
	competitor_products, my_products = search.read_dataset();

	data = []

	for item in competitor_products:
		predicted, proba = search.predict(item)
		if proba == 0.0:
			print('{0} => {1}'.format(item, predicted))
		data.append((item, predicted, proba))

	pd.DataFrame(data, columns = ['item', 'predicted', 'proba']).to_csv('results/predictions.csv', ';')


if __name__ == "__main__":
	np.set_printoptions(threshold=np.nan) # show full numpy array

	search = MLSearch()
	search.load()

	test_nonexistent_items()