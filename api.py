from flask import Flask
from flask import request
from flask import jsonify

from mlsearch import MLSearch

from webargs import fields
from webargs.flaskparser import use_args

app = Flask(__name__)


@app.route('/search', methods=['GET'])
def search():
	mlsearch = MLSearch()
	mlsearch.load()

	input = request.args.get('q')

	predicted, proba = mlsearch.predict(input)

	return jsonify(predicted=predicted, proba=proba)


if __name__ == '__main__':
    app.run()