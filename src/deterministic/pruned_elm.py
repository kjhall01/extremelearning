import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score

class PrunedRegressor:
	"""Extreme Learning Machine"""
	def __init__(self, hidden_layer_size=500, activation='sigm'):
		self.hidden_layer_size = hidden_layer_size
		assert activation in ['sigm', 'relu', 'tanh', 'lin', 'rbf_l1', 'rbf_l2', 'rbf_linf'], 'invalid activation function {}'.format(activation)
		self.activation = activation
		self.b = None

	def fit(self, x, y, c=1):
		assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
		x_features, y_features = x.shape[1], y.shape[1]
		self.hidden_neurons = [ (np.random.randn(x_features), np.random.randn(1)) for i in range(self.hidden_layer_size)]
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		scores = np.asarray([mutual_info_regression(h[:,i].reshape(-1,1), np.squeeze(y)) for i in range(h.shape[1]) ])
		new_h = []
		for i in range(len(scores)):
			new_h.append(self.hidden_neurons[np.argmax(scores)])
			scores[np.argmax(scores)] = -1

		aics = []
		for i in range(len(scores)):
			self.hidden_neurons = new_h[:i+1]
			h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
			hh = np.dot(np.transpose(h), h)
			ht = np.dot(np.transpose(h), y)
			self.b = np.dot(np.linalg.pinv(hh), ht)
			preds = self.predict(x)
			acc = r2_score(y, preds)
			aics.append(self._aic(x.shape[0], acc, i+1))

		aics = np.asarray(aics)
		best = np.argmin(aics)
		self.hidden_neurons = new_h[:best+1]
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		hh = np.dot(np.transpose(h), h)
		ht = np.dot(np.transpose(h), y)
		self.b = np.dot(np.linalg.pinv(hh), ht)


	def predict(self, x):
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		return np.dot(h, self.b)

	def _activate(self, a, x, b):
		if self.activation == 'sigm':
			return 1 / (1 + np.exp(np.dot(x, a) + b))
		elif self.activation == 'tanh':
			return np.tanh(np.dot(x, a) + b)
		elif self.activation == 'relu':
			return np.max(0, np.dot(x, a) + b)
		elif self.activation == 'lin':
			return np.dot(x, a) + b
		elif self.activation == 'rbf_l1':
			return np.exp(-cdist(x, a.T, "cityblock")**2 / b)
		elif self.activation == 'rbf_l2':
			return np.exp(-cdist(x, a.T, "euclidean")**2 / b)
		elif self.activation == 'rbf_linf':
			return np.exp(-cdist(x, a.T, "chebyshev")**2 / b)
		else:
			assert False, 'Invalid activation function {}'.format(self.activation)

	def _aic(self, N, accuracy, S):
		return 2 * N * np.log(((1.00001 - accuracy) / N)**2 / N ) + S
