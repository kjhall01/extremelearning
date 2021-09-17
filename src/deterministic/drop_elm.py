import numpy as np
from scipy.spatial.distance import cdist

class DropRegressor:
	"""Extreme Learning Machine with Dropout / DropConnect / Biased or not"""
	def __init__(self, hidden_layer_size=500, activation='sigm', dropconnect_pr=0.5, dropout_pr=0.5, dropconnect_bias_pctl=0.9, dropout_bias_pctl=0.9):
		self.hidden_layer_size = hidden_layer_size
		assert activation in ['sigm', 'relu', 'tanh', 'lin', 'rbf_l1', 'rbf_l2', 'rbf_linf'], 'invalid activation function {}'.format(activation)
		self.activation = activation

		assert  (0.0 <= dropconnect_pr <= .9999), 'dropconnect probability must be [0,1)'
		assert  (0.0 <= dropout_pr <= .9999), 'dropout probability must be [0,1)'
		assert dropconnect_bias_pctl is None or (0.0 <= dropconnect_bias_pctl <= .9999), 'biased dropconnect percentile threshold must be [0,1)'
		assert dropout_bias_pctl is None or (0.0 <= dropout_bias_pctl <= .9999), 'biased dropout percentile threshold must be [0,1)'

		self.dropconnect_pr = dropconnect_pr
		self.dropout_pr = dropout_pr
		self.dropconnect_bias_pctl = dropconnect_bias_pctl
		self.dropout_bias_pctl = dropout_bias_pctl
		self.b = None

	def fit(self, x, y, c=1):
		assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
		x_features, y_features = x.shape[1], y.shape[1]
		weights = np.random.randn(self.hidden_layer_size, x_features)

		if self.dropconnect_bias_pctl is not None:
			weight_mask = np.random.rand(*weights.shape)
			pctl = int(self.dropconnect_bias_pctl*100)
			pctl = np.percentile(weights, pctl)
			weight_mask[weights >= pctl] = 1.0 #if its greater than pctl, keep it
			weight_mask[weight_mask < self.dropconnect_pr] = 0.0 #if its less than pctl and less than dropout pr, set to 0
			weight_mask[weight_mask > 0] = 1.0 # else set to  1.0
			weights = weights*weight_mask
		else:
			weight_mask = np.random.rand(*weights.shape)
			weight_mask[weight_mask < self.dropconnect_pr] = 0.0
			weight_mask[weight_mask >= self.dropout_pr] = 1.0
			weights = weights*weight_mask

		self.hidden_neurons = [ (np.squeeze(weights[i,:]), np.random.randn(1)) for i in range(weights.shape[0])]
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T

		if self.dropout_bias_pctl is not None :
			neuron_mask = np.random.rand(h.shape[1])
			pctl = int(self.dropout_bias_pctl*100)
			pctl = np.percentile(h, pctl)
			neuron_mask[h.sum(axis=0) >= pctl] = 1.0
			neuron_mask[neuron_mask < self.dropout_pr] =0.0
			neuron_mask[neuron_mask > 0] = 1.0
		else:
			neuron_mask = np.random.rand(h.shape[1])
			neuron_mask[neuron_mask < self.dropout_pr] =0.0
			neuron_mask[neuron_mask >= self.dropout_pr] = 1.0
		neuron_mask = np.asarray([neuron_mask for i in range(x_features)]).T
		weights = weights * neuron_mask

		self.hidden_neurons = [ (np.squeeze(weights[i,:]), np.random.randn(1)) for i in range(weights.shape[0])]
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
