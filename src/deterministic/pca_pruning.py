import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

class PCPRegressor:
	"""Principal Components Pruning Learning Machine"""
	def __init__(self, hidden_layer_size=500, n_components=0.7, activation='sigm'):
		self.hidden_layer_size = hidden_layer_size
		assert activation in ['sigm', 'relu', 'tanh', 'lin', 'rbf_l1', 'rbf_l2', 'rbf_linf'], 'invalid activation function {}'.format(activation)
		self.activation = activation
		self.n_components = n_components
		self.b = None

	def fit(self, x, y, c=1):
		assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
		x_features, y_features = x.shape[1], y.shape[1]
		weights = np.random.randn(x_features, self.hidden_layer_size)
		weights = PCA(n_components=self.n_components).fit_transform(weights)

		self.hidden_neurons = [ (np.squeeze(weights[:, i]), np.random.randn(1)) for i in range(weights.shape[1])]
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		hh = np.dot(np.transpose(h), h)
		ht = np.dot(np.transpose(h), y)
		self.b = np.dot(np.linalg.pinv(hh), ht)

	def predict(self, x):
		assert self.b is not None, 'Must fit learning machine first'
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		return np.dot(h, self.b)

	def _activate(self, a, x, b):
		if self.activation == 'sigm':
			return 1 / (1 + np.exp(np.dot(x, a) + b))
		elif self.activation == 'tanh':
			return np.tanh(np.dot(x, a) + b)
		elif self.activation == 'relu':
			ret =  np.dot(x, a) + b
			ret[ret < 0] = 0
			return ret
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
