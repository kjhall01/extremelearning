import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

class PCIRegressor:
	"""Principal Components Analysis-Initialized Extreme Learning Machine"""
	def __init__(self, retained=None, activation='sigm'):
		assert self.activation in ['sigm', 'relu', 'tanh', 'lin', 'rbf_l1', 'rbf_l2', 'rbf_linf'], 'invalid activation function {}'.format(activation)
		self.activation = activation
		self.retained = retained
		self.b = None

	def fit(self, x, y, c=1):
		assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
		self.pca = PCA(n_components=self.retained)
		self.pca.fit(x)
		x = self.pca.transform(x)
		x_features, y_features = x.shape[1], y.shape[1]
		self.hidden_neurons = [ (self.pca.components_.T[:x_features,:], np.random.randn(1)) for i in range(self.pca.components_.shape[0])]
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		hh = np.dot(np.transpose(h), h)
		ht = np.dot(np.transpose(h), y)
		self.b = np.dot(np.linalg.pinv(hh), ht)

	def predict(self, x):
		x = self.pca.transform(x)
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
