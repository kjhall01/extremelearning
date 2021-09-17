import numpy as np
from sklearn.decomposition import PCA
from scipy.special import softmax

class PCIClassifier:
	"""Principal Components Transformation Probabilistic Output Extreme Learning Machine"""
	def __init__(self,retained=None):
		self.retained = retained

	def fit(self, x, y, c=1):
		y[y<0.5] = 0.0001
		y[y>0.5] = 0.9999
		#assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
		self.pca = PCA(n_components=self.retained)
		self.pca.fit(x)
		x = self.pca.transform(x)
		x_features, y_features = x.shape[1], y.shape[1]
		self.hidden_neurons = [ (self.pca.components_.T[:x_features,i], np.random.randn(1)) for i in range(self.pca.components_.T.shape[1])]
		self.H = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		hth = np.dot(np.transpose(self.H), self.H)
		inv_hth_plus_ic = np.linalg.pinv( hth + np.eye(hth.shape[0]) / c )
		ht_logs = np.dot(np.transpose(self.H), np.log(1 - y) - np.log(y))
		self.beta = -1 * np.dot( inv_hth_plus_ic, ht_logs)

	def predict_proba(self, x):
		x = self.pca.transform(x)
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		ret = 1.0 / ( 1 + np.exp(-1* np.dot(h, self.beta)))
		sums =  np.sum(ret, axis=1)
		ret1 = ret / sums.reshape(-1,1)
		ret2 = softmax(ret, axis=-1)
		retfinal = np.ones(ret.shape)
		retfinal[sums >=1, :] = ret1[sums>=1, :]
		retfinal[sums < 1, :] = ret2[sums<1, :]
		return retfinal

	def predict(self, x):
		x = self.pca.transform(x)
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		ret = 1.0 / ( 1 + np.exp(-1* np.dot(h, self.beta)))
		sums =  np.sum(ret, axis=1)
		ret1 = ret / sums.reshape(-1,1)
		ret2 = softmax(ret, axis=-1)
		retfinal = np.ones(ret.shape)
		retfinal[sums >=1, :] = ret1[sums>=1, :]
		retfinal[sums < 1, :] = ret2[sums<1, :]
		return np.argmax(retfinal, axis=-1)

	def _activate(self, a, x, b):
		return 1.0 / (1 + np.exp(-1* np.dot(a, x.T) + b) )
