import numpy as np
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from scipy.special import softmax

class PrunedClassifier:
	"""Probabilistic Output Extreme Learning Machine"""
	def __init__(self, hidden_layer_size=5):
		self.hidden_layer_size = hidden_layer_size

	def fit(self, x, y, c=1):
		y[y<0.5] = 0.0001
		y[y>0.5] = 0.9999
		#assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
		x_features, y_features = x.shape[1], y.shape[1]
		self.hidden_neurons = [ (np.random.randn(x_features), np.random.randn(1)) for i in range(self.hidden_layer_size)]
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T

		scores = np.asarray([np.squeeze(chi2(h[:,i].reshape(-1,1), np.argmax(y, axis=-1)))[0] for i in range(h.shape[1]) ])
		new_h = []
		for i in range(scores.shape[0]):
			new_h.append(self.hidden_neurons[np.argmax(scores)])
			scores[np.argmax(scores)] = -1

		aics = []
		for i in range(len(scores)):
			self.hidden_neurons = new_h[:i+1]
			h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
			hth = np.dot(np.transpose(h), h)
			inv_hth_plus_ic = np.linalg.pinv( hth + np.eye(hth.shape[0]) / c )
			ht_logs = np.dot(np.transpose(h), np.log(1 - y) - np.log(y))
			self.beta = -1 * np.dot( inv_hth_plus_ic, ht_logs)
			preds = self.predict(x)
			acc = accuracy_score(np.argmax(y, axis=-1), preds)
			aics.append(self._aic(x.shape[0], acc, i+1))

		aics = np.asarray(aics)
		best = np.argmin(aics)
		self.hidden_neurons = new_h[:best+1]

		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		hth = np.dot(np.transpose(h), h)
		inv_hth_plus_ic = np.linalg.pinv( hth + np.eye(hth.shape[0]) / c )
		ht_logs = np.dot(np.transpose(h), np.log(1 - y) - np.log(y))
		self.beta = -1 * np.dot( inv_hth_plus_ic, ht_logs)

	def predict(self, x):
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		ret = 1.0 / ( 1 + np.exp(-1* np.dot(h, self.beta)))
		sums =  np.sum(ret, axis=1)
		ret1 = ret / sums.reshape(-1,1)
		ret2 = softmax(ret, axis=-1)
		retfinal = np.ones(ret.shape)
		retfinal[sums >=1, :] = ret1[sums>=1, :]
		retfinal[sums < 1, :] = ret2[sums<1, :]
		return np.argmax(retfinal,axis=-1)

	def predict_proba(self, x):
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		ret = 1.0 / ( 1 + np.exp(-1* np.dot(h, self.beta)))
		sums =  np.sum(ret, axis=1)
		ret1 = ret / sums.reshape(-1,1)
		ret2 = softmax(ret, axis=-1)
		retfinal = np.ones(ret.shape)
		retfinal[sums >=1, :] = ret1[sums>=1, :]
		retfinal[sums < 1, :] = ret2[sums<1, :]
		return retfinal

	def _activate(self, a, x, b):
		return 1.0 / (1 + np.exp(-1* np.dot(a, x.T) + b) )

	def _aic(self, N, accuracy, S):
		return 2 * N * np.log(((1 - accuracy) / N)**2 / N ) + S
