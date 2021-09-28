import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax

class DropClassifier:
	"""Probabilistic Output Extreme Learning Machine"""
	def __init__(self, hidden_layer_size=5, dropconnect_pr=0.5, dropout_pr=0.5, dropconnect_bias_pctl=None, dropout_bias_pctl=None):
		self.hidden_layer_size = hidden_layer_size

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
		y[y<0.5] = 0.0001
		y[y>0.5] = 0.9999
		#assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
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
			weight_mask[weight_mask >= self.dropconnect_pr] = 1.0
			weights = weights*weight_mask

		self.hidden_neurons = [ (np.squeeze(weights[i,:]), np.random.randn(1)) for i in range(self.hidden_layer_size)]
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T

		if self.dropout_bias_pctl is not None :
			neuron_mask = np.random.rand(*h.shape)
			pctl = int(self.dropout_bias_pctl*100)
			pctl = np.percentile(h, pctl)
			neuron_mask[h >= pctl] = 1.0
			neuron_mask[neuron_mask < self.dropout_pr] =0.0
			neuron_mask[neuron_mask > 0] = 1.0
		else:
			neuron_mask = np.random.rand(*h.shape)
			neuron_mask[neuron_mask < self.dropout_pr] =0.0
			neuron_mask[neuron_mask >= self.dropout_pr] = 1.0
		weights = np.asarray([weights[i,:]  for i in range(weights.shape[0]) if np.sum(neuron_mask[i]) > 0 ])

		self.hidden_neurons = [ (np.squeeze(weights[i,:]), np.random.randn(1)) for i in range(weights.shape[0])]
		self.H = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T

		hth = np.dot(np.transpose(self.H), self.H)
		inv_hth_plus_ic = np.linalg.pinv( hth + np.eye(hth.shape[0]) / c )
		ht_logs = np.dot(np.transpose(self.H), np.log(1 - y) - np.log(y))
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
