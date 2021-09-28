import numpy as np
import copy
import argparse
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import datetime as dt
from sklearn.metrics import roc_auc_score


class FlexELM:
	"""Probabilistic Output Extreme Learning Machine"""
	def __init__(self, hidden_layer_size=5, initialization='random', pruning='none', pca=-999, c=1, preprocessing='std', dropconnect_pr=-1.0, dropout_pr=-1.0, verbose=False):
		assert type(hidden_layer_size) == int and hidden_layer_size > 0, 'Invalid hidden_layer_size {}'.format(hidden_layer_size)
		assert type(initialization) == str and initialization in ['random', 'pca'], 'Invalid initialization {}'.format(initialization)
		assert type(pruning) == str and pruning in ["none", "prune", "pca"], 'Invalid pruning {}'.format(pruning)
		assert type(pca) in [int, float], 'Invalid pca {}'.format(pca)
		assert type(c) is int, 'Invalid C {}'.format(c)
		assert type(preprocessing) is str and preprocessing in ['std', 'minmax', 'none'], 'Invalid preprocessing {}'.format(preprocessing)
		assert type(dropconnect_pr) is float, 'Invalid DropConnect Probability Threshold {}'.format(dropconnect_pr)
		assert type(dropout_pr) is float, 'Invalid DropOut Probability Threshold {}'.format(dropout_pr)


		self.initialization = initialization
		self.pruning = pruning
		self.verbose=verbose
		self.dropconnect_pr = dropconnect_pr
		self.dropout_pr = dropout_pr
		self.pca_retained = pca if pca != -1 else None
		self.c = c
		self.hidden_layer_size = hidden_layer_size
		self.preprocessing = preprocessing

	def fit(self, x, y):
		y[y<0.5] = 0.0001
		y[y>0.5] = 0.9999

		# first, take care of preprocessing
		if self.preprocessing == 'std':
			self.mean, self.std = x.mean(axis=0), x.std(axis=0)
			x = (x - self.mean) / self.std # scales to std normal dist
			if self.verbose:
				print('{} Applied Standard Normal Scaling '.format(dt.datetime.now()))
		if self.preprocessing == 'minmax':
			self.min, self.max = x.min(axis=0), x.max(axis=0)
			x = ((x - self.min) / (self.max - self.min)  ) * 2 - 1 #scales to [-1, 1]
			if self.verbose:
				print('{} Applied MinMax Scaling '.format(dt.datetime.now()))

		# now, if anything needs to train a PCA on the input data, do it (PCA initialization, or PCA transformation)
		if self.pca_retained != -999 or self.initialization == 'pca':
			self.pca = PCA(n_components=None if self.pca_retained==-999 or self.pca_retained == -1 else self.pca_retained )
			self.pca.fit(x)
			if self.verbose:
				print('{} Fit PCA on X with n_compnents={} '.format(dt.datetime.now(), None if self.pca_retained==-999 or self.pca_retained == -1 else self.pca_retained))

		# apply PCA transformation to input data if applicable
		if self.pca_retained != -999:
			x = self.pca.transform(x)
			if self.verbose:
				print('{} Applied PCA Transformation to X '.format(dt.datetime.now()))

		# after transformation, check feature dim
		x_features, y_features = x.shape[1], y.shape[1]

		# now, initialize weights
		w = np.random.randn(x_features, self.hidden_layer_size)
		b = np.random.randn(1, self.hidden_layer_size)
		if self.verbose:
			print('{} Randomly Initialized W and B '.format(dt.datetime.now()))

		if self.initialization == 'pca':
			w = self.pca.components_[:x_features,:self.hidden_layer_size]
			if self.verbose:
				print('{} Initialized W and B according to PCA coefficients '.format(dt.datetime.now()))

		# now apply dropconnect if applicable
		if self.dropconnect_pr > 1:
			weight_mask = np.random.rand(*w.shape)
			pctl = int(self.dropconnect_pr)
			pctl = np.percentile(w, pctl)
			weight_mask[w >= pctl] = 1.0 #if its greater than pctl, keep it
			weight_mask[weight_mask < self.dropconnect_pr / 100.0] = 0.0 #if its less than pctl and less than dropout pr, set to 0
			weight_mask[weight_mask > 0] = 1.0 # else set to  1.0
			w = w*weight_mask
			if self.verbose:
				print('{} Applied Biased DropConnect with percentile {} and probability {} '.format(dt.datetime.now(), pctl, self.dropconnect_pr / 100.0 ))
		elif self.dropconnect_pr > 0:
			weight_mask = np.random.rand(*w.shape)
			weight_mask[weight_mask < self.dropconnect_pr] = 0.0
			weight_mask[weight_mask >= self.dropconnect_pr] = 1.0
			w = w*weight_mask
			if self.verbose:
				print('{} Applied DropConnect with probability {} '.format(dt.datetime.now(), self.dropconnect_pr  ))

		# now apply dropout if applicable
		if self.dropout_pr > 0:
			self.hidden_neurons = [ (w[:,i], b[:,i]) for i in range(w.shape[1])]
			h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
			if self.dropout_pr > 1 :
				neuron_mask = np.random.rand(w.shape[1])
				pctl = int(self.dropout_pr)
				pctl = np.percentile(h, pctl)
				neuron_mask[h.mean(axis=0) >= pctl] = 1.0
				neuron_mask[neuron_mask < self.dropout_pr / 100.0] =0.0
				neuron_mask[neuron_mask > 0] = 1.0
				if np.sum(neuron_mask) < 1:
					neuron_mask[0] = 1.0
				if self.verbose:
					print('{} Applied Biased DropOut with percentile {} and probability {} '.format(dt.datetime.now(), pctl, self.dropout_pr /100.0 ))
			else:
				neuron_mask = np.random.rand(w.shape[1])
				neuron_mask[neuron_mask < self.dropout_pr] =0.0
				neuron_mask[neuron_mask >= self.dropout_pr] = 1.0
				if np.sum(neuron_mask) < 1:
					neuron_mask[0] = 1.0
				if self.verbose:
					print('{} Applied DropOut with probability {} '.format(dt.datetime.now(), self.dropout_pr  ))
			w = np.hstack([w[:,i].reshape(-1,1)  for i in range(w.shape[1]) if np.sum(neuron_mask[i]) > 0 ])

		# now, appy pruning as relevant:
		if self.pruning == 'pca':
			w= PCA(n_components=None if self.pca_retained == -1 or self.pca_retained == -999 else self.pca_retained).fit_transform(w)
			if self.verbose:
				print('{} Applied PCA pruning with n_components of  {} '.format(dt.datetime.now(),  None if self.pca_retained == -1 or self.pca_retained == -999 else self.pca_retained ))

		elif self.pruning == "prune":
			self.hidden_neurons = [ (w[:,i], b[:,i]) for i in range(self.hidden_layer_size)]
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
				inv_hth_plus_ic = np.linalg.pinv( hth + np.eye(hth.shape[0]) / (2**self.c) )
				ht_logs = np.dot(np.transpose(h), np.log(1 - y) - np.log(y))
				self.beta = -1 * np.dot( inv_hth_plus_ic, ht_logs)
				preds = self.predict(x)
				acc = accuracy_score(np.argmax(y, axis=-1), preds)
				aics.append(self._aic(x.shape[0], acc, i+1))

			aics = np.asarray(aics)
			best = np.argmin(aics)
			self.hidden_neurons = new_h[:best+1]
			w = np.hstack([np.squeeze(neuron[0]).reshape(-1,1) for neuron in self.hidden_neurons])
			b = np.hstack([np.squeeze(neuron[1]).reshape(-1,1) for neuron in self.hidden_neurons])
			if self.verbose:
				print('{} Applied AIC / Chi2 Pruning '.format(dt.datetime.now()))
		b = b[:, :w.shape[1]]

		self.hidden_neurons = [ (w[:, i], b[:,i]) for i in range(w.shape[1])]
		self.H = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		hth = np.dot(np.transpose(self.H), self.H)
		inv_hth_plus_ic = np.linalg.pinv( hth + np.eye(hth.shape[0]) / (2**self.c) )
		ht_logs = np.dot(np.transpose(self.H), np.log(1 - y) - np.log(y))
		self.beta = -1 * np.dot( inv_hth_plus_ic, ht_logs)
		if self.verbose:
			print('{} Solved POELM '.format(dt.datetime.now() ))


	def predict(self, x):
		# first, take care of preprocessing
		if self.preprocessing == 'std':
			x = (x - self.mean) / self.std # scales to std normal dist
			if self.verbose:
				print('{} Applied Standard Normal Scaling '.format(dt.datetime.now()))
		if self.preprocessing == 'minmax':
			x = ((x - self.min) / (self.max - self.min)  ) * 2 - 1 #scales to [-1, 1]
			if self.verbose:
				print('{} Applied MinMax Scaling '.format(dt.datetime.now()))

		if self.pca_retained != -999:
			x = self.pca.transform(x)
			if self.verbose:
				print('{} Applied PCA Transformation to Forecast X  '.format(dt.datetime.now()))

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
		if self.preprocessing == 'std':
			x = (x - self.mean) / self.std # scales to std normal dist
			if self.verbose:
				print('{} Applied Standard Normal Scaling '.format(dt.datetime.now()))
		if self.preprocessing == 'minmax':
			x = ((x - self.min) / (self.max - self.min)  ) * 2 - 1 #scales to [-1, 1]
			if self.verbose:
				print('{} Applied MinMax Scaling '.format(dt.datetime.now()))

		if self.pca_retained != -999:
			x = self.pca.transform(x)
			if self.verbose:
				print('{} Applied PCA Transformation to Forecast X  '.format(dt.datetime.now()))

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

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Produce Tercile Probabilistic Forecasts with POELM')
	parser.add_argument('input', type=str,  help='CSV Input filepath with years in the first column, target value in the second column, then N columns of predictor variables')
	parser.add_argument('-o', '--output', type=str, nargs='?',default='poelm_output.csv', help='Filepath to CSV Output File' )
	parser.add_argument('-p', '--pca', type=float, nargs='?',default=-999, help='PCA components retained in predictors, can be integer number of modes, or decimal percent variance [0,1). -1 to keep all pca modes')
	parser.add_argument('-i', '--initialization', type=str,nargs='?', default='random', help='type of initialization to use, can be any of ["random", "pca"]')
	parser.add_argument('-d', '--nd_runs', type=int,nargs='?', default=100, help='number of non-deterministic ensemble members to use to produce a mean forecast- probably dont use this with pca initialization, its a waste!')
	parser.add_argument('-r', '--pruning', type=str,nargs='?', default='none', help='type of hidden layer pruning to use- can be one of ["none", "pca", "prune"]')
	parser.add_argument('-n', '--neurons', type=int,nargs='?', default=5, help='number of hidden layer neurons to use')
	parser.add_argument('-c', '--c', type=int,nargs='?', default=0, help='power of 2 to use as the C in the POELM math')
	parser.add_argument('-s', '--preprocessing', type=str,nargs='?', default='std', help='type of preprocessing to apply to data , can be in ["std", "minmax", "none"]')
	parser.add_argument('-f', '--n_forecasts', type=int,nargs='?', default=0, help='number of forecast predictors at the end of the input file without labels  ')
	parser.add_argument('-u', '--upper_threshold', type=float,nargs='?', default=4.0, help='upper limit of Near Normal category such that value <= UL  ')
	parser.add_argument('-l', '--lower_threshold', type=float,nargs='?', default=3.0, help='lower limit of Near Normal category such taht value >= LL  ')
	parser.add_argument('-q', '--dropconnect', type=float,nargs='?', default=-1.0, help='DropConnect threshold-  probability between [0,1). if  >1, uses threshold /100 as a percentile for biased dropconnect  ')
	parser.add_argument('-k', '--dropout', type=float, nargs='?',default=-1.0, help='DropOut threshold-  probability between [0,1). if  >1, uses threshold /100 as a percentile for biased dropout  ')
	parser.add_argument('-v', '--verbose', type=int, nargs='?',default=1, help='Whether to print progress')
	parser.add_argument('-g', '--debug', type=int,nargs='?', default=0, help='whether to print every little thing that happens ')
	args = parser.parse_args()
	cmd = ' '.join(f'{k}={v}' for k, v in vars(args).items())

	data = np.genfromtxt(args.input, delimiter=',', dtype=float)
	years = np.squeeze(data[:,0]).reshape(-1,1)
	labels = np.squeeze(data[:,1]).reshape(-1,1)
	x, y = data[:data.shape[0]-args.n_forecasts, 2:], data[:data.shape[0]-args.n_forecasts, 1].reshape(-1,1)
	forecast_x = data[-1*args.n_forecasts:, 2:]
	if args.verbose:
		print('{} Read Data'.format(dt.datetime.now() ))

	t = np.zeros((y.shape[0], 3))
	for i in range(y.shape[0]):
		if y[i,0] < args.lower_threshold:
			t[i,0] = 1.0
		elif y[i,0] > args.upper_threshold:
			t[i,2] = 1.0
		else:
			t[i,1] = 1.0

	windows = []
	for i in range(x.shape[0]):
		if i == 0:
			windows.append([copy.deepcopy(x[1:,:]), copy.deepcopy(t[1:,:])] )
		elif i == x.shape[0]-1:
			windows.append([copy.deepcopy(x[:-1,:]), copy.deepcopy(t[:-1, :])])
		else:
			windows.append([np.vstack([copy.deepcopy(x[:i,:]), copy.deepcopy(x[i+1:,:])]), np.vstack([copy.deepcopy(t[:i,:]), copy.deepcopy(t[i+1:,:])])])

	if args.verbose:
		print('{} Prepared CrossValidation Windows'.format(dt.datetime.now()))

	start = dt.datetime.now()
	total = len(windows)*args.nd_runs
	count = 0
	if args.verbose:
		print('CROSS-VALIDATING [' + 25*' ' + '] ({}/{}) {}'.format(count, total, dt.datetime.now()-start), end='\r')

	hcsts = []
	for i in range(len(windows)):
		hcsts2 = []
		x_train, y_train = windows[i]
		for j in range(args.nd_runs):
			poelm = FlexELM(hidden_layer_size=args.neurons, initialization=args.initialization, pruning=args.pruning, pca=args.pca, c=args.c, preprocessing=args.preprocessing, dropconnect_pr=args.dropconnect, dropout_pr=args.dropout, verbose=args.debug )
			poelm.fit(copy.deepcopy(x_train), copy.deepcopy(y_train))
			hcsts2.append(poelm.predict_proba(x[i, :].reshape(1,-1)	))
			count += 1
			if args.verbose:
				print('CROSS-VALIDATING [' + int(count / total * 25)*'*' + (25 - int(count / total * 25))*' ' + '] ({}/{}) {}'.format(count, total, dt.datetime.now()-start), end='\r')
		hcsts.append(np.nanmean(hcsts2, axis=0))
	hcsts = np.squeeze(np.asarray(hcsts) )
	xvalidated_roc = roc_auc_score(t, hcsts, multi_class='ovr')

	if args.verbose:
		print('CROSS-VALIDATING [' + 25*'*' + '] ({}/{}) {}'.format(count, total, dt.datetime.now()-start))

	if args.verbose:
		print('{} Producing Real-Time Forecasts'.format(dt.datetime.now()))

	hcsts2 = []
	for j in range(args.nd_runs):
		poelm = FlexELM(hidden_layer_size=args.neurons, initialization=args.initialization, pruning=args.pruning, pca=args.pca, c=args.c, preprocessing=args.preprocessing, dropconnect_pr=args.dropconnect, dropout_pr=args.dropout, verbose=args.debug )
		poelm.fit(copy.deepcopy(x), copy.deepcopy(t))
		hcsts2.append(poelm.predict_proba(copy.deepcopy(forecast_x))	)
	fcsts = np.nanmean(hcsts2, axis=0)

	if args.verbose:
		print('{} Producing Output File'.format(dt.datetime.now()))

	vals = np.vstack([hcsts, fcsts])
	vals2 = np.vstack([t, np.ones_like(fcsts)*-1])
	vals = np.hstack([years, labels, vals, vals2])

	np.savetxt(args.output, vals,  delimiter=',', header='Probabilistic Forecasts produced with ELM Probabilistic Forecast Tool\nBy Kyle Hall & Nachiketa Acharya\nCROSSVALIDATED ROC AUC SCORE: {}\nArgs: {}\nyear,label,PredictedBelowNormal,PredictedNearNormal,PredictedAboveNormal,TrueBelowNormal,TrueNearNormal,TrueAboveNormal'.format(xvalidated_roc, cmd))
