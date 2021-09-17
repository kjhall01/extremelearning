import numpy as np
from sklearn.datasets import make_classification
import src as learning_machines
from sklearn.preprocessing import OneHotEncoder


X, y = make_classification(10, 30, n_informative=4, n_classes=3)
y1 = np.zeros((y.shape[0],3), dtype=float)
print('{0:<16s}'.format('OBSERVED'), y)
y = y.reshape(-1,1)

for i in range(y.shape[0]):
	y1[i,y[i,0]]= 1
y=y1

preds = []
for lm in learning_machines.probabilistic:
	elm = lm()
	elm.fit(X, y)
	print('{0:<16s}'.format(lm.__name__), elm.predict(X))
