from .probabilistic import *
from .deterministic import *


deterministic = [ELMRegressor, DropRegressor, PCIRegressor, PCPRegressor, PCTRegressor, PrunedRegressor ]
probabilistic = [ELMClassifier, DropClassifier, PCIClassifier, PCPClassifier, PCTClassifier, PrunedClassifier]
