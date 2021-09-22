---
title: 'ExtremeLearning: A Selection of Pure-Python Extreme Learning Machine Approaches '
tags:
  - Python
  - Extreme Learning Machine
  - Artificial Intelligence
  - Machine Learning
  - ELM
authors:
  - name: Kyle Hall
    orcid: 0000-0003-3723-0662
    affiliation: 1
  - name: Nachiketa Acharya
    orcid: 0000-0003-3010-2158
    affiliation: 2
affiliations:
 - name: International Research Institute for Climate & Society, Columbia University
   index: 1
 - name: Center for Earth System Modeling, Analysis, & Data, Department of Meteorology and Atmospheric Science, The Pennsylvania State University
   index: 2
date: 17 September 2021
bibliography: paper.bibtex
---

# Summary

An "Extreme Learning Machine" (ELM) is a generalized single-layer feed-forward neural network. Instead of optimizing each neuron's weights through backpropagation, like a traditional neural network would, ELM leaves its randomly-initialized hidden layer weights unchanged and uses a Moore-Penrose Generalized Inverse to fit its output layer. [@Huang:2004], [@Penrose:1955]  Since ELM was published in the early 2000s, numerous related methods have been proposed. [@Chen:2017] Each new method slightly changes the underlying ELM Model, either by pruning nodes in the hidden layer, applying Principal Components Analysis to the training data, modifying the hidden layer weight initialization scheme, changing the objective function, or some other new method. `ExtremeLearning` is a fully open-source python package that implements these new modified ELM Regressors and Classifiers in the style of SciKit-Learn, in a way that is compatible with Python parallelism and cluster computing. [@scikit-learn]

# Statement of need

While there are many implementations of Extreme Learning Machine available in Python, no individual package provides the flexibility, consistency, and ease of use required by most data science workflows. Many are very useful for specific domains, or with limited scopes, but none have the same appeal as the pure-python, mainstream Python statistics libraries. [@Akusok:2015] To serve the Python data science community well, a library needs to be comprehensive, well-documented, easy to install, and compatible with the larger data science ecosystem. Whether they lack clear documentation, are not distributed on a mainstream package manager, do not implement a wide enough range of ELM-based methods, or are specific to one scientific domain, the majority of the existing ELM Python libraries are problematic in one way or another.

In addition to the properties required of an effective Python library in general, for ELM, specifically, it is critical that any implementation be compatible with Python's high-performance computing libraries, `multiprocessing` and `dask`. Since ELM models train rapidly, they are uniquely well suited for applications that require numerous models to be trained independently. PyELM-MME, the gridpoint-wise climate forecasting toolkit that precipitated the implementation of `ExtremeLearning`, and other applications like it, can be greatly enhanced by Python parallelism, but only if the underlying statistical libraries they rely on are compatible with the available Python high-performance computing libraries. [@Acharya:2021] 

`ExtremeLearning` is an effective Python library for the implementation of ELM-based data science solutions. It is well-documented-- each of its classes, functions and keywords are covered in depth by the project's webpage. It is comprehensive-- it implements basic ELM algorithms, as well as a large subset of the proposed modified ELM-based regression methods. It is easy to install-- the package is distributed on Anaconda.org and is installable with one command. Finally, and critically, `ExtremeLearning` classes are fully serializable and compatible with Python's high-performance computing libraries. As a collection of many consistent implementations of modified ELM methods, `ExtremeLearning` allows data scientists and programmers to compare and contrast ELM methods without worrying about finding, installing, and verifying multiple separate ELM Python libraries.

# Overview of Methods

`ExtremeLearning` implements a collection of ELM-based regression and classification methods. The regressors use the original ELM algorithm, and the classifiers use a modification that produces probabilistic output. For each modified ELM regression method below, an analogous ELM classification method is implemented as well.  

1. **Extreme Learning Machine** ELM, proposed by Huang et. al., in 2004, is a randomly-initialized feed-forward neural network. A Moore-Penrose Generalized inverse (np.linalg.pinv) is used to fit its output layer against the target predictands, essentially solving a linear objective function. Models built with this method produce unbounded linear predictions. [@Huang:2004]
2. **Principal Components Transformation** PCT as part of an ELM approach implies that the predictor dataset has been reduced in dimensionality and transformed to an orthogonal space using Principal Components Analysis (sklearn.decomposition.PCA) [@Jinkwon:2007]
3. **Principal Components Initialization** Principal Components Initialization (otherwise known as PCA-ELM, proposed by Castaño et.al, in 2013) in an ELM indicates that the number of hidden layer neurons has been set according to the number of PCA Modes required to retain a certain percentage of the predictors' variability,  and that the weights of the hidden layer neurons are set to the coefficients of the PCA transformation. [@Castano:2013]
4. **Pruning** Pruning in an ELM methodology refers to the process, proposed by Rong et. al, in 2008, of identifying neurons in the hidden layer that are statistically irrelevant, and removing them in order to prevent overfitting. Here, the neurons are ranked and sorted by mutual information regression (sklearn.feature_selection.mutual_info_regression), and then the Akaike Information Criterion score for the output of each set of the first N neurons is calculated, and the set of the first N neurons with the minimum AIC score is kept. [@Rong:2008]
5. **Principal Components 'Pruning'** PCP in an ELM method means that PCA has been applied to the randomly initialized weight matrix- the weights have been transformed into an orthogonal space, and only the neurons required to capture a certain percentage of the variability of the initial weight matrix have been kept. 
6. **DropOut & DropConnect** DropOut and DropConnect, along with biased dropout and biased dropconnect, proposed for use in ELM by Gomes & Wang et. al. in 2020, refers to the process of randomly, or according to hidden layer activation magnitude or weight magnitude respectively, removing neurons and setting the weights of ELM neurons to zero in order to prevent over fitting. [@Gomes:2020] 
7. **Probabilistic Output ELM** POELM, which was proposed by Wong et. al in 2020, in essence, uses a sigmoid objective function instead of the linear one solved by ELM, and therefore must use modified linear programming in place of a Moore-Penrose Generalized Inverse to fit its output layer. [@Wong:2020] POELM is a multilabel classifier by default, but additional operations can be applied to its output to produce 'multiclass' classification probabilities, which sum to one. 

# Acknowledgements
Sincere thanks to the inventors and researchers that proposed and tested the methods implemented in `ExtremeLearning` 

# References
