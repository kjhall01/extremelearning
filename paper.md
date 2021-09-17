---
bibliography: paper.bib
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
 - name: International Research Institute for Climate & Society
   index: 1
 - name: Center for Earth System Modeling, Analysis, & Data (ESMAD), Department of Meteorology and Atmospheric Science, The Pennsylvania State University
   index: 2
date: 17 September 2021

---

# Summary

An "Extreme Learning Machine" (ELM) is a fast, non-linear regression model, which fits the output layer of a randomly-initialized feed forward neural network using a Moore-Penrose Generalized Inverse [@Huang:2020]. Since it was published in the early 2000s, numerous related methods have been proposed. Each new method slightly changes the underlying ELM Model, either by pruning nodes in the hidden layer, applying Principal Components Analysis to the training data, modifying the hidden layer weight initialization scheme, changing the objective function, or some other new method. `ExtremeLearning` is a fully open-source python package that implements these new Regressors and Classifiers based on ELM in the style of SciKit-Learn, in a way that is compatible with python parallelism and cluster computing .

# Statement of need

While there are some implementations of ELM available in Python, many of the proposed modifications to ELM are only implemented by their original authors in R, Stata, or Matlab, or are not available to the public. The ones that are available in Python are generally disparate, difficult to find, and implement inconsistent APIs. This requires that the user spend time collating, installing, and learning each one, which creates a significant barrier to productivity. Additionally, some implementations of ELM in Python rely on compiled code in other languages to enhance performance, which makes them non-serializable and difficult to scale with distributed computing. `ExtremeLearning` implements a selection of ELM approaches in pure python, with an API designed to mirror those of the commonly used SciKit-Learn classifiers and regressors to boost synergy with the larger Python data science ecosystem. Being pure python, `ExtremeLearning` models are fully serializable and can easily be used with python parallelism and distributed computing libraries. `ExtremeLearning` is distributed through Anaconda.org, and is extremely easy to install and use. Altogether, `ExtremeLearning` removes barriers to the research and development of ELM-based machine learning solutions in python.

# Overview of Methods

Each method here is implemented as both a regressor, using the standard 'ELM' generalized Moore-Penrose inverse approach, and as a classifier, which implements the 'Probabilistic Output' ELM  (POELM) approach.  POELM substitutes the Moore-Penrose inverse and linear objective function with a sigmoid objective function and specialized linear programming.

1. Basic Extreme Learning Machine (ELM) / Probabilistic Output Extreme Learning Machine (POELM)
2. ELM/POELM with Principal Components (PCA) transformation applied to predictors
3. ELM/POELM with weights initialized to coefficients of a PCA transformation fit on the predictors (PCA-ELM)
4. ELM/POELM with PCA transformation applied to the randomly initialized hidden layer weights
5. ELM/POELM with hidden layer node pruning according the statistical relevance
6. ELM/POELM with biased or unbiased dropout of neurons and 'dropconnect' of neuron weights 

`ExtremeLearning` also allows the user to mix and match these features, to create a model that best fits their needs. 

# Acknowledgements
Sincere thanks to the inventors and researchers that proposed and tested the methods available in `ExtremeLearning` 

# References
