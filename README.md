<!--
*** This README comes from here: https://github.com/othneildrew/Best-README-Template/edit/master/BLANK_README.md - thanks ! 
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]




<!-- PROJECT LOGO -->
<p align="center">
  <h3 align="center">ExtremeLearning: A Selection of ELM-based Approaches </h3>
  
  Extreme Learning is a python module that serves as a centralized collection of implementations of various "Extreme Learning Machine" machine learning approaches. Some basic ELM approaches are available in Python, but it can be difficult to find the right tool for the job. The cutting-edge ELM approaches are even harder to find, if they're even available. Extreme Learning implements some of these newer, more interesting ELM approaches, so users can get right to the science, rather than waste their time trying to implement ELM. 
    <br />
    <a href="#documentation"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kjhall01/extremelearning/blob/main/demo.ipynb">View Demo</a>
    ·
    <a href="https://github.com/kjhall01/extremelearning/issues">Report Bug</a>
    ·
    <a href="https://github.com/kjhall01/extremelearning/issues">Request Feature</a>
    </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- Why XCast -->
## About

Extreme Learning implements 'Extreme Learning Machine' models of various types in the style of Scikit-Learn, to make them easy to use. It is built on top of NumPy, mostly, with some SciKit-Learn and SciPy under the hood as well. ELM is was so named by Huang et. al.; in a nutshell the model is a randomly initialized feed-forward neural network, whose output layer is fit with a generalized Moore-Penrose inverse (basically Ordinary Least Squares). The different flavors of ELM do different things with Initialization, Principal Components Analysis, DropOut, and Pruning, and are detailed in the documentation section. ELM classifiers are implemented using the POELM approach, which involves replacing ELM's linear objective function with a sigmoid function, and modifying the output layer's 'fitting' process. 

This package was inspired by the need for an ELM library that: 
1. Encompasses a broad range of ELM approaches 
2. Provides a consistent API with SciKit-Learn 
3. Is 'Picklable' - and therefore, parallelizable and compatible with XCast
4. Is easy to install and quick to learn




<!-- GETTING STARTED -->
## Getting Started

1. Install with [Anaconda](https://anaconda.org/)
   ```sh
   conda install -c hallkjc01 extremelearning
   ```
2. Read the [Documentation](https://github.com/kjhall01/extremelearning/)
3. Run the [tests](https://github.com/kjhall01/extremelearning/blob/main/test.py): 
   ```sh
   python test.py
   ```


## Documentation 

ExtremeLearning implements twelve (12) ELM approaches: 6 regressors, and 6 classifiers. All of them implement the same API as SciKit-Learn's classifiers and regressors. Classifiers are implemented using the POELM approach. 

### Regressors 

1. **ELMRegressor**
A basic ELM regression implementation  (ELM)
```
import extremelearning as el
elm = el.ELMRegressor(hidden_layer_size=500, activation='sigm', dropconnect_pr=0.5, dropout_pr=0.5, dropconnect_bias_pctl=0.9, dropout_bias_pctl=0.9)
elm.fit(X, y)
res = elm.predict(X)
```

2. **PCTRegressor**
ELM, with input data transformed by Principal Components Analysis (ELM-PCA)
```
import extremelearning as el
pct = el.PCTRegressor(hidden_layer_size=500, retained=None, activation='sigm') #retained can be (0, 1) percent variation or an integer number of PCA modes to retain
pct.fit(X, y)
res = pct.predict(X)
```

3. **PCIRegressor** 
ELM, with hidden_layer_size and weights initialized according to PCA-ELM
```
import extremelearning as el
pci = el.PCIRegressor(retained=None, activation='sigm') #retained can be (0, 1) percent variation or an integer number of PCA modes to retain
pci.fit(X, y)
res = pci.predict(X)
```

4. **PCPRegressor**
ELM, with initial neuron weights transformed by PCA so that the minimum number of neurons necessary to retain X% of variance of initial neuron weights is used.
```
import extremelearning as el
pcp = el.PCPRegressor( hidden_layer_size=500, n_components=0.7, activation='sigm') #N_components can be (0, 1) percent variation or an integer number of PCA modes to retain
pcp.fit(X, y)
res = pcp.predict(X)
```

5. **PrunedRegressor** 
ELM, with pruning of 'irrelevant' neurons, according Chi-Squared and AIC minimization
```
import extremelearning as el
prune = el.PrunedRegressor( hidden_layer_size=500, activation='sigm') 
prune.fit(X, y)
res = prune.predict(X)
```

6. **DropRegressor** 
ELM with 'Dropout' and 'DropConnection'- randomized elimination of neurons and weights
```
import extremelearning as el
drop = el.DropRegressor(hidden_layer_size=500, activation='sigm', dropconnect_pr=0.5, dropout_pr=0.5, dropconnect_bias_pctl=0.9, dropout_bias_pctl=0.9) 
drop.fit(X, y)
res = drop.predict(X)
```

### Classifiers
1. **ELMClassifier**
A  Probabilistic-Output ELM classification implementation  (POELM)
```
import extremelearning as el
elm = el.ELMClassifier(hidden_layer_size=5, activation='sigm')
elm.fit(X, y)
res = elm.predict(X)
probs = elm.predict_proba(X)
```

2. **PCTClassifier**
POELM, with input data transformed by Principal Components Analysis (ELM-PCA)
```
import extremelearning as el
pct = el.PCTClassifier(hidden_layer_size=500, retained=None, activation='sigm') #retained can be (0, 1) percent variation or an integer number of PCA modes to retain
pct.fit(X, y)
res = pct.predict(X)
probs = pct.predict_proba(X)
```

3. **PCIClassifier** 
POELM, with hidden_layer_size and weights initialized according to PCA-ELM
```
import extremelearning as el
pci = el.PCIClassifier(retained=None, activation='sigm') #retained can be (0, 1) percent variation or an integer number of PCA modes to retain
pci.fit(X, y)
res = pci.predict(X)
probs = pci.predict_proba(X)
```

4. **PCPClassifier**
POELM, with initial neuron weights transformed by PCA so that the minimum number of neurons necessary to retain X% of variance of initial neuron weights is used.
```
import extremelearning as el
pcp = el.PCPClassifier( hidden_layer_size=500, n_components=0.7, activation='sigm') #N_components can be (0, 1) percent variation or an integer number of PCA modes to retain
pcp.fit(X, y)
res = pcp.predict(X)
probs = pcp.predict_proba(X)
```

5. **PrunedClassifier** 
POELM, with pruning of 'irrelevant' neurons, according Chi-Squared and AIC minimization
```
import extremelearning as el
prune = el.PrunedClassifier( hidden_layer_size=500, activation='sigm') 
prune.fit(X, y)
res = prune.predict(X)
probs = prune.predict_proba(X)
```

6. **DropClassifier** 
POELM with 'Dropout' and 'DropConnection'- randomized elimination of neurons and weights
```
import extremelearning as el
drop = el.DropClassifier(hidden_layer_size=500, activation='sigm', dropconnect_pr=0.5, dropout_pr=0.5, dropconnect_bias_pctl=0.9, dropout_bias_pctl=0.9) 
drop.fit(X, y)
res = drop.predict(X)
probs = drop.predict_proba(X)
```

### Methods 
**Regressor Methods** 
1. ```.fit(x, y)``` - fit the model to training data, where ```x``` is of shape (n_samples, n_features), and ```y``` is of shape (n_samples, 1) 
2. ```.predict(x)``` - make predictions on using a previously trained model, where ```x``` is of shape (n_samples, n_features) and has the same number of features as the original training set. 

**Classifier Methods**
1. ```.fit(x, y)``` - fit the model to training data, where ```x``` is of shape (n_samples, n_features), and ```y``` is of shape (n_samples, n_classes) and is one-hot encoded.  
2. ```.predict(x)``` - make class predictions on using a previously trained model, where ```x``` is of shape (n_samples, n_features) and has the same number of features as the original training set. Output is not one-hot encoded. 
3. ```.predict_proba(x)``` - make probabilistic class predictions using a previously trained model, where ```x``` is of shape (n_samples, n_features) and has the same number of features as the original training set. Output is of shape (n_samples, n_classes) and represents a predicted probability for each class, that will sum to 1.0. 



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Email: kjhall@iri.columbia.edu (This is a side project, so it may take a while to get back to you)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [ELM](https://ieeexplore.ieee.org/document/1380068)
* [PCA-ELM](https://link.springer.com/content/pdf/10.1007/s11063-012-9253-x.pdf)
* [ELM-PCA](https://ieeexplore.ieee.org/document/1380068)
* [Probabilistic Output ELM](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8978896)
* [Biased DropConnect-Based Extreme Learning Machine](https://www.hindawi.com/journals/mpe/2020/3604579/)
* [Drop ELM](https://www.hindawi.com/journals/mpe/2020/3604579/)
* [Pruned ELM](https://www.sciencedirect.com/science/article/pii/S0925231208000738?via%3Dihub)
* [Optimally Pruned ELM](https://ieeexplore.ieee.org/document/5350449)
* [SciKit-Learn](https://scikit-learn.org/stable/)
* [HPELM](https://hpelm.readthedocs.io/en/latest/)
* This README template comes from [here](https://github.com/othneildrew/Best-README-Template/edit/master/BLANK_README.md) - thank you!

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/kjhall01/extremelearning.svg?style=for-the-badge
[contributors-url]: https://github.com/kjhall01/extremelearning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kjhall01/extremelearning.svg?style=for-the-badge
[forks-url]: https://github.com/kjhall01/extremelearning/network/members
[stars-shield]: https://img.shields.io/github/stars/kjhall01/extremelearning.svg?style=for-the-badge
[stars-url]: https://github.com/kjhall01/extremelearning/stargazers
[issues-shield]: https://img.shields.io/github/issues/kjhall01/extremelearning.svg?style=for-the-badge
[issues-url]: https://github.com/kjhall01/extremelearning/issues
[license-shield]: https://img.shields.io/github/license/kjhall01/extremelearning.svg?style=for-the-badge
[license-url]: https://github.com/kjhall01/extremelearning/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kjhall01
