# bayes_drt2

`bayes_drt2` is a re-implementation of [`bayes_drt`](https://github.com/jdhuang-csm/bayes-drt) using the `cmdstan` interface to Stan instead of `pystan`. This addresses some issues with installation and the removal of optimization functionality in current versions of `pystan`. The functionality and structure of `bayes_drt2` is virtually identical to that of `bayes_drt`. If you are using `bayes_drt` without issue, there is no reason to switch to `bayes_drt`. See the [tutorials for `bayes_drt`](https://github.com/jdhuang-csm/bayes-drt/tree/master/tutorials) for tutorials on how to use the package.

`bayes_drt2` is a Python package for inverting  electrochemical impedance spectroscopy (EIS) data to obtain the distribution of relaxation times (DRT) and/or distribution of diffusion times (DDT).

`bayes_drt2` implements a hierarchical Bayesian model to provide well-calibrated estimates of the DRT or DDT without ad-hoc tuning. The package offers two methods for solving the model:
* Hamiltonian Monte Carlo (HMC) sampling to estimate the posterior distribution, providing both a point estimate of the distribution and a credible interval
* L-BFGS optimization to maximize the posterior probability, providing a maximum *a posteriori* (MAP) point estimate of the distribution

It is also possible to perform multi-distribution inversions, e.g. to simultaneously fit both a DRT and a DDT, with these methods. This is an experimental feature and requires some manual tuning. See the tutorials for an example.

The package also provides ordinary and hyperparametric ridge regression methods, which may be useful for comparison or for obtaining initial estimates of the distribution. The hyperparametric ridge regression method is an implementation of the method developed by Ciucci and Chen (https://doi.org/10.1016/j.electacta.2015.03.123) and expanded by Effat and Ciucci (https://doi.org/10.1016/j.electacta.2017.07.050).

## *Electrochimica Acta* article
The methods implemented in `bayes_drt2` are the subject of an article in *Electrochimica Acta* (https://doi.org/10.1016/j.electacta.2020.137493). The theory behind the model is described in detail in the journal article.

## Installation
See the installation.txt file for installation instructions.

### Dependencies
`bayes_drt2` requires:
* numpy
* scipy
* matplotlib
* pandas
* cvxopt
* cmdstan

## Issues?
If you run into any issues using the package, please feel free to raise an issue, and I will do my best to help you solve it. Additionally, if you would like to apply the method for more complex analyses, please reach out - I would be happy to help get an appropriate model set up for your use case. 

## Citing `bayes_drt2`
If you use `bayes_drt2` for published work, please consider citing the following paper:
* Huang, J., Papac, M., and O'Hayre, R. (2020). Towards robust autonomous impedance spectroscopy analysis: a calibrated hierarchical Bayesian approach for electrochemical impedance spectroscopy (EIS) inversion. *Electrochimica Acta, 367,* 137493. https://doi.org/10.1016/j.electacta.2020.137493

Additionally, if you use the `ridge_fit` method with `hyper_lambda=True` or `hyper_w=True`, please cite the corresponding work below:
* `hyper_lambda=True`: Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. *Electrochimica Acta, 167,* 439–454. https://doi.org/10.1016/j.electacta.2015.03.123
* `hyper_w=True`: Effat, M. B., & Ciucci, F. (2017). Bayesian and Hierarchical Bayesian Based Regularization for Deconvolving the Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy Data. *Electrochimica Acta, 247,* 1117–1129. https://doi.org/10.1016/J.ELECTACTA.2017.07.050
