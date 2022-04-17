# BISoN: A Bayesian Framework for Inference of Social Networks - Examples

This repository is a collection of working examples for implementing the BISON framework with different types of social data. These examples show how to fit simple edge weight models, run basic diagnostic checks, and conduct high level animal social network analysis while propagating uncertainty through the analysis.

## Introduction

The BISoN framework is a Bayesian modelling framework for capturing uncertainty in inferred edge weights from observation data. The key idea behind the framework is to model the sampling process that generates observational data, and use that to quantify uncertainty around estimated edge weights between dyads. Importantly the framework allows for uncertainty around edges to be propagated to higher order metrics such as node centrality, and into subsequent analyses such as regression or non-random edge weight tests.

The preprint can be found at: [https://www.biorxiv.org/content/10.1101/2021.12.20.473541v1](https://www.biorxiv.org/content/10.1101/2021.12.20.473541v1).

## Examples

This is a non-exhaustive list of some example BISoN models and auxiliary scripts. The examples are still in development and should be checked carefully before use in real analysis, but should be a good basis for beginning to work with BISoN. Please get in touch if you have any requests or spot any errors.

*Note that most of these examples depend on stochastic functions for various purposes, so the outputs might not always match exactly with the text of these notebooks.*

### Edge Weight Models
* [Binary Edge Weight Model (Stan)](examples/ewm_binary.md)
* [Count Edge Weight Model (Stan)](examples/ewm_count.md)
* [Duration Edge Weight Model (Stan)](examples/ewm_duration.md)
* [Binary Edge Weight Model (INLA)](examples/ewm_binary_inla.md)
* [Count Edge Weight Model (INLA)](examples/ewm_count_inla.md)
* [Group Edge Weight Model (INLA)](examples/ewm_group_inla.md)
* [Binary Edge Weight Model with Partial Pooling (Stan)](models/pooled_count_model.stan)

### Social Network Analysis
* [Dyadic Regression (Stan)](examples/dyadic_regression_stan.md)
* [Nodal Regression (Stan)](examples/nodal_regression_stan.md)
* ~~[Non-random Edge Weight Tests (Stan)]()~~
* [Dyadic Regression (Metropolis)](examples/dyadic_regression_metropolis.md)
* [Nodal Regression (Metropolis)](examples/nodal_regression_metropolis.md)

### Data processing and diagnostics
* ~~[Prior Predictive Checks]()~~
* [Group Data Conversion](examples/convert_gbi.md)

### Coming soon
* Binary Directed Edge Weight Model (Stan)
* Prior Predictive Checks
* Posterior Predictive Checks


## R Package (In Development)

An R package (bisonR) is currently in development and will be available soon. Follow [@jordan_hart_96](https://twitter.com/jordan_hart_96) on Twitter to keep up to date with the latest developments.

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/jordan_hart_96.svg?style=social&label=Follow%20%40jordan_hart_96)](https://twitter.com/jordan_hart_96)

