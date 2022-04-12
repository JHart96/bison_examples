# BISoN: A Bayesian Framework for Inference of Social Networks - Examples

This repository is a collection of working examples for implementing the BISON framework with different types of social data. These examples show how to fit simple edge weight models, run basic diagnostic checks, and conduct high level animal social network analysis while propagating uncertainty through the analysis.

## Introduction

The BISoN framework is a Bayesian modelling framework for capturing uncertainty in inferred edge weights from observation data. The key idea behind the framework is to model the sampling process that generates observational data, and use that to quantify uncertainty around estimated edge weights between dyads. Importantly the framework allows for uncertainty around edges to be propagated to higher order metrics such as node centrality, and into subsequent analyses such as regression or non-random edge weight tests.

The preprint can be found at: [https://www.biorxiv.org/content/10.1101/2021.12.20.473541v1](https://www.biorxiv.org/content/10.1101/2021.12.20.473541v1).

## Examples

### Edge Weight Models
* [Binary Edge Weight Model (Stan)](examples/ewm_binary.md)
* [Count Edge Weight Model (Stan)](examples/ewm_count.md)
* [Duration Edge Weight Model (Stan)](examples/ewm_duration.md)
* [Binary Edge Weight Model (INLA)](examples/ewm_binary_inla.md)
* [Count Edge Weight Model (INLA)]()
* [Group Edge Weight Model (INLA)]()
* [Binary Directed Edge Weight Model (Stan)]()
* [Binary Edge Weight Model with Partial Pooling (Stan)]()


### Social Network Analysis
* [Dyadic Regression (Stan)](examples/dyadic_regression_stan.md)
* [Nodal Regression (Stan)]()
* [Non-random Edge Weight Tests (Stan)]()
* [Dyadic Regression (Metropolis)](examples/dyadic_regression_metropolis.md)
* [Nodal Regression (Metropolis)]()

### Data processing and diagnostics
* [Prior Predictive Checks]()
* [Group Data Conversion](examples/convert_gbi.md)

## R Package (In Development)

An R package (bisonR) is currently in development and will be available soon.
