# BISoN: A Bayesian Framework for Inference of Social Networks - Examples

This repository is a collection of working examples for implementing the BISON framework with different types of social data. These examples show how to fit simple edge weight models, run basic diagnostic checks, and conduct high level animal social network analysis while propagating uncertainty through the analysis.

## Introduction

The BISoN framework is a Bayesian modelling framework for capturing uncertainty in inferred edge weights from observation data. The key idea behind the framework is to model the sampling process that generates observational data, and use that to quantify uncertainty around estimated edge weights between dyads. Importantly the framework allows for uncertainty around edges to be propagated to higher order metrics such as node centrality, and into subsequent analyses such as regression or non-random edge weight tests.

The preprint can be found at: [https://www.biorxiv.org/content/10.1101/2021.12.20.473541v1](https://www.biorxiv.org/content/10.1101/2021.12.20.473541v1).

## Examples
* [Binary Data](examples/binary_data.md)
* [Count Data](examples/count_data.md)
* [Duration Data](examples/duration_data.md)
* [Gambit of the Group Data Conversion](examples/convert_gbi.md)
* [Partial Pooling](models/pooled_count_model.stan)

## R Package (In Development)

An R package (bisonR) is currently in development and will be available soon.
