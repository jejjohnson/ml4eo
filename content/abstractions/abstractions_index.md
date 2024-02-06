---
title: Abstractions Overview
subject: Abstractions for Learning
short_title: Overview
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CSIC
      - UCM
      - IGEO
    orcid: 0000-0002-6739-0053
    email: juanjohn@ucm.es
license: CC-BY-4.0
keywords: data
---

## Geoscience Tasks

**Interpolation**. Within Convex Hull of points; Missingness

**Extrapolation**. X-Casting; Problems - Data Drift, Distribution Shift, Bad Generalization

**Variable Transformation**. Multivariate, High-Correlation, High-Dimensionality

***

## Operator Learning

> A Space, time, and quantity perspective

***

## Quadrant of Things That Can Go Wrong

* Measurements
* Domain Shape
* Model
* Solution Procedure


***
## Designing Models for Dynamical Systems

> A Hierarchical Sequence of Decisions ([Gharari et al, 2021](https://doi.org/10.1029/2020WR027948))

***
## Data-Driven Model Elements with PGMs

* Pieces - Observations, Covariates, Latent Variables, Quantity of Interest
* Directions - Directional, Bi-Directional, Non-Directional
* Independence

***
## Bayesian Modeling

* Data Likelihood, Prior, Marginal Likelihood
* Posterior, Prior Predictive Distribution, Posterior Predictive Distribution
* Variational Posterior

***
## ML Algorithm Abstractions

> How to read ML papers effectively.

* Data Module
* Model
* Criteria
* Optimizer
* (Learner)


***
## Software Stack

* Hardware Agnostic Tensor Library - `jax`
* AutoDifferentiation - `jax`
* Deep Learning Library - `keras`
* Probabilistic Programming Language - `numpyro`