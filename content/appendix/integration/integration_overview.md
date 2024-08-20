---
title: Appendix - Integration - Overview
subject: ML4EO
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
keywords: notation
---



**Example**
- 1D - Time Series
- 2D - Spatial Field
- 3D - Spatial Field
- 2D+T - Spatiotemporal Field

---
## **Exact Integration**

> These are the cases where we can find a closed-form expression of our integral.
> This usually stems from very simple cases, i.e., linear and Gaussian.

- Conditions - Linear & Gaussian
- Symbolic - Calculus Course
- Series Approximation
- Conjugate

## **Approximate Integration**

> This first section looks as many of the classical methods for approximating integrals like **Newton-Cotes**, **Quadrature**, **Bayesian Quadrature**, or **Monte-Carlo** methods.
> We will outline the methods

$$
\int f(x)w(x)dx=\sum_n f(x_n)w(x_n)
$$

- Newton-Cotes
	- $\int f(x)dx=\sum_n f(x_n)$
	- Locally Linear Interpolation between nodes
	- Nodes - Equidistant Node
	- Interpolating - 6-degree Polynomial 
	- e.g. Trapezoid - Linear, Simpsons - Quadratic
- Quadrature
	- $\int f(x)w(x)dx=\sum_n f(x_n)w(x_n)$
	- Nodes - User Defined
	- Interpolant - Roots of Orthogonal Polynomial
	- Polynomials - e.g., Hermite, Legendre, Chebychev, Laguerre
	- e.g., Gaussian
- Bayesian Quadrature
- Monte Carlo

#### **Uncertainty Propagation**

> This is an extension to numerical integration whereby we wish to integrate a quantity defined by a distribution.

$$
\int f(x)p(x)dx = \mathbb{E}_{x\sim p(x)}[f(x)]
$$

**Applications**
- Integration
- Dynamical Models
**Complexity**
- function
- prob distribution
- dimensionality
- integral method
**Methods**
- Exact - Linear, Gaussian
- Taylor - Linearized Function
- Unscented - Linearized Distribution
- Quadrature - Assumed Density 
- Bayesian Quadrature - Kernels , $\approx$ 10
	- GP Parameterization —> For Free, .e.g., Observations, 
	- Otherwise —> Function Approximation, e.g., expensive or black-box simulators
- Monte Carlo - Stochastic
- Markov Chain Monte Carlo

## **Applications**

- Convolution + Filtering
- Uncertainty Propagation
- Inference
- Sensitivity Analysis
- Bayesian Filtering-Smoothing
