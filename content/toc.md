---
title: Table of Contents
subject: ML4EO
short_title: Extended TOC
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


## Preface

## Table of Contents

**Observation Data**

**Abstractions**

**Problem Formulations**

**Discretizations**

**Interpolation**

**Non-Parametric Interpolators**

**Coordinate-Based Parametric Interpolators**

**Field-Based Parametric Interpolators**

**Parametric Dynamical Models**

**X-Casting**

**Operator Learning**



***
## Extended Table of Contents

### **Observation Data**
* Geoscience Data Anatomy - Spatiotemporal Field + Coordinates + Domain
* Data Types Whirlwind - L2, L3, L4, L5
* Missingness Characterization - MAR, MNAR, MCAR
* Case Study
	* Remote Sensing —> LST/SST
	* AlongTrack —> SSH


***
### **Abstractions**

- Geoscience Objectives - Estimation, Learning
- Geoscience Tasks - Interpolation, Extrapolation, Variable Transformation
	- Extrapolation - Data Drift, Distribution Shift, Bad Generalization
	- Interpolation - Missingness
	- Variable Transformation - Multivariate, High Correlation
- Operator Learning - A Space, Time, Quantity & Shape Perspective
- Quadrant of Things That Can Go Wrong - Measurements, Shape, Model, Solution Procedure
- An Abstraction of Designing Models for Learning - A Hierarchical Sequence of Decisions
- Data-Driven Model Elements w/ PGMs - Measurements, State, QoI, Latent Variable
- Bayesian Modeling - Data Likelihood, Prior, Posterior, Marginal Likelihood
- ML Algorithm Abstractions - DataModule, Model, Criteria, Optimizer, Learner
- Software - Hardware Agnostic Tensor Library, AutoDiff, Deep Learning, PPL

***
### **Problem Formulations**

> The overview can be viewed through the lens of *operator learning*.

$$
\boldsymbol{f}: 
\left\{ \boldsymbol{x}:\boldsymbol{\Omega}_x\times\mathcal{T}_x \right\}
\times
\boldsymbol{\Theta}\rightarrow
\left\{\boldsymbol{u}:\boldsymbol{\Omega}_u\times\mathcal{T}_u \right\}
$$

- Interpolation
- Extrapolation
- Variable Transformation

***
### **Discretization**

> To go from observations to models, we almost always need to have some sort of structure.
> We will look at the tried and true classic of the discretization methods: histogram binning.
> We will also look at some extra things we can do when creating histograms like defining specifying the binning from prior knowledge.
> We will also look at more adaptive binning methods for more irregular structures.

- Histogram Formulation (**TODO**)
- Equidistant Binning 4 Cartesian Grids (**TODO**)
- Adaptive Binning 4 Rectilinear & Curvilinear Grids (**TODO**)
- Graph-Node Binning (**TODO**)

***
### **Nonparametric Interpolation**

> In this section, we will look at some of the staple methods for nonparametric interpolation.
> We will outline each

- Naive Whirlwind Tour with applications for Data Assimilation
- Nearest Neighbours
	- K-NN
	- Weighted Distances
	- Scaling the Algorithm - KDTree + BallTree
	- Scaling the hardware - parallelization, GPU hardware
- Kernel Density Estimation
	- KDE 
	- FFT for Equidistant Grids
	- scaling the hardware - GPU hardware
	- Regression
- Gaussian Processes
	- Appendix: Playing with All things Gaussian
	- Spatial Autocorrelation with (Semi-)Variograms
	- 3 Views of GPs
	- GP with Numpyro
	- Scaling - Kernel Approximations
	- Scaling - Inducing Points
	- Scaling - 
	- Appendix GPs in practice
		- From Scratch
		- With TinyGP & GPJax
		- With PPL Numpyro
		- Customizing the Numpyro Implementation
		- Distances
		- Kernel Matrices
		- Kernel Matrix Derivatives
- Improved Gaussian Processes
	- Moment-Based
		- Sparse GPs
		- SVGPs
		- Structured GPs
		- Deep Kernel Learning
	- Basis Functions
		- Fourier Features GP
		- Spherical Harmonics GP
		- Sparse Spherical Harmonics GPs

***
### **Coordinate-Based Parametric Interpolator**

$$
\boldsymbol{f} : \mathbb{R}^{D_s}\times\mathbb{R}^+\times\mathbb{R}^{D_\theta}\rightarrow\mathbb{R}^{D_z}
$$

- Functa: A Physics-Informed Introduction
- Why naive MLPs don’t work - FF, SIREN
- Spatial Coordinate Encoders
- Temporal Coordinate Encoders
- PINNs
- How to Train your Functa
- Modulation
- Scaling
	- Hashing

***
### **Field-Based Parametric Interpolators**

$$
\boldsymbol{f} : \mathbb{R}^{D_\Omega}\times\mathbb{R}^+\times\mathbb{R}^{D_\theta}\rightarrow\mathbb{R}^{D_\Omega}
$$

- Interpolation Operator: A Physics-Informed Approach (Spatiotemporal Decomposition)
- Abstraction: Amortization vs Objective-Based
- Whirlwind Tour for 3 Architectures - CNNs, Transformers, Graphs
- Convolutions
	- Explaining Convolutions via Finite Differences
	- More on Convolutions - FOV, Separable, 
	- FFT Convolutions via Pseudospectral Methods
	- Missing Values & Masks
	- Partial Convolutions
- Transformers
	- Attention is All You Need
	- Transformers & Kernels
	- Missing Data - Masked Transformers
- Graphical Models
	- Graphs and Finite Element Methods
	- Missing Data
- Dimension Reduction
	- Dimensionality Reduction - What is it and why we need it? (SWM vs Linear SWM vs ROM)
	- AutoEncoders I - PCA/EOF/SVD/POD
	- AutoEncoders II - CNNs
	- AutoEncoders III - Transformers (MAE)
	- AutoEncoders IV - Graphs
- Multiscale
	- Introduction to Multiscale - Power Spectrum Approach
	- U-Net I - CNN
	- U-Net II - Transformers
	- U-Net III - Graphs
- Objective-Based Approaches
	- Implicit Models I - Fixed Point/Root Finding
	- Implicit Models II - Argmin Differentiation
	- Implicit Models III - Deep Equilibrium Models 
	- From Scratch
	- Packages - JaxOpt, optimistix
- Conditional Generative Models
	- Latent Variable Models
	- Bijective Flows
	- Stochastic Flows
	- Surjective Flows
	- Stochastic Interpolants

***
### **Parametric Dynamical Models**

$$
\boldsymbol{f} : \mathbb{R}^{D_\Omega}\times\mathbb{R}^+\times\mathbb{R}^{D_\theta}\rightarrow\mathbb{R}^{D_\Omega}
$$


- Operator Learning Revisited - Universal Differential Equations
- Whirlwind Tour - Spatial Operators
- Training
	- Experimental Setup - OSSE vs OSE
	- Online
	- Offline
- Spatial Operators Deep Dive
	- Linear Spatial Operator & MLP
	- Convolutions
	- FFT Convolutions
	- Spectral Convolutions
	- Transformers
	- Graphical Models
- Bayesian Filtering
	- State Space Models
	- Parameter & State Inference in SSMs
	- Linear Models + Exact Inference - KF
	- Non-Linear Model + "Exact" Inference - EKF, UKF, ADF
	- Whirlwind Tour of Deterministic Inference for SSMs
	- Amortized Variational Posteriors (Encoders)
	- Whirlwind Tour of Stochastic Inference for SSMs
- Nonparametric Revisited
    - Markovian Gaussian Processes
    - Sparse Markovian Gaussian Processes
- Latent Generative Dynamical Models
	- Latent State Space Models
	- Conjugate Transforms - Conditional Markov Flows
	- Stochastic Transform Filters
	- Observation Operator Encoders
	- Stochastic Differential Equations
	- Neural Stochastic Differential Equations