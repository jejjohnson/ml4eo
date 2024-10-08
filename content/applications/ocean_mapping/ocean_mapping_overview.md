---
title: Modeling Ocean Surface Variables
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


## Overview

In this application, we are interested in modeling surface observations on the ocean.
These observations include temperature, height, salinity, and colour.
Contrary to the previous application for extremes, here we will focus more on different parameterizations which we can learn based on different datasets which are available.

*** 
### Quick Walk-Through

***
#### **Datasets**.
The first set of datasets will feature a range of different

$$
\begin{aligned}
\text{Multivate Spatiotemporal Series}: && &&
\mathcal{D} &= \left\{(t_n, \mathbf{s}_n), \mathbf{y}_t \right\}_{n=1}^{N_T} && &&
\mathbf{y}_n\in\mathbb{R}^{D_y} \\
\text{Coupled Multivate Spatiotemporal Series}: && &&
\mathcal{D} &= 
\left\{(t_n, \mathbf{s}_n), \mathbf{x}_n, \mathbf{y}_t \right\}_{n=1}^{N_T} 
&& &&
\mathbf{y}_n\in\mathbb{R}^{D_y} \\
\end{aligned}
$$


**Gaussian Processes**.
Even when we don't have any data, we can still train parameterizations based on random fields with structures.
This is where Gaussian processes come into play.

**Simulations**.
If we are lucky enough to have access to simulations, this will enable us to train really good priors.

**L1 Raw Observations**.
Sometimes we are lucky enough to have high quality raw observation with good covereage and minimal noise.
This is possible with some satellite observations like polar-orbiting or geostationary satellites.
However, this is rarely the cases with derived variables like sea surface height or sea surface temperature.

**L2-L3 Clean Observations**.
These are typically cleaned observations where we have some derived products.

**L3 Interpolated Harmonized Observations**.

**L4 Reanalysis**.
The final dataset is the reanalysis data which is a combination of physical models and measurements.
This is arguably the highest quality dataset we can hope to have for learning models.


***
#### **Sub-Topics**

**Learning Cycle**.
We will implement the recursive learning cycle: Data, Learning, Estimation, & Predictions.
This cycle will be reimplemented with every subsequent new dataset we acquire with better and better quality.

**Parameterization Deep Dive**.
We will look at different parameterizations.
This will include linear, basis functions, non-linear, and neural networks.
In addition, we will look at how we can use structure to our advantage to make learning easier.



**Game of Dependencies**.
Many people do not have a solid grasp about dependencies within spatiotemporal data.
We will walk-through this step-by-step by slowly introducing more complex representations, e.g., IID, time conditioning, dynamical models, and state-space models.

**Extreme Values**.
Extreme values are a worlds apart from the standard mean and standard deviation.
In this section, we will introduce this from fir
We will introduce this from the perspective of Temporal Point Processes (TPPs).
Some staple distributions will include the Poisson Process, the GEVD, and the GPD.

**Marked Temporal Point Processes**.
The end-goal for extreme values (and many schemes in general) is to make predictions, i.e., what is the probability that there will be an extreme event, $\lambda$, with a certain intensity, $z$, at a certain location, $\mathbf{s}$, or time, $t$, given historical observations, $\mathcal{H}$, and some covariate parameter, $x$.


***
#### **Useful Skills**

**Parameter Estimation**.
We will dig deep into the data-driven learning playbook.

**State Estimation**.
We will be looking at the greats from the field of data assimilation.

**Transfer Learning**.
We will be reusing components as we repeat the same process over and over again.

***
### Cyclic

1. Gather Data
2. Learn Prior via Parameter Estimation
3. Estimate State via Data Assimilation
4. Make Predictions

***
#### Data Cycle

> Below is the data cycle that we wish to use for learning priors.

* Gaussian Processes
* Simulation Data
* Clean Observations
* Interpolated Observations
* Reanalysis Data

***
#### Module Cycle

* Parameter Estimation
* State Estimation
* Predictions

***
#### Complexity

* Coordinate-Based Interpolation
* Field-Based Interpolation
* Strong-Constrained Dynamical Priors
* Weak-Constrained Dynamical Priors


***
## Blogs

### Part I

In this part, we will investigate **Discretization Strategies**.

***
**Data Download + EDA**
* Data - Sea Surface Height, Sea Surface Temperature, Ocean Colour
* EDA - Scatter, Histogram, Statistics

***
**Metrics**
* Real Space - RMSE, R2, Correlation
* Spectral Space - Isotropic PSD, Spatiotemporal PSD, Wavelet PSD

***
**Discretization I - Histogram**
* Histogram Interpolation

***
**Discretization II - Neighbours**
* KNNs 
* [RAPIDS](https://docs.rapids.ai/api/cuml/stable/api/#id33)


***
### Part II

> Here, we investigate non-parametric interpolation strategies.
> In other words, we are looking at coordinate-based interpolation strategies.

***
**Non-Parametric Interpolation I - Nearest Neighbours**
* KNN + GPUS
* Weights

***
**Non-Parametric Interpolation II - Radius Neighbours**
* Weights

***
**Scaling - I**
* KNN + GPUs - [RAPIDS](https://docs.rapids.ai/api/cuml/stable/api/#nearest-neighbors)

***
**Non-Parametric Interpolation III - KDE**
* KDE

***
**Scaling**
* FFT + Equidistant Grids
* Patching
* GPU Hardware - [RAPIDS](https://docs.rapids.ai/api/cuml/stable/api/#kernel-density-estimation)

***
**Non-Parametric Interpolation IV - Gaussian Processes**
* Naive Gaussian Processes


**Scaling I - GPUs**
* 

**Scaling II - Inducing Points**


**Scaling III -**


***
### Part III

> We are looking at parametric coordinate-based interpolation strategies.
> This will involve neural fields.

***
**Non-Parametric Interpolation V - Neural Fields**

* Baseline -> RBFSampler/Nystrom + Linear SGD
* Fourier Features vs GPs
* Deep Approximate Gaussian Processes vs Deep Neural Networks.


**Linear**
* Process Convolutions with Gaps

***
**Neural Fields**
* Background - Deep Dive
* Algorithms - MLP, Fourier Features, SIREN


***
### Part IV

> We look at field-base interpolation strategies.

***
### Part V

> We look at dynamical model-based interpolation strategies.