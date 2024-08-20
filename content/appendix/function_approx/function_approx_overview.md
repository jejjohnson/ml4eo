---
title: Function Approximation
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



---
## **Function Approximation**

#### **Discretization - Non-Parametric**
> Can I go from unstructured data to structured data.

**Unstructured —> Regular**
- Histogram
- k-Nearest Neighbours
- Radius Neighbours
- Kernel Density Estimation
**Unstructured —> Irregular**
- Voronoi
- Gaussian Mixture Model
- K-Means
- HDBScan
**Scale**
- Parallelization
- Algorithm - Ball-Tree, KD-Tree, R-Tree - [Overview](https://www.geeksforgeeks.org/ball-tree-and-kd-tree-algorithms/)
- Hardware - CUDA/GPU

#### **Regression - Non-Parametric**
- Nearest Neighbour Regression
- Radius Neighbour Regression
- Gaussian Process

#### **Regression - Parametric**
- Linear
- Basis Function - FFT, Splines, RBF
- Neural Fields

#### **Spatial Encoders**
- Splines
- Trigonometry
- Spherical Harmonics
- Scaling, e.g., Hashing

#### **Temporal Encoders**
- Linear
- Bounded
- Exponential
- Fourier Features, e.g., Sinusoidal Embedding

---
#### Neural Fields
- Why MLPs don’t work
- Parameterizations - FF, SIREN, MFN
- Connections
	- 1 Layer - GP, RFF
	- Multiple Layers - Deep GPs, Random Feature Expansions
- Modulation, aka, HyperNetworks
- Uncertainty
- Physics-Informed Loss Function
- Scaling

Examples
- Ocean Land Mask - Discrete
- Orography - Continuous



Examples
- Spatial Encoders
- Discretization

