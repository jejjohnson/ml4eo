---
title: Numerical Linear Algebra
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


Bla

> Matrix Inversions (Approximate Matrices) and Log Determinant (Jacobians)


- Reduced Order Matrix
	- Gram, Covariance
	- Data Independent - Fourier Features
	- Data Dependent - Nyström
- Sparse Matrix
- Structured Matrix - Diagonal, TriDiagonal, Kronecker, Block Diagonal, Triangular
- Log Determinant - Stochastic, Hutchins
- SVD/PCA/EOFs/POD
	- Randomized Sketching + Deterministic Post-Processing
	- Order of Matrix Multiplications
- Vector Jacobian Products
- Linear Solve
	- Cholesky
	- Iterative - Conjugate Gradient

**Applications**
- randomized SVD
	- Gap-Filling with EOFs
	- Eigenmaps - Laplacian, Schrodinger, Locality Projections
- Log Determinants - GPs, Fokker-Planck PDE
- Vector Jacobian Products - 3DVar, 4DVar 
- Linear Solvers
	- Regression - Linear, Ridge, Lasso, Elastic Net, Kernel, Fourier Features, Nystrom
	- State Estimation - DeNoise, DeBlurr, Gap Filling (SVD), 3DVarNet, SC 4DVarNet
	- Poisson Solver - Periodic, Dirichlet, Neumann, Staggered Grid

