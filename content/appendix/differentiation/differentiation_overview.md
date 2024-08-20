---
title: Appendix - Differentiation - Overview
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
#### **Operators**

> In this section, we will introduce some basics for differentiation by thinking of them as operators.
> We will represent it as symbolic differentiation which will serve as a basis for the numerical approximations

* Difference
* Gradient, Jacobian, Hessian
* Divergence, Curl, Laplacian
* Fused Operators
	* Gradient + Divergence —> Laplace
	* Jacobian + Trace —> Divergence
	* Vector + Jacobian Product


#### **Automatic Differentiation**

```python
# gradient operations
dudx = grad(f)
u_grad = gradient(f)
u_jac = jacobian(f)
u_hess = hessian(f)
```

- Back propagation

**Argmin Differentiation**

> Argmin differentiation ([slides](https://mblondel.org/talks/mblondel-ulisboa-2021-06.pdf)) is a 

- Unrolling
- Implicit Differentiation, Adjoint

#### **Approximate Differentiation**

> We will introduce each of the methods where we will stress that they are ultimately a decision defined by the underlying discretization.

- Finite Difference
- Finite Volume
- Finite Element
- Stochastic

#### **Applications**

- ODEs, PDEs
- Sensitivity Analysis
- 1st Order Optimization - Gradient, Jacobian
- 2nd Order Optimization - Hessian
- ArgMin Differentiation - Unrolling, Implicit/Adjoint