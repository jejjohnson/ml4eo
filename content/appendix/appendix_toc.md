---
title: Appendix - TOC
subject: ML4EO
short_title: TOC
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

> In this section, we will host all of my notes for the deep dive sections.
> Each of these sections will have information that is relevant for the application tracks but they will be relatively agnostic to be used for general purposes.
> It can be thought of as the topics in a classic numerical analysis course but from a data-driven modeling perspective.

---
## Mini-TOC


**Numerical Analysis**

- Function Approximation
- Differentiation
- Integration
- Numerical Linear Algebra

**Parameterizations**

- Representation Learning
- State Space Models

**Learning**

- Optimization
- Inference
- Sensitivity Analysis

**Algorithms**

- Gaussian Processes
- State Space Models

**Programming**

- GeoData

---
## Numerical Analysis


---
### **Function Approximation**


:::{seealso} **Topics**
:class: dropdown

**Discretization**.
In this section, we will look at how we can go from unstructured data to structured data, i.e., discretization.
We will look at various methods for accomplishing this which includes

**Non-Parametric Regression**.
In this section, we will look at how we can use non-parametric functions to approximate underlying functions.
These include methods like **nearest-neighbour regression**, **radius-neighbour regression**, and **Gaussian processes**.

:::


:::{seealso} **Applications**
:class: dropdown

**Discretization**.

**Compression**.

**Interpolation**.

:::


---
### **Differentiation**

> Taking Derivatives is arguably the most import component in data-driven learning.
> It will serve as a foundation for all subsequent topics and application surrounding learning.
> Gradients in general are the workhorse of data-driven methods.
> In addition, thinking about how we parameterize our models often involve thinking about gradients which stem from classical numerical analysis.



:::{seealso} **Applications**
:class: dropdown

- ODEs, PDEs
- Sensitivity Analysis
- 1st Order Optimization - Gradient, Jacobian
- 2nd Order Optimization - Hessian
- ArgMin Differentiation - Unrolling, Implicit/Adjoint

:::


:::{seealso} **Topics**
:class: dropdown

**Operators**.
In this section, we will introduce some basics for differentiation by thinking of them as operators.
We will represent it as symbolic differentiation which will serve as a basis for the numerical approximations

**Automatic Differentiation**.

**Argmin Differentiation**.

**Approximation Differentiation**.
These include the core topics like finite difference, finite volume, finite element and stochastic.

:::

---
### **Integration**


:::{seealso} **Applications**
:class: dropdown

**Variational Inference**.
Variational inference will require instances of this.

**Dynamic Models**.
The `TimeStepper` component of a dynamical model will require numerical integration.

**Temporal Point Process**.
We will use a simple iteration of this when looking at extreme values.

:::


:::{seealso} **Topics**
:class: dropdown

**Exact Integration**.
These are the cases where we can find a closed-form expression of our integral.
This usually stems from very simple cases, i.e., linear and Gaussian.

**Numerical Integration**.
This first section looks as many of the classical methods for approximating integrals like **Newton-Cotes**, **Quadrature**, **Bayesian Quadrature**, or **Monte-Carlo** methods.
We will outline the methods where we focus on when each of them should be used as a question of data dimensionality, i.e., low dimensional, medium dimensional or high dimensional.

**Uncertainty Propagation**.
This is an extension to numerical integration whereby we wish to integrate a quantity defined by a distribution.


:::



---
### **Numerical Linear Algebra**


:::{seealso} **Topics**
:class: dropdown

**Reduced Order Matrices**


**Structured Matrices**.
These include **diagonal**, **tri-diagonal**, **block-diagonal**, **Kronecker**, and **triangular**.


**Linear Solvers**.
These include simple inversions with special matrix structures like **triangular** or **block-diagonal**.
There will also more more scalable options like iterative matrix inversion like the **conjugate gradient** methods.
There will also be more advanced topics like **preconditioning**.


**Log Determinants**

:::

---
## **Parameterizations**

---
## **Algorithms**

### **Gaussian Processess**

### **State Space Models**