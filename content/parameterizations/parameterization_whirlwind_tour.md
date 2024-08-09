---
title: Parameterization
subject: ML4EO
short_title: Whirlwind Tour
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

> Overview

* Independent Realization
* Time
* Space
* Variable

***
## **Unconditional Density Estimation**


***
### **IID Data**

$$
\mathcal{D} = \{ \mathbf{y}_n\}_{n=1}^N
$$

**Examples**:
* Ensembles
* Batches
* Patches


#### **Fully Pooled Model** (Temperature, Precipitation)

$$
p(\mathbf{Y},\boldsymbol{\theta}) = p(\boldsymbol{\theta})\prod_{n=1}^N
p(\mathbf{y}_n|\boldsymbol{\theta})
$$

#### **Non-Pooled Model**

$$
p(\mathbf{Y},\boldsymbol{\theta}) = \prod_{n=1}^N
p(\mathbf{y}_n|\boldsymbol{\theta}_n)
$$


#### **Partially Pooled Model**

$$
p(\mathbf{Y},\mathbf{Z},\boldsymbol{\theta}) = p(\boldsymbol{\theta})\prod_{n=1}^N
p(\mathbf{y}_n|\mathbf{z}_n)p(\mathbf{z}_n|\boldsymbol{\theta})
$$

***
### **Time Series Data**

$$
\mathcal{D} = \{ t_n, \mathbf{y}_n\}_{n=1}^N
$$

#### **Temporally Conditioned Model**


$$
p(\mathbf{Y},\mathbf{t},\mathbf{Z},\boldsymbol{\theta}) = 
p(\boldsymbol{\theta})
\prod_{n=1}^{N_T}
p(\mathbf{y}_n|\mathbf{z}_n)p(\mathbf{z}_n|t_n,\boldsymbol{\theta})
$$

#### **Dynamical Model**

$$
p(\mathbf{Y},\mathbf{Z},\boldsymbol{\theta}) = 
p(\boldsymbol{\theta})
p(\mathbf{z}_0|\boldsymbol{\theta})
\prod_{t=1}^T
p(\mathbf{y}_t|\mathbf{z}_t)p(\mathbf{z}_t|\mathbf{z}_{t-1},\boldsymbol{\theta})
$$


***
### **Spatial Field Data**

$$
\mathcal{D} = \{\mathbf{s}_m,\mathbf{y}_m\}_{m=1}^{N_\Omega}
$$

#### **Spatially Conditioned Model**


$$
p(\mathbf{Y},\mathbf{S},\mathbf{Z},\boldsymbol{\theta}) = p(\boldsymbol{\theta})\prod_{n=1}^{N_T}
p(\mathbf{y}_n|\mathbf{z}_n)p(\mathbf{z}_n|\mathbf{s}_n,\boldsymbol{\theta})
$$


***
### **Spatio-Temporal Data**

$$
\mathcal{D} = \{t_n, \mathbf{s}_m,\mathbf{y}_{nm}\}_{n=1,m=1}^{N_T,N_\Omega}
$$

#### **Spatiotemporal Conditioned Model**


$$
p(\mathbf{Y},\mathbf{t},\mathbf{S},\mathbf{Z},\boldsymbol{\theta}) = 
p(\boldsymbol{\theta})
\prod_{n=1}^{N_T}
\prod_{m=1}^{N_\Omega}
p(\mathbf{y}_{nm}|\mathbf{z}_{nm})p(\mathbf{z}_{nm}|t_n, \mathbf{s}_{m},\boldsymbol{\theta})
$$

#### **Dynamical Model**

$$
p(\mathbf{Y},\mathbf{Z},\boldsymbol{\theta}) = 
p(\boldsymbol{\theta})
p(\mathbf{z}_0|\boldsymbol{\theta})
\prod_{t=1}^T
p(\mathbf{y}_t|\mathbf{z}_t)p(\mathbf{z}_t|\mathbf{z}_{t-1},\boldsymbol{\theta})
$$



***
## **Conditional Density Estimation**


***
### **IID Data**

$$
\mathcal{D} = \{ \mathbf{x}_n,\mathbf{y}_n\}_{n=1}^N
$$


#### **Non-Pooled Model**

$$
p(\mathbf{Y},\mathbf{X},\mathbf{Z},\boldsymbol{\theta}) = 
\prod_{n=1}^N
p(\mathbf{y}_n|\mathbf{z}_n)p(\mathbf{z}_n|\mathbf{x}_n,\boldsymbol{\theta}_n)
$$


#### **Partially Pooled Model**

$$
p(\mathbf{Y},\mathbf{X},\mathbf{Z},\boldsymbol{\theta}) = p(\boldsymbol{\theta})\prod_{n=1}^N
p(\mathbf{y}_n|\mathbf{z}_n)p(\mathbf{z}_n|\mathbf{x}_n,\boldsymbol{\theta})
$$

***
### **Time Series Data**

$$
\mathcal{D} = \{ t_n, \mathbf{x}_n, \mathbf{y}_n\}_{n=1}^N
$$

#### **Temporally Conditioned Model**


$$
p(\mathbf{Y},\mathbf{t},\mathbf{X},\mathbf{Z},\boldsymbol{\theta}) = 
p(\boldsymbol{\theta})
\prod_{n=1}^{N_T} 
p(\mathbf{y}_{n}|\mathbf{z}_n)p(\mathbf{z}_n|t_n,\mathbf{x}_n,\boldsymbol{\theta})
$$

#### **Dynamical Model**

$$
p(\mathbf{Y},\mathbf{X},\mathbf{Z},\boldsymbol{\theta}) = 
p(\boldsymbol{\theta})
p(\mathbf{z}_0|\boldsymbol{\theta})
\prod_{t=1}^T
p(\mathbf{y}_t|\mathbf{z}_t)p(\mathbf{z}_t|\mathbf{z}_{t-1},\mathbf{x}_t,\boldsymbol{\theta})
$$


***
### **Spatial Field Data**

$$
\mathcal{D} = \{\mathbf{s}_n,\mathbf{x}_n,\mathbf{y}_n\}_{n=1}^N
$$

#### **Spatially Conditioned Model**


$$
p(\mathbf{Y},\mathbf{S},\mathbf{X},\mathbf{Z},\boldsymbol{\theta}) = 
p(\boldsymbol{\theta})
\prod_{m=1}^{N_\Omega}
p(\mathbf{y}_m|\mathbf{z}_m)p(\mathbf{z}_m|\mathbf{s}_m,\mathbf{x}_m,\boldsymbol{\theta})
$$


***
### **Spatio-Temporal Data**

$$
\begin{aligned}
\mathcal{D} &= \{t_n, \mathbf{s}_m, \mathbf{x}_{nm}, \mathbf{y}_{nm}\}_{n=1,m=1}^{N_T,N_\Omega},
&& &&
N = N_TN_\Omega
\end{aligned}
$$

#### **Spatiotemporal Conditioned Model**


$$
p(\mathbf{Y},\mathbf{X},\mathbf{t},\mathbf{S},\mathbf{Z},\boldsymbol{\theta}) = 
p(\boldsymbol{\theta})
\prod_{n=1}^{N_T}
\prod_{m=1}^{N_\Omega}
p(\mathbf{y}_{nm}|\mathbf{z}_{nm})
p(\mathbf{z}_{nm}|\mathbf{s}_m,t_n,\mathbf{x}_{nm},\boldsymbol{\theta})
$$

#### **Dynamical Model**

$$
p(\mathbf{Y},\mathbf{X},\mathbf{Z},\boldsymbol{\theta}) = 
p(\boldsymbol{\theta})
p(\mathbf{z}_0|\boldsymbol{\theta})
\prod_{t=1}^T
p(\mathbf{y}_t|\mathbf{z}_t)p(\mathbf{z}_t|\mathbf{z}_{t-1},\mathbf{x}_t,\boldsymbol{\theta})
$$


