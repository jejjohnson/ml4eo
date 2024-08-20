---
title: ML4EO - TOC Apps
subject: ML4EO
short_title: Application Track
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



## Application Track

### **Extreme Values Modeling**

#### **Introduction**

In this section, we will look at **why** this is such an important topic for the community.

#### **Anomalous Values**

In this section, we will explore the canonical dataset of *Global Mean Surface Temperature Anomaly*.
We will take inspiration from the [ClimateDataStore](https://climate.copernicus.eu/climate-indicators/temperature) and the [NOAA-NCEI](https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/mean) institutions which have graphs and host the source datasets.

#### **Mean Values**

In this section, we will take a step back and look at some local datasets for Spain.
We will start with a single weather station and go through the exact same setup as the previous section for anomalous values.
Then, we will increase the difficulty by taking into account multiple weather stations.

#### **Extreme Values**

In this section, we will look at the daily max values.
Now, we will do the exact same process for modeling but taking into account the extreme values.


---
### **Sea Surface Mapping**

#### **Introduction**

* The Big Why

#### **Unstructured Observations to Structured Observations**

In this first section, we will tackle unstructured observations.
We will do this through a deep dive into *function approximations* and demonstrate how these can be used to create structured observations.
We will do simple discretization procedures like histogram or nearest neighbours.
Then we will look at more advanced methods like non-parametric interpolation.
* Discretization
* Non-Parametric Interpolation


#### **Learning to Predict**

In this next section, we will investigate how we can
Finally, we will introduce parametric interpolation which is where we learn operators from the observations themselves.
* Problem Formulation & Experimental Setup - OSE
* Parametric Interpolation
* Parametric Forecasting

#### **Observations to Reanalysis**

* Differentiable Physics Models
* Representation Learning

#### **Learning to Assimilate**

* Problem Formulation + Experimental Setup
* Equilibrium Models

#### **Learning to Predict**
* Representation Learning
* Parametric Forecasting

---
### **Instrument-2-Instrument**

#### **Introduction**

#### **Satellite Datasets**

- Different Satellites
- Data Downloader Pipelines

#### **GeoProcessing Pipelines**

- Geostationary Satellites
- Polar Orbiting Satellites - SWATH
- Polar Orbiting Satellites - AlongTrack

#### **GeoData 4 ML**

- Reprojections

#### **Representation Learning**

- Parameterization
- Foundation Model
- Transfer Learning

#### **Operator Learning**

- Neural Operators

***
**Extreme Values Modeling**

The first application track we have are looking at extreme values.
This allows us to slowly introduce some core concepts that will be useful in the later application tracks.

$$
\begin{aligned}
D_s &= \left[ \text{Altitude, Longitude, Latitude}\right] \\
\Omega &= \left[ \text{Unstructured Weather Stations}\right] \\
\Delta t &= \left\{ \text{Daily, Hourly} \right\} \\
D_y &= \left[ \text{Temperature, Precipitation, Wind Speed}\right]
\end{aligned}
$$

***
**Sea Surface Mapping**

The second application track is working with sea surface mapping of variables like temperature, height, and ocean colour.
In this setting, we will deal with how to transform between different geometries like the alongtrack satellite observations and the regular gridded applications.
These measurements feature a lot of missing values, so we will learn how to deal with this.
The bulk of this application will involve learning-to-learn, whereby we find various ways to learn prior models

$$
\begin{aligned}
D_s &= \left[ \text{Depth, Longitude, Latitude}\right] \\
\Omega &= \left[ \text{Irregular Along Track, Regular Gridded}\right] \\
\Delta t &= \left\{ \text{Daily, Hourly} \right\} \\
D_y &= \left[ \text{Temperature, Height, Salinity, Colour}\right]
\end{aligned}
$$

***
**Instrument-2-Instrument Translation**

In this third application track, we will investigate how we can apply a method to translate one type of satellite to another type of satellite.
Here, we have to do all tasks including interpolation, X-casting, and variable transformation.
The interpolation component is due to the non-consistent geometry amongst the satellites.
The X-casting component comes from the 
We also have to deal with a 


$$
\begin{aligned}
D_s &= \left[ \text{Height, Longitude, Latitude}\right] \\
\Omega &= \left[ \text{Irregular Polar-Orbiting Satellite, Regular Geostationary Satellite}\right] \\
\Delta t &= \left\{ \text{Days, 15 Minutes} \right\} \\
D_y &= \left[ \text{Spectral Signature}\right]
\end{aligned}
$$

***
### Break Down

Here, we have a breakdown of how we will structure the application section.
We will analyse the data which will be used to learn.
Then, we will do whatever interpolation or discretization necessary to transform it to the form we want.
Then, we will do some parameter estimation to learn a representation of the system.
Lastly, we will do state estimation using whichever form is 

* **Data Acquisition** - GPs, Simulations, L1 Data, L2 Data, L3 Data
* **Interpolation & Discretization** - Histograms, Nearest Neighbors, Gaussian Processes
* **Learning Priors** - Gaussian Processes, L2 Data, L3 Data, Simulations
* **Parameter Estimation** - Representation Learning, Dynamical Systems
* **State Estimation** - 3DVar, Strong-Constrained 4D Var, Weak-Constrained 4DVar, Deep Equillibrium 


## 

## Walk-Through


For each of these sections


### **Download EO Data**

> Examples: Sea Surface Height/Temperature/Salinity, Satellite Data, Weather Stations

$$
\begin{aligned}
\text{LS Data}: && &&
&\left[\text{GP Priors}, \text{Simulations} \right] \\
\text{L1 Data}: && &&
&\left[\text{Weather Stations}, \text{ARGO Floats}, \text{Satellite}\right] \\
\text{L2 Data}: && &&
&\left[\text{Interpolated Data} \right] \\
\text{L3 Data}: && &&
&\left[\text{Reanalysis} \right] \\
\text{L4 Data}: && &&
&\left[\text{Reanalysis} \right] \\
\end{aligned}
$$

***
### **Discretizations**

> Examples: Histogram, Nearest Neighbours, Radius Neighbours, Kernel Density Estimation

$$
p(y_{1:T},\boldsymbol{\theta}) = \prod_{n=1}^{N_T}p(y_t|\theta)p(\theta|s_n, t_n)
$$


***
### **Functional Regression**

> Examples: Nearest Neighbors, Gaussian Processes, Neural Fields

$$
p(y_{1:N},s_{1:N},t_{1:N},\boldsymbol{\theta}) = p(\theta)\prod_{n=1}^{N_T}p(y_n|s_n, t_n,\theta)
$$


***
### **Param. Est. - GP Priors**

> Examples: RBF, Matern

$$
\begin{aligned}
\text{Joint}: && &&
p(f, \alpha, z, \theta) &= p(f|z)p(z|\alpha,\theta)p(\theta) \\
\text{Posterior}: && &&
q(\alpha, z, f, \theta) 
&=
q(\alpha) \prod_{n=1}^N
q(z|f,\alpha) \\
\end{aligned}
$$


***
**State Estimation**

$$
q(y_{1:T},z_{1:T}^*,\alpha^*,\theta) = q(\theta^*)q()
$$

***
### **Param. Est. - Simulations**



***
### **Param. Est. - Spatial Field Priors**

$$
\begin{aligned}
\text{Joint}: && &&
p(y,z,\theta) &= p(\theta)\prod_{n=1}^Np(y_n|z_n)p(z_n|\theta) \\
\text{Posterior}: && &&
q(z,y,\theta) &= q(\theta)\prod_{n=1}^N
q(z|y,\theta)
\end{aligned}
$$

***
### **State Estimation**

$$
q(z,y,\theta) = q(z|\theta,\phi^*)q(\theta|\phi^*)
$$


***
### **Equilibrium Models**

$$
z^* = f(z^*, y, x, \theta)
$$

***
### **Param. Est. - Strong-Constrained Dynamical Priors**

$$
\begin{aligned}
\text{Joint}: && &&
p(y_{1:T},z_{0:T},\theta) 
&= 
p(\theta)p(z_0) 
\prod_{n=1}^N
p(y_t|z_t,\theta) \\
\text{Variational}: && &&
q(z_{0:T},y_{1:T},\theta) &=
q(\theta) q(z_0)
\prod_{n=1}^N
q(z_t|y_t,\theta)
\end{aligned}
$$


***
### **State Estimation**



***
### **Param. Est. - Weak-Constrained Dynamical Prior**

$$
\begin{aligned}
\text{Joint}: && &&
p(y_{1:T},z_{0:T},\theta) 
&= 
p(\theta)p(z_0) 
\prod_{n=1}^N
p(y_t|z_t)p(z_t|z_{t-1},\theta) \\
\text{Variational}: && &&
q(z_{0:T},y_{1:T},\theta) &=
q(\theta) q(z_0)
\prod_{n=1}^N
q(z_t|z_{t-1},y_t)
q(z_t|z_{t-1},\theta)
\end{aligned}
$$


***
### **State Estimation**

$$
q(y_{1:T},z_{1:T}^*,\theta^*) = q(\theta^*)q(z_0^*)
\prod_{t=1}^Tq(z_t|z_{t-1},y_t)
$$



> We download some


***
## Applications

**Extreme Values**. 
Here, we learn how we can estimate the parameters of a model for extreme values.

**Ocean Surface Height**.
We will learn how to create mapping products for Sea Surface Height (SSH) and Sea Surface Temperature (SST).

**Instrument-2-Instrument**.
We will learn how to map one set of spatio-spectral channels to a different spatio-spectral channels.

***
### Application I - Extremes

In this application, we are interested in modeling temperatures, and in particular, temperature extremes.

**Datasets**.
We will use some standard datasets that are found within the AEMET database.

$$
\begin{aligned}
\mathcal{D} &= \left\{(t_n, \mathbf{s}_n), \mathbf{x}_n, \mathbf{y}_n \right\}_{n=1}^{N},
&& &&
\mathbf{y}_n\in\mathbb{R}^{D_y} &&
\mathbf{x}_n\in\mathbb{R}^{D_y} &&
\mathbf{s}_n\in\mathbb{R}^{D_s} &&
t_n\in\mathbb{R}^{+}
\end{aligned}
$$


* Covariante, $x_n$
  * Global Mean Surface Temperature Anomaly (GMSTA)
* Measurement, $y_n$: 
  * Daily Mean Temperature (TMax)
  * Daily Maximum Temperature (TMean)


***
**Extreme Events**.
$$
\mathcal{D} = \left\{y_n\right\}_{n=1}^N
$$
We will use some standard techniques to extract the extreme events from the time series.
* Block Maximum (BM)
* Peak-Over-Threshold (POT)
* Temporal Point Process (TPP)


***
**Data Likelihood**.

$$
y \sim p(y|z)
$$
We need to decide which likelihood we want to use.
For the mean data, we can assume that it follows a Gaussian distribution or a long-tailed T-Student distribution.
However, for the maximum values, we need to assume an EVD like the GEVD, GPD or TPP.
* Mean -> Gaussian, T-Student
* Maximum -> GEVD, GPD, TPP

***
**Parameterizations**.
$$
p(\mathbf{y},\mathbf{z},\boldsymbol{\theta}) = p(y|z)p(z|\theta)p(\theta)
$$
We have to decide the parameterization we wish to use.
* IID
* Hierarchical
* Strong-Constrained Dynamical, i.e., Neural ODE/PDE
* Weak-Constrained Dynamical, i.e., SSM, EnsKF

***
**State Estimation**

$$
z^*(\theta) = \argmin \hspace{2mm}\boldsymbol{J}(z;\theta)
$$

* Weak-Constrained
  * Gradient-Based - 4DVar
  * Derivative-Free - Ensemble Kalman Filter
  * Minimization-Based - DEQ

***
**Predictions**

$$
u \sim p(u|z^*)p(z^*|\mathcal{D})
$$

Our final task is to use a our new model to make predictions.
* Feature Representation
* Interpolation
* HindCasting
* Forecasting


***
## Application II - Sea Surface Height Mapping


In this application, we will investigate different ways that we can create maps of sea surface height (SSH).