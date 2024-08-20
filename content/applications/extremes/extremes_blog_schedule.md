---
title: Modeling Extreme Values
subject: ML4EO
short_title: Blog Schedule
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


***
## Blog Schedule

**Part I**: Global Mean Surface Temperature Anomaly

**Part II**: Single Weather Station *Means*

**Part III**: Multiple Weather Station *Means*

**Part IV**: Weather Station *Extremes*

**Part V**: Multiple Weather Station *Extremes*

---
## **Part I**: **The Big Why**

* Extremes are Difficult - The Tails of the distribution


---
## **Part II**: *GMSTA*

> This first part will provide a nice base case to step through each of the individual modeling decisions we have to do once we


:::{seealso} **Topics**
:class: dropdown

- Data Download + EDA
- Data Pipelines
- Recreating the Anomalies
- Signal Decomposition
- GeoProcessing Pipelines
- Unconditional Model
- Metrics
- Likelihood Whirlwind Tour
- Inference Whirlwind Tour
- Bayesian Hierarchical Model
- Temporal Conditioned Model
- Dynamical Model
- State Space Model
- Structured State Space Model
- Ensembles

:::




---
##### **1. Data Download + EDA**

> In this tutorial, we want to showcase some of the immediate data properties that we can see just from plotting the data and calculate some statistics.
> A common theme we would like to showcase is that there are different ways to measure samples: trend, spread, and shape.
> Furthermore, we can get different things depending upon the discretization.

* Data Sources - [ClimateDataStore](https://climate.copernicus.eu/climate-indicators/temperature), [NOAA-NCEI](https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/mean)
* EDA - Statistics, Stationarity, Noise
* Viz - Scatter, Histogram
* Temporal Binning - All, Decade, Year, Season, Month
* Library - `matplotlib`, `numpy`

---
##### **2. Data Pipelines**

> In this tutorial, we want to showcase some of the immediate data properties that we can see just from plotting the data and calculate some statistics.

* Data Sources - [ClimateDataStore](https://climate.copernicus.eu/climate-indicators/temperature), [NOAA-NCEI](https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/mean)
* API Keys
* Packages - `Typer`, `Hydra`, `DVC`


---
##### **3. Recreating the Anomalies**

> We do some manual feature extraction where we try to recreate these anomalies from the original data.
> This involves decomposing the signal, spatial averaging and temporal smoothing.
> In essences, this is a quick tutorial about how we can gather anomalies in a classical way and what we will **not** be doing in the following tutorials.

* Manual Feature Extraction
* Periods Signal - Climatology, Reference Periods
* Spatial Averaging - Weighted Means
* Temperal Smoothing - Filtering
* Library - `xarray`

---
##### **4. Signal Decomposition**

> We do some classical signal decomposition assuming 3 underlying components: trends, cycles, and residuals.
> We also showcase some different ways to express a model using these 3 components: additive, multiplicative, and non-linear.
> This will serve as a precursor to the parameter estimation task where we need to describe an underlying parameterization of the signal.

* *Components* - Trend, Cycle, Residuals
* *Combination* - Additive, Multiplicative, Non-Linear
* Precursor to parameter estimation
* Library - `statsmodels`


---
##### **5. Unconditional Model**

> We will apply the simplest parameterization: assuming IID.
> This will establish a baseline and we will start to get comfortable with the Bayesian language for modeling.



* Baseline Model - Fully Pooled Constrained
* Data Likelihood - Normal
* Prior - Uninformative
* Inference - MAP + MCMC
* Library - `Numpyro`

:::{seealso} **Equations**
:class: dropdown

$$
p(\mathbf{y},\boldsymbol{\theta}) = p(\boldsymbol{\theta})\prod_{n=1}^N p(y_n|\boldsymbol{\theta})
$$

:::


---
##### **6. Metrics**

* Fit - PP-Plot, QQ-Plot
* Parameters - Joint Plot
* Tails - Return Period vs Empirical 
* Sensitivity Analysis - Gradient-Based, Sampling-Based, Proxy-Based
* Summary Stats - NLL, AIC, BIC
* Libraries - `pyviz, xarray`

---
##### **7. Bayesian Hierarchical Model**

> We will explore some improved parameterization strategies when thinking about.

* Parameterizations
	* Baseline Model - Fully Pooled Constrained
	* Non-Pooled - Unconstrained
	* Partially Pooled - Bayesian Hierarchical Model
* Data Likelihood - Normal
* Inference - MAP + MCMC
* Library - `Numpyro`

:::{seealso} **Equations**
:class: dropdown

$$
\begin{aligned}
\text{Fully Pooled}: && &&
p(\mathbf{y},\boldsymbol{\theta}) &= p(\boldsymbol{\theta})\prod_{n=1}^N p(y_n|\boldsymbol{\theta}) \\
\text{Non-Pooled}: && &&
p(\mathbf{y},\boldsymbol{\theta}) &= \prod_{n=1}^N p(y_n|\boldsymbol{\theta}_n)p(\boldsymbol{\theta}_n) \\
\text{Partially-Pooled}: && &&
p(\mathbf{y},\boldsymbol{\theta}) &= p(\boldsymbol{\theta})\prod_{n=1}^N p(y_n|\boldsymbol{z}_n)p(\boldsymbol{z}_n|\boldsymbol{\theta}) \\
\end{aligned}
$$

:::

---
##### **8. Likelihoods Whirlwind Tour**

> The likelihood is an important piece of the Bayesian modeling framework.
> We show which likelihoods make sense for which datasets depending upon the structure of the data.
> We showcase the standard Gaussian but we also explore more heavy-tailed distributions like Log-Normal and T-Student.

* Simple Likelihoods
    * Gaussian
    * Log-Normal
    * T-Student
	* GEVD
* Inference - MAP + MCMC
* Library - `Numpyro`

:::{seealso} **Equations**
:class: dropdown

$$
\begin{aligned}
\text{Normal}: && &&
p(\mathbf{y};\boldsymbol{\theta}) &=  
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(- \frac{(y - \mu)^2}{2\sigma^2}\right)
\\
\text{Log-Normal}: && &&
p(\mathbf{y};\boldsymbol{\theta}) &=  
\frac{1}{y\sigma\sqrt{2\pi}}
\exp
\left(- \frac{(\ln y - \mu)^2}{2\sigma^2}\right)
\\
\text{T-Student}: && &&
p(\mathbf{y},\boldsymbol{\theta}) &=  
\frac{1}{2} + y \Gamma\left(\frac{\nu+1}{2}\right)
\frac{2 F_1\left(\frac{1}{2},\frac{\nu+1}{2},\frac{3}{2},-\frac{y^2}{\nu}\right)}{\sqrt{\pi \nu}\Gamma(\nu/2)}\\
\end{aligned}
$$

:::


---
##### **9. Inference Whirlwind Tour**

> This will give a whirlwind tour of some basic inference schemes: how we will learn the parameters of our model.
> We will give the barebones scheme where there is no uncertainty, i.e., uniform priors --> MLE.
> We will also demonstrate how to find the parameters using the sampling scheme, MCMC.
> Lastly, we will give an overview of a method to approximate the posterior distribution, i.e., VI.

* Inference Schemes
	* Sample-Based - MCMC
    * Non-Bayesian - MLE
    * Approximate Bayesian - VI
* Library - `Numpyro`


---
##### **10. Temporally Conditioned Model**

> We will introduce the notion of conditioning our data on the time stamp.
> This is a natural introduction to how to properly model time series data.

* Time Coordinate Encoder - Year, Season, Month
* Priors - Normal, Uniform, Laplace, Delta
	* Bias - Intercept, i.e., $t=0$ --> Normal + Mean @ $t=0$
	* Weight - Slope/Tendency --> Normal
	* Noise - Normal, Cauchy
* Inference - MAP + MCMC
* Metrics
	* Parameters - Joint Plots, Scatter
	* Tails Analysis - Return Periods + Empirical
	* Differences between T1 and T0 -  Return Periods (1D, 2D), Line Plots, 
	* Sensitivity Analysis - Gradient-Based e.g., $\partial_t f(t,\theta)=w$
* Predictions
	* Hindcasting
	* Forecasting
* Extreme Values - Block Maximum v.s. Peak-Over-Threshold
* Parameterization - Temporal Point Process
* Relationship with common dists, GEVD & GPD
* Custom Likelihood in `Numpyro` - GEVD, GPD


:::{seealso} **Equations**
:class: dropdown

$$
\begin{aligned}
\text{Data}: && &&
\mathcal{D} &= 
\left\{ t_n, y_n\right\}_{n=1}^N 
\\
\text{Joint Distribution}:
&& &&
p(\mathbf{y},\mathbf{t},\mathbf{z}, \boldsymbol{\theta}) 
&=
p(\boldsymbol{\theta})
\prod_{n=1}^N 
p(y_n|\mathbf{z}_n)
p(\mathbf{z}_n|t_n,\boldsymbol{\theta})
\end{aligned}
$$

where $y_n\in\mathbb{R}$ and $t_n\in\mathbb{R}^+$.

:::

---
##### **11. Dynamical Model**

> In this module, we will introduce dynamical model formulism as an alternative parameterization.
> We do not explicitly condition on the time step itself
> Instead, we condition on the state at a previous time step as well as observations.

- Dynamical Model Formalization
	- Initial Condition
	- Equation of Motion
	- TimeStepper —> ODESolver
	- Observation Operator
- Equation of Motion Parameterizations
	-  Closed Form - Constant, Linear, Exponential, Logistic —> Closed-Form Solution
	- Structured - Linear, Reduced Order, Exponential, 
	- Free-Form - Neural Network
- TimeStepper - Quadrature (Runge-Kutta)
- Observation Operator - Linear
- Inference - MAP + MCMC
- Predictions - Hindcasting + Forecasting


:::{seealso} **Equations**
:class: dropdown

$$
\begin{aligned}
\text{Data}: && &&
\mathcal{D} &= 
\left\{ t, y_t\right\}_{t=1}^T 
\\
\text{Joint Distribution}:
&& &&
p(\mathbf{y},\mathbf{z}, \boldsymbol{\theta}) 
&=
p(\boldsymbol{\theta})
p(\mathbf{z}_0|\boldsymbol{\theta})
\prod_{t=1}^T
p(y_t|\mathbf{z}_t)
p(\mathbf{z}_t|\mathbf{z}_{t-1},\boldsymbol{\theta})
\end{aligned}
$$

where $y_n\in\mathbb{R}$ and $t_n\in\mathbb{R}^+$.

:::



---
##### **12. State Space Model**

- State Space Formalization
	- Initial Distribution
	- Transition Distribution
	- Emission Distribution
	- Posterior - Filtering, Smoothing
- Connections (Generalization)
	- ODE —> Strong-Constrained vs Weak-Constrained
	- Time Conditioned —> Full-Form vs Gradient-Form
- Inference - MAP + MCMC
- Predictions - Hindcasting + Forecasting

:::{seealso} **Equations**
:class: dropdown

$$
\begin{aligned}
\text{Data}: && &&
\mathcal{D} &= 
\left\{ t, y_t\right\}_{t=1}^T 
\\
\text{Joint Distribution}:
&& &&
p(\mathbf{y},\mathbf{z}, \boldsymbol{\theta}) 
&=
p(\boldsymbol{\theta})
p(\mathbf{z}_0|\boldsymbol{\theta})
\prod_{t=1}^T
p(y_t|\mathbf{z}_t)
p(\mathbf{z}_t|\mathbf{z}_{t-1},\boldsymbol{\theta})
\end{aligned}
$$

where $y_n\in\mathbb{R}$ and $t_n\in\mathbb{R}^+$.

:::

---
##### **13. Structured State Space Model**

- Structured
	- Time Dependence - Cycle, Season
	- Trend, Locally Linear
	- Temporal History Dependence - Autoregressive
- Inference - MAP + MCMC

---
##### **14. Whirlwind Tour**
* Linear
* Basis Function
* Neural Network
* Gaussian Processes

---
##### **15. Ensembles**
* Multiple GMSTA Perspectives


* X-Casting
* Strong-Constrained Dynamic Model, aka, NeuralODE
* Weak-Constrained Dynamical Model, aka, SSM

---
## **Part III**: *Single Weather Station*

> In this module, we start to look at single weather stations for Spain.

---
##### **16. EDA Revisited**

> In this tutorial, we want to showcase some of the immediate data properties that we can see just from plotting the data and calculate some statistics.
> A common theme we would like to showcase is that there are different ways to measure samples: trend, spread, and shape.
> Furthermore, we can get different things depending upon the discretization.

* Data Sources - [AEMET-OpenData](https://github.com/Noltari/AEMET-OpenData/tree/master), [python-aemet](https://github.com/pablo-moreno/python-aemet/tree/master)
* Datasets - 
* EDA - Statistics, Stationarity, Noise
* Viz - Scatter, Histogram
* Temporal Binning - All, Decade, Year, Season, Month
* Library - `matplotlib`, `numpy`

---
##### **17. Baseline Model**

* Data Download + EDA - Histograms, Stationarity, Noise
* Datasets
	* Temperature
	* Precipitation
	* Wind Speed
* Data Likelihoods
    * Standard - Gaussian, Generalized Gaussian
    * Long-Tailed - T-Student, LogNormal
* Parameterizations - Fully Pooled, Non-Pooled, Partially Pooled

---
##### **18. State Space Model**


* Predictions - Hindcasting, Forecasting



---
## **Part IV**: *Multiple Weather Stations*

**Introduction**
- EDA - Multiple Weather Stations, AutoCorrelation, Variogram

**Unconditional Models** - *Spatiotemporal Series*
- Baseline Model - State Space Model w/ Spatial Dims
- Spatial Models - EDA + Weight Matrix
- Spatial State Space Model
- Scale - Variational Posterior

**Conditional Models** - *Spatiotemporal Series*
- EDA - Multiple Weather Stations + Covariates
- Baseline Model - IID —> Bayesian Hierarchical Model
- Conditional SSMs
- Reparameterization

**Other**
* EDA - Exploring Spatial Dependencies (Altitude, Longitude, Latitude)
* Spatial Autocorrelation with (Semi-)Variograms
* Discretization - Histogram
* Dynamical Model
* Spatial Operator - Finite Difference, Convolutions

---
##### **19. EDA**

- Histograms - Grouped (Time)
- Scatter Plots - Binned
- Multiple Temporal AutoCorrelation Plots
- Spatial Autocorrelation, Variogram
- Clustering - GMMs (Grouped)


---
##### **20. Baseline Model**

* Batch Processing

$$
\begin{aligned}
p(\mathbf{Y},\mathbf{z}, \boldsymbol{\theta})
&=
\prod_{m=1}^{N_\Omega}
p(\boldsymbol{\theta}_m)
\prod_{n=1}^{N_T}
p(\mathbf{y}_{nm}|\mathbf{z}_{nm})
p(\boldsymbol{z}_{nm}|\boldsymbol{\theta}_m)
\end{aligned}
$$




---
##### **21. Regressor** - *Weight Matrix*

- EDA - Spatial Correlation, Variogram
- Domain Shape
	- Unstructured, Irregular - Graph —> Adjacency Matrix
	- Regular - Convolution —> Kernel



---
##### **22. GP Regressor**

$$
\begin{aligned}
p(\mathbf{Y},\mathbf{z}, \boldsymbol{\theta})
&=
p(\boldsymbol{\theta})
p(\boldsymbol{\alpha})
p(\boldsymbol{f}|\boldsymbol{\alpha})
\prod_{n=1}^{N_T}
p(\mathbf{y}_{n}|\mathbf{z}_{n})
p(\mathbf{z}_{n}|\boldsymbol{f},\boldsymbol{\theta})
\end{aligned}
$$


---
##### **23. SSM** - *Spatial Model*

$$
\begin{aligned}
p(\mathbf{Y},\mathbf{z}, \boldsymbol{\theta})
&=
p(\boldsymbol{\theta})
p(\mathbf{z}_0|\boldsymbol{\theta})
\prod_{t=1}^{T}
p(\mathbf{y}_{t}|\mathbf{z}_{t})
p(\mathbf{z}_{t}| \mathbf{z}_{t-1}\boldsymbol{\theta})
\end{aligned}
$$



- Spatial Operator Parameterizations
	- Fully Connected
	- Convolutions

---
##### **24. Scale** - *VI*


- Whirlwind Tour
	- Filter-Update Posterior, $q(z_t|z_{t-1}, y_t)$
	- Smoothing Posterior, $q(z_t|z_{t-1}, y_{1:T})$



---
## **Part V**: Spain Weather Stations (Extremes)

##### **TOC**

- What is an Extreme Event?
- Classic Method I - Block Maximum
- Classic Method II - Peak Over Threshold
- Classic Method III - Point Process
- Revised Method - Marked Temporal Point Process

##### **25. What is an Extreme Event?**

- What is an event?
- Objective - Forecasting, Return Period
- Definitions
	- Mean vs Tails
	- Maximum/Minimum, Thresholds
	- Power Law
- Problems
	- Tails - Few/No Observations
	- Independence - even with observations, not independent
	- Models - Few Obs + Dependence —> Difficult to Fit a model 
- Whirlwind Tour - BM, POT, TPP
- Example
	- Gaussian, Generalized Gaussian, T-Student, GEVD, GPD
	- Sample Data Likelihood - x100, x1000, x10000


##### **26. Block Maximum**

- What is an event? - The maximum over within a block of time.
- Temporal Resolution - Year, Season, Month, Day
- Viz - Histogram, Scatter Plot, Violin Plot, Ridge Plot
- EDA - `seaborn` simple linear regressors, i.e., trends

##### **27. Peak Over Threshold**

- What is an event? - An Event Over a Threshold
- Threshold Selection - Quantiles (90, 95, 98, 99)
- Temporal Resolution (Declustering) - Year, Season, Month, Day
- Viz - Histogram, Scatter Plot, Violin Plot, Ridge Plot
- EDA - `seaborn` simple linear regressors, i.e., trends


##### **28. Temporal Point Process**
- What is an event? - The Events Over a Threshold within a block of time.
- Block Maximum Temporal Resolution - Year, Season, Month, Week, Days
- Threshold Selection - Quantiles
- Theory - Point Process for Extremes
- Viz - Histogram, Scatter Plot, Violin Plot, Ridge Plot
- EDA - `seaborn` simple linear regressors, i.e., trends
- Data Likelihood - Point Process
- Baseline Model - Pooled, Non-Pooled, Partially Pooled

##### **Engineering I** - *Block Maximum*
- Data - Download from DVC
- Geoprocessing - Select Station, Clean Labels
- ML Pre-Processing - Standardization, Train/Valid/Test Split
- ML Training - Model Load, Model Train, Model Save
- MLOps - EDA, Metrics, Hindcasting, Forecasting


##### **29. Parameterization MTPP**
- What is an event? - The Events Over a Threshold within a block of time.
- What is a mark? - The intensity of an event if it happens.
- Block Maximum Temporal Resolution - Year, Season, Month, Week, Days
- Threshold Selection - Quantiles
- Theory - Marked Decoupled Point Process
- Data Likelihood - Point Process + Marks Distribution
- Baseline Model - Pooled, Non-Pooled, Partially Pooled

##### **30. Baseline Model** - *Univariate*

- Data Likelihood - GEVD: Limiting Distribution for Extremes
- Baseline Model
	- Fully Pooled - Constrained
	- Non-Pooled - Unconstrained
	- Partially Pooled - Bayesian Hierarchical Model
- Constraints - [Tails](https://www.financialriskforecasting.com/files/Danielsson-Financial-Risk-Forecasting-Slides-9.pdf)
	- Frechet - e.g., Temperature
	- Weibull - e.g., Precipitation
- Inference - MAP + MCMC
- Metrics 
	- Fit - PP-Plot, QQ-Plot
	- Parameters - Joint Plot
	- Tails - Return Period + Empirical
	- Sensitivity Analysis - Gradient-Based, Sampling-Based, Proxy-Based
- Pipeline - Data, Model, Inference, Metrics



##### **State Space Model** - *Baseline*

- State Space Formalization
	- Initial Distribution
	- Transition Distribution
	- Emission Distribution
- Connections (Generalization)
	- ODE —> Strong-Constrained vs Weak-Constrained
	- Time Conditioned —> Full-Form vs Gradient-Form
- Inference
	- Filter-Update, Smoothing
	- MAP + MCMC
- Predictions - Hindcasting + Forecasting


##### **State Space Models** - *Structured*

- Structured
	- Time Dependence - Cycle, Season
	- Trend, Locally Linear
	- Temporal History Dependence - Autoregressive
- Inference - MAP + MCMC



---
## **Part VI**: Spain Weather Stations (Extremes)






<!-- ### Temporally Conditioned Model

* Temporal Encoding - Linear, Seasonal, Monthly, 
* Priors
	* Normal, Uniform, Laplace, Delta
	* Bias - Intercept, i.e., $t=0$
	* Weight - Slope/Tendency
	* Noise - Normal, Cauchy
* Inference - MAP + Inference
* Metrics
	* Parameters - Joint Plots, Scatter
	* Tails Analysis - Return Periods + Empirical
	* Differences between T1 and T0 -  Return Periods (1D, 2D), Line Plots, 
	* Sensitivity Analysis - Gradient-Based e.g., $\partial_t f(t,\theta)=w$
* Predictions
	* Hindcasting
	* Forecasting
* Extreme Values - Block Maximum v.s. Peak-Over-Threshold
* Parameterization - Temporal Point Process
* Relationship with common dists, GEVD & GPD
* Custom Likelihood in `Numpyro` - GEVD, GPD


---
### Dynamic Model

- Dynamical Model Formalization
	- Initial Condition
	- Equation of Motion
	- TimeStepper —> ODESolver
- Equation of Motion Parameterizations
	-  Closed Form - Constant, Linear, Exponential, Logistic —> Closed-Form Solution
	- Structured - Linear, Reduced Order, Exponential, 
	- Free-Form - Neural Network
- Inference 
	- Latent Variable Model
	- MAP + MCMC
	- Variational Posterior
- Predictions - Hindcasting + Forecasting

---
### State Space Model

- State Space Formalization
	- Initial Distribution
	- Transition Distribution
	- Emission Distribution
- Connections (Generalization)
	- ODE —> Strong-Constrained vs Weak-Constrained
	- Time Conditioned —> Full-Form vs Gradient-Form
- Inference
	- Filter-Update, Smoothing
	- MAP + MCMC
- Predictions - Hindcasting + Forecasting

---
### Structured State Space Model

- Structured
	- Time Dependence - Cycle, Season
	- Trend, Locally Linear
	- Temporal History Dependence - Autoregressive
- Inference - MAP + MCMC


---
### Conditional

* EDA - Simple Models (LR, Binning, Stats)
* Viz - Correlation Plots


---
### Conditional Dynamical Model


- Feature Representations - Linear, Basis, Non-Linear
- Inference - MAP + MCMC
- Predictions - Hindcasting, Forecasting

---
### Reparameterized Dynamical Model

- Non-dimensionalization
- Predictions - Hindcasting, Forecasting


---
### Conditional SSM


- Feature Representations - Linear, Basis, Non-Linear
- Inference - MAP + MCMC
- Predictions - Hindcasting, Forecasting

---
### Reparameterized SSM

- Non-dimensionalization
- Predictions - Hindcasting, Forecasting
 -->
