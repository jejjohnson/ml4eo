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

##### **TOC**

- **EDA** - Data Download
- **Engineering** - Data Pipelines
- **Feature Extraction** - Recreating the Anomalies
- **Feature Extraction** - Signal Decomposition
- **Engineering** - GeoProcessing Pipelines
- **Modeling** | **Parameterization** - Unconditional Model
- **Validation** - Metrics
- **Modeling** | **Parameterization** - Likelihood Whirlwind Tour
- **Modeling** | **Inference** - Inference Whirlwind Tour
- **Modeling** | **Parameterization** - Bayesian Hierarchical Model
- **Modeling** | **Parameterization** - Temporal Conditioned Model
- **Modeling** | **Parameterization** - Dynamical Model
- **Modeling** - State Space Model
- **Modeling** | **Parameterization** - Structured State Space Model
- **Modeling** | **Uncertainty** - Ensembles

---
##### **Data Download + EDA**

> In this tutorial, we want to showcase some of the immediate data properties that we can see just from plotting the data and calculate some statistics.

* EDA - Statistics, Stationarity, Noise
* Viz - Scatter, Histogram
* Data Sources - [ClimateDataStore](https://climate.copernicus.eu/climate-indicators/temperature), [NOAA-NCEI](https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/mean)
* Library - `matplotlib`, `numpy`

---
##### **Engineering** - Data Pipelines

> In this tutorial, we want to showcase some of the immediate data properties that we can see just from plotting the data and calculate some statistics.

* Data Sources - [ClimateDataStore](https://climate.copernicus.eu/climate-indicators/temperature), [NOAA-NCEI](https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/mean)
* API Keys
* Scripts - `Typer`, `Hydra`


---
##### **Recreating the Anomalies**
* Manual Feature Extraction
* Periods Signal - Climatology, Reference Periods
* Spatial Averaging - Weighted Means
* Temperal Smoothing - Filtering
* Library - `xarray`

---
##### **Signal Decomposition**

* Trend, Cycle, Residuals
* Additive, Multiplicative, Non-Linear
* Precursor to parameter estimation
* Library - `statsmodels`


---
##### **Unconditional Model**

* Baseline Model - Fully Pooled Constrained
* Data Likelihood - Normal
* Inference - MAP + MCMC
* Library - `Numpyro`

---
##### **Metrics**
* PP-Plot, QQ-Plot, Posterior, Joint Plot, Return Period
* NLL, AIC, BIC
* `pyviz, xarray`

---
##### **Bayesian Hierarchical Model**

* Baseline Model
    * Fully Pooled Constrained
    * Non-Pooled - Unconstrained
    * Partially Pooled - Bayesian Hierarchical Model
* Data Likelihood - Normal
* Inference - MAP + MCMC
* Library - `Numpyro`

---
##### **Likelihoods Whirlwind Tour**

* Simple Likelihoods
    * Gaussian
    * Log-Normal
    * T-Student
* Inference - MAP + MCMC
* Library - `Numpyro`

---
##### **Inference Whirlwind Tour**

* Simple Likelihoods
    * Non-Bayesian - MLE
    * Approximate Bayesian - MAP
    * Sample-Based - MCMC
* Library - `Numpyro`


---
##### **Temporally Conditioned Model**

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
##### **Dynamical Model**

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
##### **State Space Model**

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
##### **Structured State Space Model**

- Structured
	- Time Dependence - Cycle, Season
	- Trend, Locally Linear
	- Temporal History Dependence - Autoregressive
- Inference - MAP + MCMC

---
##### **Whirlwind Tour**
* Linear
* Basis Function
* Neural Network
* Gaussian Processes

---
##### **Ensembles**
* Multiple GMSTA Perspectives


* X-Casting
* Strong-Constrained Dynamic Model, aka, NeuralODE
* Weak-Constrained Dynamical Model, aka, SSM

---
## **Part III**: *Single Weather Station*

> In this module, we start to look at single weather stations for Spain.

* Data Download + EDA - Histograms, Stationarity, Noise
* Data Likelihoods
    * Standard - Gaussian, Generalized Gaussian
    * Long-Tailed - T-Student, LogNormal
* IID
* Discretization
    * Regular + Finite Diff + Convolutions
    * Irregular + Symbolic + AD
* Dynamical


---
## **Part IV**: *Multiple Weather Stations*

* EDA - Exploring Spatial Dependencies (Altitude, Longitude, Latitude)
* Spatial Autocorrelation with (Semi-)Variograms
* Discretization - Histogram
* Dynamical Model
* Spatial Operator - Finite Difference, Convolutions

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


##### **EDA**

- Histograms - Grouped (Time)
- Scatter Plots - Binned
- Multiple Temporal AutoCorrelation Plots
- Spatial Autocorrelation, Variogram
- Clustering - GMMs (Grouped)
##### **Baseline Model** 

- Baseline - Fully Pooled, Non-Pooled, Partially Pooled
- Inference - MAP + MCMC


##### **Spatial Model** - *Weight Matrix*

- EDA - Spatial Correlation, Variogram
- Domain Shape
	- Unstructured, Irregular - Graph —> Adjacency Matrix
	- Regular - Convolution —> Kernel


##### **State Space Model** - *Spatial Model*

- Spatial Operator Parameterizations
	- Fully Connected
	- Convolutions

##### **Scale** - *Variational Posterior*


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

##### **What is an Extreme Event?**

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


##### **Classical Methods I** - *Block Maximum*

- What is an event? - The maximum over within a block of time.
- Temporal Resolution - Year, Season, Month, Day
- Viz - Histogram, Scatter Plot, Violin Plot, Ridge Plot
- EDA - `seaborn` simple linear regressors, i.e., trends


##### **Classic Methods II** - *Peak Over Threshold*

- What is an event? - An Event Over a Threshold
- Threshold Selection - Quantiles (90, 95, 98, 99)
- Temporal Resolution (Declustering) - Year, Season, Month, Day
- Viz - Histogram, Scatter Plot, Violin Plot, Ridge Plot
- EDA - `seaborn` simple linear regressors, i.e., trends



##### **Baseline Model II** - *Block Maximum*
- Data Likelihood - GPD: Limiting Distribution for Tails
- Baseline Model - Fully Pooled, Non-Pooled, Partially Pooled
- Metrics - PP-Plot, QQ-Plot, Joint Plot

##### **Revised Method I** - *Temporal Point Process*
- What is an event? - The Events Over a Threshold within a block of time.
- Block Maximum Temporal Resolution - Year, Season, Month, Week, Days
- Threshold Selection - Quantiles
- Theory - Point Process for Extremes
- Viz - Histogram, Scatter Plot, Violin Plot, Ridge Plot
- EDA - `seaborn` simple linear regressors, i.e., trends
- Data Likelihood - Point Process
- Baseline Model - Pooled, Non-Pooled, Partially Pooled

##### **Revised Method II** - *Marked Temporal Point Process*
- What is an event? - The Events Over a Threshold within a block of time.
- What is a mark? - The intensity of an event if it happens.
- Block Maximum Temporal Resolution - Year, Season, Month, Week, Days
- Threshold Selection - Quantiles
- Theory - Marked Decoupled Point Process
- Data Likelihood - Point Process + Marks Distribution
- Baseline Model - Pooled, Non-Pooled, Partially Pooled





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
