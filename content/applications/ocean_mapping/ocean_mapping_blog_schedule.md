---
title: Ocean Mapping
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


- Data Size - Small, Medium, Large
- Linear -> NonLinear
- Deterministic, Probabilistic, Bayesian


## 0.0 - Datasets

* Fire
* Drought
* Agro
* Temperature, Precipitation


##  1.0 - Learning with Observation Data

######  **Data**:
- L2 Gappy Observations
	- Global Land Surface Atmospheric Variables - [CDS](https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-surface-land?tab=overview)
	- Global Marine Surface Meteorological Variables - [CDS](https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-observations-surface-marine?tab=overview)
	- SOCAT - [Website](https://socat.info)
- L3 Gap-Filled Observations

###### **Cases**:
* Data - L2 Obs
* Data - L2 Obs and L3 Interpolated Obs

###### **Formulation**:

$$
f: y (\Omega_y, \mathcal{T}_y ) \times \Theta \rightarrow y (\Omega_z, \mathcal{T}_z)
$$

***
### **1.1 - Discretization**

> A method that discretizes the unstructured data into a structured representation, e.g., a Cartesian, rectilinear or curvilinear grid.

**Use Case**
- Data 4 Learning -> Parameters, Interpolator
- Data 4 Estimation -> State, Latent State


***
#### 1.1.1 - Histogram

 - a - Equidistant Binning 4 Cartesian Grids - Global + Masks + Weights - [Boost-Histogram](https://boost-histogram.readthedocs.io/en/latest/index.html) | [xarray-histogram](https://github.com/Descanonge/xarray-histogram) | [dask-histogram](https://dask-histogram.readthedocs.io/en/stable/) |  [xarray](https://docs.xarray.dev/en/latest/examples/area_weighted_temperature.html) | [xcdat](https://xcdat.readthedocs.io/en/latest/examples/spatial-average.html)
- b - Adaptive Binning 4 Rectilinear Grids - KBinsDiscretizer - [sklearn tutorial](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization.html) | [tutorial](https://machinelearningmastery.com/discretization-transforms-for-machine-learning/) 
- c - Graph-Node Binning:
	- Voronoi - [Voronoi w/ Python](https://www.youtube.com/watch?v=g5FnaNtcCzU) | [Object Seg. w/ Voronoi](https://www.youtube.com/watch?v=wtrKToZNgAg) | [Semi-Discrete Flows](https://arxiv.org/abs/2203.06832)
	- K-Means - [Ocean Clustering Example](https://annefou.github.io/metos_python/05-scipy/) | [Region Joining](https://medium.com/@francode77/interpolation-using-knn-and-idw-fe546d5fb9ae)

***
#### Supp. Material

* Formulation
* Viz of neighbors and radius Neighbours - [](https://caam37830.github.io/book/08_geometry/nearestneighbor.html)
- Link between histogram and parzen window - [](https://milania.de/blog/Introduction_to_kernel_density_estimation_%28Parzen_window_method%29)
- Regression - [](https://events.asiaa.sinica.edu.tw/school/20170904/talk/chen2.pdf) 
- Gridding with geopandas - [](https://james-brennan.github.io/posts/fast_gridding_geopandas/)
* sparse + xarray + geopandas - [](https://notebooksharing.space/view/c6c1f3a7d0c260724115eaa2bf78f3738b275f7f633c1558639e7bbd75b31456#displayOptions=) 

***
### **1.2 - Non-Parametric Interpolator (Coordinate-Based)**

> A method that applies a non-parametric, coordinate-based regression algorithm to interpolate the observations based on SpatioTemporal location.

**Use Case**
- Learning - Interpolated Maps
- Estimation - Initial Conditions & Boundary Conditions 4 Data Assimilation

***
#### 1.2.1 - Naive Methods

> We will revisit the same methods used for the Discretization. 
> This will include the kernel density method and the k nearest neighbors method.

a - PyInterp Baselines - [Linear](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator), [IDW](https://pangeo-pyinterp.readthedocs.io/en/latest/generated/pyinterp.RTree.inverse_distance_weighting.html#pyinterp.RTree.inverse_distance_weighting), [RBF](https://pangeo-pyinterp.readthedocs.io/en/latest/generated/pyinterp.RTree.radial_basis_function.html#pyinterp.RTree.radial_basis_function), [Window Function](https://pangeo-pyinterp.readthedocs.io/en/latest/generated/pyinterp.RTree.window_function.html#pyinterp.RTree.window_function), [Kriging/OI/GPs](https://pangeo-pyinterp.readthedocs.io/en/latest/generated/pyinterp.RTree.universal_kriging.html#pyinterp.RTree.universal_kriging), [Splines](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html#scipy.interpolate.CloughTocher2DInterpolator)

***
#### 1.2.1a - Kernel Density Estimation

> In this section, we look at kernel density estimation as a nonparametric methodology to gap-fill unstructured observations.
> We will start with the most basic method of k-nearest neighbors.
> Then we will look at scalable alternatives like KD-Tree, Ball-Tree, or FFT.
> We’ll also look at some ways to scale it via hardware like KeOps or cuml which both use advanced methods for taking advantage of GPUs.

**Basic Methods**:
- Naive, Brute Force -  [`sklearn tutorial`](https://scikit-learn.org/stable/modules/density.html#kernel-density) | [`sklearn.neighbors.KernelDensity`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html) 

**Scaling**
* Algorithm:
	* Tree-Based - [jakevdp tutorial](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html) | [`sklearn.neighbors.KernelDensity`](http://scikit-learn.org/stable/modules/density.html) | [numba-neighbors](https://github.com/jackd/numba-neighbors)
	* Advanced Approximate NN- [`sklearn.ann`](https://github.com/frankier/sklearn-ann) | [`PyNNDescent`](https://pynndescent.readthedocs.io/en/latest/pynndescent_in_pipelines.html)
	- FFT (Equidistant) - KDEPy - [kdepy](https://kdepy.readthedocs.io/en/latest/) 
- Data Structure
	- Sparse - [sklearn.neighbors](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression)
- Hardware:
	- cuML - [KDE cuml](https://docs.rapids.ai/api/cuml/stable/api/#kernel-density-estimation)
	- KeOps - [KNN Example](https://www.kernel-operations.io/keops/_auto_tutorials/knn/plot_knn_torch.html#sphx-glr-auto-tutorials-knn-plot-knn-torch-py)

**Applied Problems**:
- KDE Regression - [kdepy example](https://kdepy.readthedocs.io/en/latest/examples.html#one-dimensional-kernel-regression) | [wiki](https://en.wikipedia.org/wiki/Kernel_regression) | [Derivation](https://faculty.washington.edu/yenchic/17Sp_403/Lec8-NPreg.pdf) | [Video Derivation](https://youtu.be/0BThC65FgOo?si=EOPIsrxmWVuGkAU9) | [Error Analysis](https://youtu.be/QBIUrvpN4RU?si=WDyx8dn_NgUZ1ySG) | [pytorch example](https://github.com/Josuelmet/Discriminative-Kalman-Filter-4.5-Python/blob/main/nw_est.py)
- Connection to Attention - [d2l.ai](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-pooling.html#attention-pooling-via-nadarayawatson-regression) | [blog](https://teddykoker.com/2020/11/performers/)
- KDE Examples with Viz - [Visualizing GeoData](https://gawron.sdsu.edu/python_for_ss/course_core/book_draft/visualization/visualizing_geographic_data.html) | [Point Pattern Analysis](https://geographicdata.science/book/notebooks/08_point_pattern_analysis.html) 

***
#### 1.2.1b - KNN Interpolation

> Here, we use k-nearest neighbors (KNN) to do interpolation.
> This is one of the simplest, most versatile algorithms available for learning.
> This is a more scalable method which uses the nearest neighbors to interpolate gappy data.
> We also showcase how we can modify the distance metric with inverse weighting or a custom distance function, e.g., Gaussian kernel.

**Basic Methods**:
* Probabilistic Interpretation - [Course](https://kuleshov-group.github.io/aml-book/contents/lecture9-density-estimation.html#outlier-detection-using-probabilistic-models)
- Naive, Brute-Force, Parallel - [`sklearn.neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor) | [`sklearn.neighbors.RadiusNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor) | [From Scratch](https://medium.com/where-quant-meets-data-science/building-k-nearest-neighbour-algorithm-from-scratch-bd0c5df13192)
* Distance - Uniform, IDW, Gaussian - [example.ipynb](https://github.com/seho0808/knn_gaussian_medium/blob/master/Medium_KNN.ipynb)

**Scaling**:
- Algorithm:
	- Tree-Based - [`sklearn.neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor) | [`sklearn.neighbors.RadiusNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor) 
- Hardware:
	- cuML + Dask - [Demo Blog](https://medium.com/rapids-ai/scaling-knn-to-new-heights-using-rapids-cuml-and-dask-63410983acfe)  | [`cuml.neighbors.KNeighborsRegressor`](https://docs.rapids.ai/api/cuml/stable/api/#nearest-neighbors-regression) 

**Example Applications**:
- Housing Interpolation w/ KNN + IDW - [Medium](https://medium.com/@francode77/interpolation-using-knn-and-idw-fe546d5fb9ae)

**Strengths:** K-nearest neighbors regression

1. is a simple, intuitive algorithm,
2. requires few assumptions about what the data must look like, and
3. works well with non-linear relationships (i.e., if the relationship is not a straight line).
4. The key merit of KNN is the quick computation time, easy interpretability, versatility to use across classification and regression problems and its non parametric nature (no need to any assumptions or data tuning)

**Weaknesses:** K-nearest neighbors regression

1. becomes very slow as the training data gets larger,
2. may not perform well with a large number of predictors, and
3. may not predict well beyond the range of values input in your training data.
4. In the KNN algorithm, for every new test data point, we need to find its distance to all of the training data points. This is quite hectic when we have a large data with several features. To solve this issue we can use some KNN extension methods like KD tree. I will discuss more on this in later blog posts.
- KNN is also sensitive to irrelevant features but this issue can be addressed by feature selection. A possible solution is to perform PCA on the data and just chose the principal features for the KNN analysis.
- KNN also needs to store all of the training data and this is can be quite costly in case of large data sets.
***

#### 1.2.2 - GPs/OI/Kriging

> This will feature tutorials to build up our GP/OI/Kriging mathematical proficiency. We will start by start by We will also look at some specific terminology, e.g., length scale vs lag

***
**Applications**
* Data Assimilation - DA Window + [LOWESS](https://x.com/daansan_ml/status/1740311776397512889?s=61&t=ULkzymf_k6remRZLKB0jtw)

> We will use the LOWESS method to do interpolation on a subset of spatiotemporal data. We will look at 3 data types: 
> 1. sea surface height with very sparse structured randomness
> 2. Sea surface temperature - dense structured randomness
> 3. Land Temperature Data - 
> 4. 

**Software**
* Optimal Interpolation 4 Data Assimilation (OI4DA) - package + xarray interface + sklearn column transforms


***

**From Scratch**
- a - GP From Scratch - JAX +  [Cola](https://cola.readthedocs.io/en/stable/notebooks/03_GPs.html)  - [Demo NB](https://github.com/jejjohnson/uncertain_gps/tree/master/notebooks)
- b - GP w/ Libs - JAX + [TinyGP](https://tinygp.readthedocs.io/en/latest/index.html) + [Bayesian Inference](https://arxiv.org/abs/1912.13440) ([Demo NBs](https://github.com/jejjohnson/research_notebook/tree/main/code/jax/notebooks/egp/exact))
- c - GP w/ PPLs - JAX + [Cola](https://cola.readthedocs.io/en/stable/notebooks/03_GPs.html) + [Numpyro](https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/continuous.py#L1422) 
- d - Customizing GP w/ PPLs - [Custom TFP Distribution](https://github.com/JaxGaussianProcesses/GPJax/blob/main/gpjax/distributions.py#L79)  | [Custom Numpyro Distribution](https://github.com/dfm/tinygp/blob/main/src/tinygp/numpyro_support.py)

**Canonical Example**
- Mauna Loa - [Part I](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MaunaLoa.html) | [Part II](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-MaunaLoa2.html)

**Scaling**
- d - Kernel Matrix Approximations - [`sklearn.kernel_approximation`](http://scikit-learn.org/stable/modules/kernel_approximation.html) | [My kernellib](https://github.com/jejjohnson/kernellib/tree/master/kernellib/regression)
- e - Hardware -  [KeOps](https://www.kernel-operations.io/keops/_auto_tutorials/interpolation/plot_RBF_interpolation_torch.html#sphx-glr-auto-tutorials-interpolation-plot-rbf-interpolation-torch-py) | [KeOPs + GPyTorch](https://www.kernel-operations.io/keops/_auto_tutorials/backends/plot_gpytorch.html#sphx-glr-auto-tutorials-backends-plot-gpytorch-py)

**Appendix**
- `jax` + kernel functions + `jax.vmap`
	- Distances - [scipy overview](https://caam37830.github.io/book/08_geometry/distances.html) | [jax demo](https://github.com/jejjohnson/uncertain_gps/blob/master/notebooks/distances.ipynb)
	- Kernel Matrices -  [jax demo](https://github.com/jejjohnson/uncertain_gps/blob/master/notebooks/distances.ipynb)
	- Kernel Matrix Derivatives - [jax demo](https://github.com/jejjohnson/uncertain_gps/blob/master/notebooks/kernel_derivatives.ipynb)

***
#### 1.2.3 - Improved GPs - Moment-Based

- a - Sparse GPs w/ PPLs - [My Jax Code](https://github.com/jejjohnson/research_notebook/blob/main/code/jax/lib/egp/sparse.py) + [Bayesian Inference](https://arxiv.org/abs/2211.02476)
- b - SVGPs w/ PPLs - GPJax | Pyro-PPL | GPyTorch
- c - Structured GPs -  SKI/SKIP ([Precious Work](https://github.com/jejjohnson/gps4oi/tree/e3e42b66d0b246c91b36d8e984564dd25b429e9c), [Example](https://github.com/sebascuri/rllib/tree/4d4e7524c61f11980b197874d9a83a530ad0dcec/rllib/util/gaussian_processes))
- d - Deep Kernel Learning - [DUE](https://github.com/y0ast/DUE/blob/f29c990811fd6a8e76215f17049e6952ef5ea0c9/due/dkl.py)  | GPyTorch | Pyro-PPL

***
#### 1.2.4 - Improved GPs - Basis Functions

- Fourier Features GP - [RFF](https://github.com/tiskw/random-fourier-features) | [PyRFF](https://github.com/michaelosthege/pyrff) | [GPyTorch](https://github.com/jejjohnson/gps4oi/blob/main/kernellib/models/rff.py) 
- Spherical Harmonics GPs (SHGPs) - [GPfY](https://github.com/stefanosele/GPfY) | [SphericalHarmonics](https://github.com/vdutor/SphericalHarmonics) | [Torch-Harmonics](https://github.com/NVIDIA/torch-harmonics) | [LocationEncoder](https://github.com/MarcCoru/locationencoder) | [kNerF List](https://github.com/jejjohnson/knerf/issues/4) 
- Sparse SHGPs -  [GPfY](https://github.com/stefanosele/GPfY)

***
#### 1.2.5 - State Space Gaussian Processes

> In this improvement, we add the Markovian assumption which improves the scalability.
> See this [video](https://youtu.be/kN4aiAlDtb0?si=tIDa27njFUIN027R) for a better introduction.

- Markovian GPs (MGPs) - [BayesNewton](https://github.com/AaltoML/BayesNewton) | [MarkovFlow](https://github.com/secondmind-labs/markovflow) | [Dynamax](https://github.com/probml/dynamax) 
- Sparse MGPs

***
### **1.3 - Parametric Interpolator (Coordinate-Based)**

> Learns a parametric, coordinate-based, Differentiable Interpolator for fast queries and online training.

###### **Use Case**
- Learning - Compressed Representation, Online Learning
- Estimate - Fast Queries, Online Estimation

###### Formulation
$$
y(s,t) = f(s,t;\theta)
$$

###### **Algorithms**
- Baseline - SIREN
- Improvements - SpatioTemporal Encoders
- Research - Physics Informed, Modulated, Scalable, Stochastic

- a - SIREN
- b - spatial coordinate encoders
- c - temporal coordinate encoders
- d - modulation

Scale
- Hashing

Background - TimeEmbedding, SpatialEmbeddings

***
### **1.4 - Parametric SpatioTemporal Field Interpolator (Field-Based)**

> These methods are parametric interpolators.
> They directly operate on the gappy fields and output a gap-free field.
> They are parametric which implies that they will use neural networks to some degree.
> Because it’s space and time, we will need physics inspired architectures which decompose the field into a spatial operator and TimeStepper.
> For example, for the spatial operator, we will use architectures like convolutions, transformers or graphs.
> For the TimeStepper, we can use convolutions, recurrent neural networks, transformers, or graphs.

$$
y(\Omega_u, t) = f(\Omega_y, t, \theta)
$$

**Use Cases:**
- Learning - Fast, Compressed Interpolator, ROM, PnP Priors, Anomaly Detectors, Pretraining 4 DA
- Estimation - Latent Variable Data Assimilation

**Algorithms**
- Baseline: (Spectral) Conv, UNet, DINEOF, [Convolutional Neural Operator](https://github.com/bogdanraonic3/ConvolutionalNeuralOperator)
- Improved: Deep Equilibrium Models
- Research: Transformers, Graphical Neural Networks

***
##### 1.4.1 - **Direct CNN Models**

> We apply some simple NN models that are specifically designed to deal with masked inputs. 
> We’re dealing with spatiotemporal data, we will directly apply convolutions. 
> We can increase the difficulty by applying Convolutional LSTMs which is a popular architecture for spatiotemporal data.
> To deal with the missing data, we’ll start with some simple ad-hoc masks techniques which is similar the kernel methods.
> We’ll do more advanced methods like partial convolutions which are compatible with neural networks.


- a - Convolutions w/ Masks -  [astropy](https://docs.astropy.org/en/stable/convolution/) | [serket](https://serket.readthedocs.io/API/convolution.html#serket.nn.conv_nd) 
- b - Partial Convolutions - [keras - partial conv](https://github.com/MathiasGruber/PConv-Keras) | [NVidia](https://github.com/NVIDIA/partialconv) 
- c - Partial Convolution + TimeStepper -  LSTMs -  [PConvLSTM](https://ieeexplore.ieee.org/abstract/document/9187792)
- Appendix - Masked Losses, Interpolation Losses, Convolution Family, RNN/GRU/LSTMs

***
##### 1.4.2 - Direct Transformer Models

> Here, we will use more advanced models called transformers.
> We look at the same task of dealing with missing values.
> However, we can use patch Embeddings to deal with missing data.


- a - Masked AutoEncoder - [keras](https://keras.io/examples/vision/masked_image_modeling/) | [keras](https://github.com/three0-s/MAE-keras) |  [SST](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-1385/) | [SatMAE](https://sustainlab-group.github.io/SatMAE/)
- b - SpatioTemporal Masked AutoEncoder - [keras](https://github.com/innat/VideoMAE)
- Appendix - Transformer, Attention, UNet, AE, PatchEmbedding Masks, Time Embeddings

***
##### 1.4.3 - Graphical Models

> We will look at Graphical Models as a different data structure for dealing with spatiotemporal data.

- [Neural Spatiotemporal Forecastinf (PyTorch)](https://github.com/TorchSpatiotemporal/tsl) | [Sparse GNN](https://github.com/Graph-Machine-Learning-Group/spin)
* Appendix - GNN

***
##### 1.4.4 - Deep Equilibrium Models

> We will add an extra



- a - DEQ from Scratch - [Implicit Layers Tutorial](http://implicit-layers-tutorial.org)
- b - [jaxopt](https://jaxopt.github.io/stable/)
- c - [Optimistix](https://github.com/patrick-kidger/optimistix)

***
##### 1.4.5 - Conditional Flow Models

> Here, we will use conditional flow models.
> These are conditional stochastic models.
> They include architectures such as bijective, Surjective, or stochastic.
> The nice thing here is that we can reuse some of the previous architectures, e.g., the Conv, the partial convolutions, and/or the transformers.

- Variational AutoEncoder + Masks - [pyro-ppl](http://pyro.ai/examples/cvae.html)
- [PriorCVAE](https://github.com/elizavetasemenova/PriorCVAE)
- Stochastic Interpolants -  [Video](https://youtu.be/V2XL7d3DKVk?si=LyBxpW6yS-gYysDM) | [Video](https://youtu.be/cejbXob8rvE?si=gK3gwj6HgX8EaJku) | [Conditional Flow Matching](https://github.com/atong01/conditional-flow-matching) | [Stochastic Interpolants](https://github.com/malbergo/stochastic-interpolants) 

***
### **1.5 - Parametric Dynamical Model (Field-Based)**

> In this application, we train a dynamical model that best fit the observations. The model complexity ranges from linear to nonlinear. The physics can range from a PDE to a surrogate model.

###### **Use Cases**:
- Learning - Scientific Discovery, Surrogate Model
- Estimation - Latent Variable Data Assimilation

###### Formulation

$$
\begin{aligned}
z(\Omega_z, t) &= f[z;\theta](\Omega_z,t-\delta t) \\
y(\Omega_y,t) &= h[z;\theta](\Omega_z,t)
\end{aligned}
$$

###### **Algorithms**
- Baseline: Kalman Filter Family
- Improved: PDE, Neural ODE, UDE
- Research: Deep Markov Model

***

#### **1.5.1 - Learning Spatial Operators**

> Look at this from a Spatiotemporal decomposition perspective. 
> We go over the basics of a state space model including the dynamical (transition) model and the observation (emission) model. 
> We then talk about the complexity of the system.
> In the case of observations only, we keep it simple with a masked.
> We will use a simple TimeStepper for all models, e.g., we can use a “continuous” time stepper like a traditional ODESolver or a “discrete” time stepper like Euler.


- Universal Differential Equations (UDE) - Framework
- a - Linear Spatial Operator
- b - Convolutional (Finite Difference) Spatial Operator
- c - Spectral Convolutional Spatial Operator

Appendix
- Faster Neural ODEs - [](https://github.com/a-norcliffe/torch_gq_adjoint)
- Gradients - FD, AutoDiff., Adjoint/ Implicit Diff.

***
#### 1.5.2 - Probabilistic Dynamical Models

> In this section, we will look at how we can perform inference with time series. 
> This will be useful for Reanalysis and Forecasting.
> A great introduction can be found [here](https://youtu.be/N4AgbWrJHc4?si=1wMnT7jICzzX77dS)


***
##### 1.5.2a - Conjugate Inference

> Basically using conjugate priors and linear models will magically give us exact inference.


- a - Linear Model + Exact Inference
	* Linear Kalman Filter -  [Diffrax Example](https://docs.kidger.site/diffrax/examples/kalman_filter/) [Dynamax](https://github.com/probml/dynamax)[Simple KF](https://github.com/ptandeo/Kalman) | [Neural KF](https://arxiv.org/abs/2102.10021) | [KalmanNet](https://arxiv.org/abs/2107.10043) | [Training](https://arxiv.org/abs/2012.14313)

***
##### 1.5.2b - Parametric Inference

> a.k.a. Deterministic Approximate Inference.
> This is a local approximation whereby we cover one mode of the potentially complex, multi-modal distribution really well.
> We approximate the posterior with a simpler distribution, $q(\theta;\alpha)$
> These include staples like MLE, MAP, Laplace Approx, VI, and EP.

- Non-Linear Model + Deterministic Approximate Inference
	- Standard Approaches - EKF, UKF, ADF - [Dynamax](https://github.com/probml/dynamax) | [Neural EKF](https://arxiv.org/abs/2210.04165) | [Training](https://arxiv.org/abs/2012.14313)
	- Approximate Expectation Propagation - [](https://www.sotakao.com/blog/2022-approx-ep-smoother/) [](https://github.com/jejjohnson/research_notebook/tree/main/code/jax/lib/egp/uncertain)
	- Variational Approximate Inference - [Slides](https://www.ecmwf.int/sites/default/files/elibrary/2012/14002-stable-and-accurate-variational-kalman-filter.pdf)
	- Unified - [Bayes-Newton](https://arxiv.org/abs/2111.01721)

***
##### 1.5.2c - Stochastic Inference

>  a.k.a. Stochastic Approximate Inference
>  We draw samples from the posterior.
>  This includes staples like MCMC, HMC/NUTS, SGLD, Gibbs, ESS


Non-Linear Model + Stochastic Approximate Inference
- Ensemble Kalman Filter - [](https://github.com/ir-lab/DEnKF) [](https://github.com/ymchen0/torchEnKF) [](https://github.com/mchoblet/ensemblefilters)  [](https://github.com/ir-lab/DEnKF) [](https://github.com/ymchen0/ROAD-EnKF) 
- Particle Filter - [pfilter](https://github.com/johnhw/pfilter) | [pc - tutorial](https://github.com/jelfring/particle-filter-tutorial)

*** 

Appendix 
- Sequential Model Inference - Exact, (V)EM, (V)EP, 
- Packages - [Nested Sampling](https://github.com/Joshuaalbert/jaxns) | [SGMCMC](https://sgmcmcjax.readthedocs.io/en/latest/all_samplers.html) | [BlackJax](https://blackjax-devs.github.io/blackjax/index.html)

***
#### 1.5.3 - Latent Probabilistic Dynamical Models

> We look at state space models in general starting with linear models.

- a - Conjugate Transform  (Conditional Markov Flows)
	- Exact Inference  - [Code](https://github.com/johannaSommer/KF_irreg_TS) | [Paper](https://arxiv.org/abs/2310.10976)
- b - Stochastic Transform Filter 
	- Stochastic Inference -  [ROAD-EnsKF](https://github.com/ymchen0/ROAD-EnKF) 
	- Variational Inference - [pyro - DMM](http://pyro.ai/examples/dmm.html) | [numpyro - DMM](https://num.pyro.ai/en/latest/examples/stein_dmm.html) | [DMM](https://github.com/yjlolo/pytorch-deep-markov-model) | [PgDMM](https://github.com/liouvill/PgDMM)
- observation operator encoder - [KVAE](https://github.com/ngunnar/med-dyn-reg) 
- c - 
- d - Neural SDE

***


## 2.0 - Observations to Reanalysis

**Data**:
- L2 Gappy Observations
- L3 Gap-Filled Observations
- L4 Reanalysis

**Cases**:
* Data - L2 Obs
* Data - L2 Obs and L3 Interpolated Obs
* Data - L2 Obs, L3 Interpolated Obs, L4 Reanalysis

###### Formulation
$$
f: y (\Omega_z, \mathcal{T}_z ) \times 
u_b (\Omega_z, \mathcal{T}) \times
\Theta \rightarrow u_a (\Omega_z, \mathcal{T}_z)
$$

**Ideas**:
- Sequential DA, Variational DA, Amortized DA
- Dynamical Model - Physical, Hybrid, Surrogate
- Bi-Level Optimization
- Dynamical Inference - MLE, MAP, Variational, Laplace, EM, VEM
- Amortized Model - Direct, DEQ

***

- Bilevel Optimization
- Plug n Play Prior

***
### Physics

- [ClimateLab](https://brian-rose.github.io/ClimateLaboratoryBook/home.html)
- [Ocean Functions](https://oceanspy.readthedocs.io/en/latest/api.html)

***
### 2.1 - Parametric Dynamical Model (Field-Based)

###### **Use Cases**
- Estimation - Reanalysis
- Learning - Physical Models


###### Formulation

$$
\begin{aligned}
z(\Omega_z, t) &= f[z;\theta](\Omega_z,t-\delta_t) \\
y(\Omega_y,t) &= h[z;\theta](\Omega_z,t) \\
f(z,t) &= \alpha f_{dyn}(z,t) + \beta f_{param}(z,t)
\end{aligned}
$$

###### **Algorithms**
- 
- Baseline - Parametric Dynamical Model + 3D/4DVar + BiLevel Optimization
- Improved - Hybrid Dynamical Model - 3D/4DVar + VI
- Research - LatentVar


***
### 2.2 - Amortized Parametric Model

###### Use Cases
- Learning - Surrogate Modeling, Surrogate Reanalysis

###### Formulation


$$
u_a = f(u_b, y_{obs})
$$

###### Algorithms
- Baseline - Deep Equilibrium Model




***
## 3.0 - Reanalysis to X-Casting


**Data**:
- L2 Gappy Observations
- L3 Gap-Filled Observations
- L4 Reanalysis

**Cases**:
* Data - L2 Obs
* Data - L2 Obs and L3 Interpolated Obs
* Data - L2 Obs, L3 Interpolated Obs, L4 Reanalysis

Use Cases:
- NowCasting
- ForeCasting
- Projections


###### Formulation

$$
f: u_a (\Omega_z,\mathcal{T}) \times \delta_t \times
\Theta \rightarrow u_a (\Omega_z, \mathcal{T}_z+\delta_t)
$$


***

#### Parametric Surrogate Model

Algorithms
- Baseline: Spectral Conv, UNet, 
- Improvements: GNN, Transformer

Ideas:
- Bilevel Optimization




