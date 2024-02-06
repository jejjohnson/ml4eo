---
title: Geoscience Data Overview
subject: Available Datasets in Geosciences
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
keywords: data
---

A few resources for quickly accessing datasets via python.
Often times for larger experiments, we would need to download the entire dataset.
However, for quickly prototyping, it's sufficient to have a simple dataset.

***
## Geoscience Data Anatomy

* Coordinates
* Domain
* Field

***
## Data Structures

* Unstructured Data - Points
* Irregularly Structured Data - Polygons
* Regularly Structured Data - Rasters

***
## Data Types

In general, there are three different data types we will see within the geoscience community: observations, simulations, reanalysis data.

* Observations
* Simulations
* Reanalysis

***
## Data Readiness

* Level 1
* Level 2
* Level 3
* Level 4
* Level 5


***
## Data Access

[**Google Earth Engine**](https://developers.google.com/earth-engine/datasets/catalog).
This is the first most famous way to download data.
There are many known add-on packages that are useful, e.g, [`xee`](https://github.com/google/Xee/tree/main), [`eemount`](https://github.com/davemlz/eemont), and [`wxee`]().

***
[**Climate Data Store (CDS)**]()


***
[**Marine Data Store (MDS)**]()


***
[**MARS Archive**](https://www.ecmwf.int/en/forecasts/access-forecasts/access-archive-datasets)


***
[`CliMetLab`]()



***
[**WeatherBench 2**]()

There is a [guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) for downloading .
They include reanalysis datasets like ERA5, ERA5 Climatology, IFS HRES t=0 "Analysis".
They also include some forecast datasets like IFS and all of the AI-based datasets available on the [metrics](https://sites.research.google/weatherbench/) webpage.


***
## Case Studies

***
### Sea Surface Height

* Sensor Type - NADIR AlongTrack 
* L2 - MDS - [AlongTrack Satellite Data](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L3_NRT_008_044/description) (`7x7 km`, `5Hz`)
* L3 - MDS - [Interpolated Satellite Data](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046/description) (`0.25 x 0.25 deg, Daily`)
* ARGO - [ARGO Floats](https://github.com/euroargodev/argopy)

***
### Temperature

* Variable - LST/SST
* Sensor Type - Infrared
* L2 - MDS - [Multi-Sensor Fusion (ODYSSEA)](https://data.marine.copernicus.eu/product/SST_GLO_SST_L3S_NRT_OBSERVATIONS_010_010/description) (`0.1 x 0.1 deg, daily`)
* L3
    * MDS - [Multi-Sensor Fusion (ODYSSEA)](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046/description)(`0.1 x 0.1 deg, daily`)
    *  MDS - [oSTIa](https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description) (`0.05 x 0.05 deg, daily`)
* ARGO - [ARGO Floats](https://github.com/euroargodev/argopy)

***
### Precipitation

* Variable - Precipitation
* Sensor Type - Weather Station