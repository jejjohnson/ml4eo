---
title: Geoscience Data Overview
subject: Available Datasets in Geosciences
short_title: Data Overview
authors:
  - name: J. Emmanuel Johnson
    affiliations:
      - CSIC
      - UCM
      - IGEO
    orcid: 0000-0002-6739-0053
    email: juanjohn@ucm.es
license: CC-BY-4.0
keywords: simulations
abbreviations:
    ERA5: ECMWF Reanalysis Version 5
    CMIP6: Coupled Model Intercomparison Project Phase 6
    AMIP6: Atmospherical Model Intercomparison Project Phase 6
    PDEs: Partial Differential Equations
    RHS: Right Hand Side
    TLDR: Too Long Did Not Read
    SSP: Shared Socioeconomic Pathways
    CDS: Climate Data Store
---

A few resources for quickly accessing datasets via python.
Often times for larger experiments, we would need to download the entire dataset.
However, for quickly prototyping, it's sufficient to have a simple dataset.

***
## Data Structures

***
### Unstructured Data

***
### Irregularly Structured Data

***
### Regularly Structured Data

***
## Data Types

In general, there are three different data types we will see within the geoscience community: observations, simulations, reanalysis data.

***
### Observations

***
### Simulations

***
### Reanalysis


***
## Data Access

### [**Google Earth Engine**](https://developers.google.com/earth-engine/datasets/catalog)

***
#### [`xee`](https://github.com/google/Xee/tree/main)


```python
import ee
import xarray as xr
# initialise (no need for a log-in!)
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
# initialise image collection
ic = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")    # ERA5 Reanalysis
ic = ee.ImageCollection("NASA/GDDP-CMIP6")           # CMIP6 Simulations
# filter for specific date
ic = ic.filterDate('1992-10-05', '1993-03-31')
# define geometry
geometry = ee.Geometry.Rectangle(113.33, -43.63, 153.56, -10.66)
projection = ic.first.select(0).projection()
scale = 0.25 # km
crs = "EPSG:4326" # Coordinate Reference system
# open dataset
ds = xr.open_dataset(
    ic, 
    engine='ee',
    projection=projection,
    crs=crs, scale=0.25, geometry=geometry
)

```

---
#### [`eemount`](https://github.com/davemlz/eemont)

---
#### [`wxee`]()

***
### [**Climate Data Store (CDS)**]()


### [**Marine Data Store (MDS)**]()


### [**MARS Archive**](https://www.ecmwf.int/en/forecasts/access-forecasts/access-archive-datasets)









---
## [`CliMetLab`]()


```python

# define data source
data_source = "ecmwf-open-data" # "cds" | "mars"
# define dataset name
dataset_name = "reanalysis-era5-single-levels"
# define parameters
param = ["2t", "msl"]
param = {"param1": "val1"}
# ==========
# define domain
domain = "France" # "Spain"
area = [lon_min, lon_max, lat_min, lat_max]
# define period
period = (1991, 2001)
date = ["2012-12-12", "2012-12-13"],         # "2012-12-12"
time = [600, 1200, 1800],                    # "12:00" | 12
# define output format
format = "netcdf",                           # "grib" | "odb"
# load source
source = cml.load_source(
   "cds",
   dataset_name,
   param=param,
   product_type="reanalysis",
   grid='5/5',
   area=area, domain=domain,
   date=date, period=period, time=time,
   format=format,
)
# convert to xarray
ds: xr.Dataset = source.to_xarray()
df: pd.DataFrame = source.to_dataframe()
```


***
#### [**WeatherBench 2**]()

There is a [guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) for downloading .
They include reanalysis datasets like ERA5, ERA5 Climatology, IFS HRES t=0 "Analysis".
They also include some forecast datasets like IFS and all of the AI-based datasets available on the [metrics](https://sites.research.google/weatherbench/) webpage.