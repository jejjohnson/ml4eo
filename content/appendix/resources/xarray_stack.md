---
title: XArray Stack
subject: Available Datasets in Geosciences
short_title: XArray Stack
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


## Remote Sensing

### [`rioxarray`]()

### [`regionmask`]()

### [`GeoWombat`](https://github.com/jgrss/geowombat)

***
## Geoscience

### [`xcdat`]()


***
## Linear Algebra

### [`xarray_einstats`](https://einstats.python.arviz.org/en/latest/tutorials/einops-basics-port.html)

**General Array Operations**

```python
ds: XRDataset["T X Y"] = ...
# combine dimensions
ds: XRDataset["N"] = rearange(ds, "(T X Y)=N")
ds: XRDataset["N T"] = rearange(ds, "(X Y)=Samples")
# Creating Patches
ds: XRDataset["N"] = rearange(ds, "T X Y -> ")
# aggregate dimensions
# options: mean, min, max, sum, prod
ds: XRDataset["T"] = reduce(ds, "X Y", "mean")
ds: XRDataset["X Y"] = reduce(ds, "T", "mean")
# pooling (max, average)
```

**Linear Algebra** (TODO)