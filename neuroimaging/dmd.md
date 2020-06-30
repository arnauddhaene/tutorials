{% include lib/mathjax.html %}

# Dynamic Mode Decomposition

## Introduction

### What is dynamic mode Decomposition?

### How can we use it for neuroimaging data?

## Manual Dynamic Mode Decomposition computation

### Input data format

The format of the input data should be a 2-dimensional matrix having `N` rows, corresponding to the number of regions of interest—or nodes—and `T` columns, corresponding to the number of sampled timepoints.

As an example, let us load some test data from the nidmd Python package, downloaded from its GitHub source.


``` python
import scipy.io as sio

data = sio.loadmat('nidmd/nidmd/tests/data/glasser.mat')['TCSnf']
data.shape
```

```
(360, 1190)
```

We can observe that our sample data contains 360 nodes (which are corresponding regions of interest of the Glasser cortical parcellation) and 1190 timepoints.

### Normalization

Before getting to the actual decomposition step, it is important to normalize our data with respect to each timepoint.

``` python
avg = np.mean(data, axis=1)  # mean
std = np.std(data, axis=1)  # standard deviation

shape = (mean.shape[0], 1)  # normalize columns

data -= avg.reshape(shape)  # make mean 0
data /= std.reshape(shape)  # make standard deviation 1
```

### Autoregressive model

As explicited in Casorso et al. [[2](#references)], the dynamic mode decomposition uses a 1st-order autoregressive model.

\[
x_t = A \cdot x_{t-1} + \epsilon_t, \ \ \ \ \ \ \ \forall t \in [2, ..., T]
\]

where \(x_t\) of length \(N\) represents the fMRI time series at time
t,



### Eigendecomposition


## nidmd for Python

### Import necessary libraries

For this task, here are the useful libraries we can import:

``` python
import nidmd as nd
import numpy as np
import pandas as pd
```

### Loading data and creation of a TimeSeries instance

To load time-series fMRI data into the `nidmd` framework, we have the choice to use pre-existing cortical parcellation information or not. Currently, the nidmd package supports two different cortical parcellations: Glasser, having 360 regions of interest [[3](#references)], and Schaefer, which has 400 regions of interest [[4](#references)].

We can load data into a TimeSeries (or a Decomposition, which includes the parcellation info, and is simply a child of TimeSeries) using two methods:

* Indicate the `.mat` or `.csv` files where our data is stored.
* Handle import from file manually, and give the resulting Array-like object to nidmd.

For instance, we can use data included in the test data of the nidmd project as follows

``` python
file = '../nidmd/tests/data/glasser.csv'

dcp = nd.Decomposition(filenames=file)  # Using filename

matrix = np.genfromtxt(file, delimiter=',')
dcp = Decomposition(data=matrix)  # Using manual import
```





{% include test.html %}



## References

[1] <a href="https://doi.org/10.3389/fninf.2014.00014" target="_blank">Abraham, A., Pedregosa, F., Eickenberg, M., Gervais, P., Mueller, A., Kossaifi, J., Gramfort, A., Thirion, B., Varoquaux, G., 2014. Machine learning for neuroimaging with scikit-learn. Front. Neuroinform. 8.</a>

[2] <a href="https://doi.org/10.1016/j.neuroimage.2019.03.019" target="_blank">Casorso, J., Kong, X., Chi, W., Van De Ville, D., Yeo, B.T.T., Liégeois, R., 2019. Dynamic mode decomposition of resting-state and task fMRI. NeuroImage 194, 42–54.</a>

[3] <a href="https://doi.org/10.1038/nature18933" target="_blank">Glasser, M.F., Coalson, T.S., Robinson, E.C., Hacker, C.D., Harwell, J., Yacoub, E., Ugurbil, K., Andersson, J., Beckmann, C.F., Jenkinson, M., Smith, S.M., Van Essen, D.C., 2016. A multi-modal parcellation of human cerebral cortex. Nature 536, 171–178.</a>

[4] <a href="https://doi.org/10.1093/cercor/bhx179" target="_blank">Schaefer, A., Kong, R., Gordon, E.M., Laumann, T.O., Zuo, X.-N., Holmes, A.J., Eickhoff, S.B., Yeo, B.T.T., 2018. Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. Cerebral Cortex 28, 3095–3114.</a>
