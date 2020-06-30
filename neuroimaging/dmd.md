{% include lib/mathjax.html %}

# Dynamic Mode Decomposition

## Introduction

### What is dynamic mode Decomposition?

### How can we use it for neuroimaging data?

## Manual Dynamic Mode Decomposition computation

### Input data format

The format of the input data should be a 2-dimensional matrix having $$N$$ rows, corresponding to the number of regions of interest—or nodes—and $$T$$ columns, corresponding to the number of sampled timepoints.

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

$$
x_t = A \cdot x_{t-1} + \epsilon_t, \ \ \ \ \ \ \ \forall t \in [2, ..., T]
$$

where $$x_t$$ of length $$N$$ represents the fMRI time series at time $$t$$ , matrix $$A$$ of size $$N \cdot N$$ is the model parameter that encodes the linear relationship between successive time points, $$\epsilon_t$$ are the residuals of the model, and $$T$$ is the number of time points.

The matrix A is computed by solving the following equation

$$
\min_A \sum_{t=2}^{T} || x_t - A \cdot x_{t-1} ||^2
$$

The optimal solution is

$$
A = XY^T(YY^T)^{-1} \ \ \ \text{with} \ \ \ X = [x_2, ... x_T], \ \ Y = [x_1, ... x_{T-1}]
$$

Jumping back into Python, this yields

``` python
x = data[:, 1:]
y = data[:, :-1]
a = (x @ y.T) @ np.linalg.inv(y @ y.T)
```

### Eigendecomposition

The eigendecomposition of A gives rise to the following notation:

$$
A = S \Lambda S^{-1}
$$

Here, the columns of $$S$$ are the eigenvectors of A, with the diagonal matrix $$\Lambda$$ containing its corresponding eigenvalues. These values can be symmetric sine $$A$$ is not symmetric and real.

This eigendecomposition allows for the formulation of a _dynamic system_ as the suum of __linearly__ decoupled modes—here called dynamic modes.

The temporal characteristics of each mode are its damping time and period:

$$
\Delta_i = \frac{-1}{\log \lambda_i}, \ \ \ \ \ \ T_i = \frac{2\pi}{\arg \lambda_i}
$$

For the spatial characteristics, the eigenvector of each mode shows the relationship between different nodes—which are __regions of interest__ in the neuroimaging case.

Back to Python, we have

``` python
eig_val, eig_vec = np.linalg.eig(a)

eig_idx = np.abs(eig_val).argsort()[::-1]  # descending index
```

where `eig_idx` can be used to sort the eigenvalues and eigenvectors in the descending manner with respect to the absolute value.

We must not forget to adjust the phase of the eigenvectors in order to assure their orthogonality

``` python
eig_vec = adjust_phase(eig_vec)
```

With the  `adjust_phase` method defined as follows

``` python
def adjust_phase(x):
  """
  Adjust phase of matrix for orthogonalization of columns.
  Parameters
  ----------
  x : Array-like
      data matrix
  Returns
  -------
  ox : Array-like
      data matrix with orthogonalized columns
  """

  x = np.asarray(x)
  assert isinstance(x, np.ndarray)

  # create empty instance for ox
  ox = np.empty(shape=x.shape, dtype=complex)

  for j in range(x.shape[1]):

      # seperate real and imaginary parts
      a = np.real(x[:, j])
      b = np.imag(x[:, j])

      # phase calculation
      phi = 0.5 * np.arctan(2 * (a @ b) / (b.T @ b - a.T @ a))

      # compute normalised a, b
      anorm = np.linalg.norm(np.cos(phi) * a - np.sin(phi) * b)
      bnorm = np.linalg.norm(np.sin(phi) * a + np.cos(phi) * b)

      if bnorm > anorm:
          if phi < 0:
              phi -= np.pi / 2
          else:
              phi += np.pi / 2

      adjed = np.multiply(x[:, j], cmath.exp(complex(0, 1) * phi))
      ox[:, j] = adjed if np.mean(adjed) >= 0 else -1 * adjed

  return ox
```


## nidmd for Python

Was the above process a bit too lengthy for your taste?

You're in luck! A Python package called `nidmd` handles all the mathematical aspects of what was covered in the previous part. Moreover, it allows you to plot your results visually! Let's have a look.

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
