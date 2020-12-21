# Taming strong selection with large sample sizes

This repository contains the code and manuscript source.

## Requirements

The core code relies on `numpy`, `scipy`, and `numba` to perform the transition matrix
construction.

As an additional dependency for generating comparison figures, we require
[`moments`](https://bitbucket.org/simongravel/moments/).

## Creating figures

The figures are generated via several python scripts, automated through `make`.
Note that this will first have to generate a number of tables, which might take a little time.

```
make figures
```

## `LaTeX` compiltation

The pdf for the manuscript can be built with:

```
make taming-strong-selection.pdf
```
