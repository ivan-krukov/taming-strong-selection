# Taming strong selection with large sample sizes

This repository contains the code and manuscript source.

## Requirements

The core code relies on `numpy`, `scipy`, and `cython` to perform the transition matrix
construction.

As an additional dependency for generating comparison figures, we require
[`moments`](https://bitbucket.org/simongravel/moments/).

## `LaTeX` compiltation

The pdf for the manuscript can be built with:

```
make taming-strong-selection.pdf
```

## Creating figures

All the figures in the paper are available in `fig/`.
The figures can be re-generated via several python and R scripts, automated through `make`.
Note that this will first have to generate a number of tables, which might take a little time.

```
make figures
```

## Computing transition matrices

### Computing the rectangular T matrices

To compute the transition matrices discussed in the article, we first compute the $(n_o+1)\times(n_p+1)$ matrices $T$, 
defined in equation (6). This is the most computationally intensive part of the calculation, and can be done using 
the c program transition_probability_explicit.c: 

`./transition_probability_explicit $N $Ns $no $k $j > "t_mat_${N}_${Ns}_${no}_${k}_${j}.txt"`

Here `N` is the population size, `Ns` the scaled selection coefficient, `no` the sample size, `k` the maximum number of 
failures allowed per sampled ploid, and `j` is the order of the jackknife to be considered. Note 
transition_probability_explicit.c does not compute the jackknife approximation itself, but computes transition matrices 
T up to order $(n_o+1)\times(n_o+1+j)$. All the T matrices are then stored in file t_mat_N_Ns_no_k_j.txt

### Computing the square Q matrices

Once the T matrices are computed, we can compute the Q matrices using equation 8. This is implemented in 
read_matrices.py

`cat data/t_mat_${N}_${Ns}_200_3_10.txt | python src/read_matrices.py -j 5 -n 200 data/q_mat_${N}_${Ns}_${no}_${k}_${j}.txt`.


