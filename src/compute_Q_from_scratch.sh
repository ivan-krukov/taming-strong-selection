# A small example computation of transition matrices

no=30   # sample size
j=10    # jackknife
k=3     # max number of resamples (maximum selection failures)
Ns=1 #
N=100 # A small population for this small example

#cd $PBS_O_WORKDIR

# Note that you will need to compile transition_probability_explicit.c before running running this!
echo "Computing T matrices"
./transition_probability_explicit $N $Ns $no $k $j > "../data/t_mat_${N}_${Ns}_${no}_${k}_${j}.txt"
echo "saved to data/t_mat_${N}_${Ns}_${no}_${k}_${j}.txt"
echo "Computing Q matrices"
cat ../data/t_mat_${N}_${Ns}_${no}_${k}_${j}.txt | python ./read_matrices.py -j $j -n $no ../data/q_mat_${N}_${Ns}_${no}_${k}_${j}.txt
echo "saved to data/q_mat_${N}_${Ns}_${no}_${k}_${j}.txt"