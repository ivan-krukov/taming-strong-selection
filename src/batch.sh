# Example invocation:                                                                                                                        
# for N in 2000 1000 200; do
#   for Ns in 0 1 10 50; do
#     qsub -v N=$N Ns=$Ns batch.sh
#   done
# done

# then convest these into Q matrices
# for N in 2000 1000 200; do for Ns in 0 1 10 50; do cat data/t_mat_${N}_${Ns}_200_3_10.txt | python src/read_matrices.py -j 5 -n 200 data/q_mat_${N}_${Ns}_200_3_5.txt; done; done

# The first two arguments are specified through qsub                                                                                         
# N=$1  # population size                                                                                                                    
# Ns=$2 # selection                                                                                                                          
no=200  # sample size                                                                                                                        
j=10    # jackknife                                                                                                                          
k=3     # max number of resamples (maximum selection failures)                                                                               

cd $PBS_O_WORKDIR
./transition_probability_explicit $N $Ns $no $k $j > "t_mat_${N}_${Ns}_${no}_${k}_${j}.txt"
