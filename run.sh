for ((i=1; i<=6; i++)); do
    mpirun main 2000 2000 2 500 0.1 0.01 1.0 -np ${i} > "data_${i}.csv"
done