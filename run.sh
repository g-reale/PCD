for ((i=1; i<=5; i++)); do
    poscript=$((2 ** i))
    ./main 0 5000 5000 ${poscript} 500 0.1 0.01 1.0 > "data_${poscript}.csv"
done