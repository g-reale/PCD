for ((i=1; i<=4; i++)); do
    poscript=$((2 ** i))
    ./main 0 2000 2000 ${poscript} 500 0.1 0.01 1.0 > "data_${poscript}.csv"
done