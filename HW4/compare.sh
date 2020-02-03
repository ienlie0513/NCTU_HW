#!/bin/bash
usage="Usage: $0 <total points> <time step>"
serial_result_file="a"
cuda_result_file="b"
diff_result_file="diff_result"
if [ $# -eq 2 ]; then
    make all
    echo "Serial version:"
    time ./serial_wave $1 $2 > $serial_result_file
    echo "CUDA version:"
    time ./0856056 $1 $2 > $cuda_result_file
    nDiff=$(diff -y --suppress-common-lines $serial_result_file $cuda_result_file | wc -l)
    echo "Numeber of different line: ${nDiff}"
    if [ $nDiff -ne 0 ]; then
        echo "Diff result store at ${diff_result_file}"
        diff $serial_result_file $cuda_result_file > $diff_result_file
    fi
    rm $serial_result_file $cuda_result_file
else
    echo $usage
    exit 1
fi
