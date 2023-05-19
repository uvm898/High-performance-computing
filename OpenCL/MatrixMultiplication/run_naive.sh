#!/bin/bash
gcc -w -o naive_solution naive_solution.c -lOpenCL -lm
echo "start 1/15"
./naive_solution 128 128 128 1000 >> log_naive.txt
echo "start 2/15"
./naive_solution 160 160 160 1000 >> log_naive.txt
echo "start 3/15"
./naive_solution 192 192 192 1000 >> log_naive.txt
echo "start 4/15"
./naive_solution 512 512 512 1000 >> log_naive.txt
echo "start 5/15"
./naive_solution 544 544 544 1000 >> log_naive.txt
echo "start 6/15"
./naive_solution 576 576 576 1000 >> log_naive.txt
echo "start 7/15"
./naive_solution 1024 1024 1024 1000 >> log_naive.txt
echo "start 8/15"
./naive_solution 1056 1056 1056 1000 >> log_naive.txt
echo "start 9/15"
./naive_solution 1088 1088 1088 1000 >> log_naive.txt
echo "start 10/15"
./naive_solution 1120 1120 1120 1000 >> log_naive.txt
echo "start 11/15"
./naive_solution 1184 1184 1184 1000 >> log_naive.txt
echo "start 12/15"
./naive_solution 608 608 608 1000 >> log_naive.txt
echo "start 13/15"
./naive_solution 640 640 640 1000 >> log_naive.txt
echo "start 14/15"
./naive_solution 672 672 672 1000 >> log_naive.txt
echo "start 15/15"
./naive_solution 704 704 704 1000 >> log_naive.txt
