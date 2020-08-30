#!/bin/sh


echo "baseline 5"
time ./baseline.sh 5
sleep 20

echo "Lasso 0.1 5"
./run_lasso_01.sh 5
sleep 20

echo "Lasso 0.15 5"
./run_lasso_015.sh 5
sleep 20

echo "Lasso 0.2 5"
./run_lasso_02.sh 5
sleep 20


echo "Lasso 0.25 5"
./run_lasso_025.sh 5
sleep 20

echo "reconf 2 5"
./run_reconf_2.sh 5
sleep 20

echo "reconf 5 5"
./run_reconf_5.sh 5
sleep 20

echo "reconf 10 5"
./run_reconf_10.sh 5

echo "Thres 0.1 5"
./run_thres11.sh 5
sleep 20

echo "Thres 0.01 5"
./run_thres21.sh 5
sleep 20

