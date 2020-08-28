#!/bin/sh


echo "Thres 0.1 1"
time ./run_thres11.sh 1
sleep 20

echo "Thres 0.1 2"
./run_thres11.sh 2
sleep 20

echo "Thres 0.1 3"
./run_thres11.sh 3
sleep 20

echo "Thres 0.1 4"
./run_thres11.sh 4
sleep 20

echo "Thres 0.1 5"
./run_thres11.sh 5
sleep 20


echo "Thres 0.01 1"
./run_thres21.sh 1
sleep 20

echo "Thres 0.01 2"
./run_thres21.sh 2
sleep 20

echo "Thres 0.01 3"
./run_thres21.sh 3
sleep 20

echo "Thres 0.01 4"
./run_thres21.sh 4
sleep 20

echo "Thres 0.01 5"
./run_thres21.sh 5
sleep 20


echo "Thres 0.001 1"
./run_thres31.sh 1
sleep 20

echo "Thres 0.001 2"
./run_thres31.sh 2
sleep 20

echo "Thres 0.001 3"
./run_thres31.sh 3
sleep 20

echo "Thres 0.001 4"
./run_thres31.sh 4
sleep 20

echo "Thres 0.001 5"
./run_thres31.sh 5
sleep 20


echo "Thres 0.0001 1"
./run_thres41.sh 1
sleep 20

echo "Thres 0.0001 2"
./run_thres41.sh 2
sleep 20


echo "Thres 0.0001 3"
./run_thres41.sh 3
sleep 20

echo "Thres 0.0001 4"
./run_thres41.sh 4
sleep 20

echo "Thres 0.0001 5"
./run_thres41.sh 5
sleep 20

echo "Thres 0.00001 1"
./run_thres51.sh 1
sleep 20

echo "Thres 0.00001 2"
./run_thres51.sh 2
sleep 20

echo "Thres 0.00001 3"
./run_thres51.sh 3
sleep 20

echo "Thres 0.00001 4"
./run_thres51.sh 4
sleep 20

echo "Thres 0.00001 5"
./run_thres51.sh 5
sleep 20