#!/bin/sh



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


echo "Thres 0.0001 5"
./run_thres41.sh 5
sleep 20

echo "Thres 0.00001 5"
./run_thres51.sh 5
sleep 20

echo "lr 0.2 5"
./run_lr2.sh 5
sleep 3

echo "lr 0.1 5"
./run_lr1.sh 5
sleep 3

echo "lr 0.05 5"
./run_lr05.sh 5
sleep 3

echo "lr 0.025 5"
./run_lr025.sh 5
sleep 3

echo "Thres 0.0125 5"
./run_lr0125.sh 5
sleep 3