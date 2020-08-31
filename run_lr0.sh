#!/bin/sh

echo "lr 0.05 2"
./run_lr05.sh 2
sleep 3

echo "lr 0.05 3"
./run_lr05.sh 3
sleep 3

echo "lr 0.05 4"
./run_lr05.sh 4
sleep 3

echo "lr 0.05 5"
./run_lr05.sh 5
sleep 3

echo "lr 0.025 1"
./run_lr025.sh 1
sleep 3

echo "lr 0.025 2"
./run_lr025.sh 2
sleep 3

echo "lr 0.025 3"
./run_lr025.sh 3
sleep 3

echo "lr 0.025 4"
./run_lr025.sh 4
sleep 3

echo "lr 0.025 5"
./run_lr025.sh 5
sleep 3


echo "Thres 0.0125 1"
./run_lr0125.sh 1
sleep 3

echo "Thres 0.0125 2"
./run_lr0125.sh 2
sleep 3

echo "Thres 0.0125 3"
./run_lr0125.sh 3
sleep 3

echo "Thres 0.0125 4"
./run_lr0125.sh 4
sleep 3

echo "Thres 0.0125 5"
./run_lr0125.sh 5
sleep 3