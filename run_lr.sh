#!/bin/sh

echo "lr 0.2 1"
./run_lr2.sh 1
sleep 3

echo "lr 0.2 2"
./run_lr2.sh 2
sleep 3

echo "lr 0.2 3"
./run_lr2.sh 3
sleep 3

echo "lr 0.2 4"
./run_lr2.sh 4
sleep 3

echo "lr 0.2 5"
./run_lr2.sh 5
sleep 3



echo "lr 0.1 1"
./run_lr1.sh 1
sleep 3

echo "lr 0.1 2"
./run_lr1.sh 2
sleep 3

echo "lr 0.1 3"
./run_lr1.sh 3
sleep 3

echo "lr 0.1 4"
./run_lr1.sh 4
sleep 3

echo "lr 0.1 5"
./run_lr1.sh 5
sleep 3



echo "lr 0.05 1"
./run_lr05.sh 1
sleep 3

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

