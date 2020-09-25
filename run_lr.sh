#!/bin/sh

echo "lr 0.2 6"
./run_lr2.sh 6
sleep 3

echo "lr 0.2 7"
./run_lr2.sh 7
sleep 3

echo "lr 0.2 8"
./run_lr2.sh 8
sleep 3

echo "lr 0.2 9"
./run_lr2.sh 9
sleep 3

echo "lr 0.2 10"
./run_lr2.sh 10
sleep 3



echo "lr 0.1 6"
./run_lr1.sh 6
sleep 3

echo "lr 0.1 7"
./run_lr1.sh 7
sleep 3

echo "lr 0.1 8"
./run_lr1.sh 8
sleep 3

echo "lr 0.1 9"
./run_lr1.sh 9
sleep 3

echo "lr 0.1 10"
./run_lr1.sh 10
sleep 3



echo "lr 0.05 6"
./run_lr05.sh 6
sleep 3

echo "lr 0.05 7"
./run_lr05.sh 7
sleep 3

echo "lr 0.05 8"
./run_lr05.sh 8
sleep 3

echo "lr 0.05 9"
./run_lr05.sh 9
sleep 3

echo "lr 0.05 10"
./run_lr05.sh 10
sleep 3



echo "lr 0.025 6"
./run_lr025.sh 6
sleep 3

echo "lr 0.025 7"
./run_lr025.sh 7
sleep 3

echo "lr 0.025 8"
./run_lr025.sh 8
sleep 3

echo "lr 0.025 9"
./run_lr025.sh 9
sleep 3

echo "lr 0.025 10"
./run_lr025.sh 10
sleep 3

