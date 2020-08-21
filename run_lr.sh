#!/bin/sh

echo "lr 0.2"
./run_lr2.sh $1
sleep 20


echo "lr 0.1"
./run_lr1.sh $1
sleep 20


echo "lr 0.05"
./run_lr05.sh $1
sleep 20

echo "lr 0.025"
./run_lr025.sh $1
sleep 20

echo "Thres 0.0125"
./run_lr0125.sh $1
