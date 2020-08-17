#!/bin/sh


echo "Thres 0.1"
./run_thres11.sh $1
sleep 20

echo "Thres 0.01"
./run_thres21.sh $1
sleep 20

echo "Thres 0.001"
./run_thres31.sh $1
sleep 20

echo "Thres 0.0001"
./run_thres41.sh $1
sleep 20

echo "Thres 0.00001"
./run_thres51.sh $1
