#!/bin/sh

echo "Lasso 0.05"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 ./run_lasso_005.sh $1
sleep 20

echo "Lasso 0.1"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 ./run_lasso_01.sh $1
sleep 20

