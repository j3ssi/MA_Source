#!/bin/sh

echo "Lasso 0.05"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_005.sh $1
sleep 20

echo "Lasso 0.1"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_01.sh $1
sleep 20

echo "Lasso 0.15"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_015.sh $1
sleep 20

echo "Lasso 0.2"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_02.sh $1
sleep 20

echo "Lasso 0.25"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_025.sh $1  
