#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_005.sh 5 &> experimente2/prune_lasso_005_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_01.sh 5 &> experimente2/prune_lasso_01_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_015.sh 5 &> experimente2/prune_lasso_015_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_02.sh 5 &> experimente2/prune_lasso_02_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_025.sh 5 &> experimente2/prune_lasso_025_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_reconf_2.sh 5 &> experimente2/prune_reconf_2_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_reconf_5.sh 5 &> experimente2/prune_reconf_5_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_reconf_10.sh 5 &> experimente2/prune_reconf_10_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres01.sh 5 &> experimente2/prune_thres01_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres001.sh 5 &> experimente2/prune_thres001_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres0001.sh 5 &> experimente2/prune_thres0001_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres00001.sh 5 &> experimente2/prune_thres00001_5.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres000001.sh 5 &> experimente2/prune_thres000001_5.txt
sleep 20
