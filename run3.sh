#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_005.sh 3 &> experimente2/prune_lasso_005_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_01.sh 3 &> experimente2/prune_lasso_01_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_015.sh 3 &> experimente2/prune_lasso_015_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_02.sh 3 &> experimente2/prune_lasso_02_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_025.sh 3 &> experimente2/prune_lasso_025_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_reconf_2.sh 3 &> experimente2/prune_reconf_2_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_reconf_5.sh 3 &> experimente2/prune_reconf_5_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_reconf_10.sh 3 &> experimente2/prune_reconf_10_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres01.sh 3 &> experimente2/prune_thres01_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres001.sh 3 &> experimente2/prune_thres001_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres0001.sh 3 &> experimente2/prune_thres0001_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres00001.sh 3 &> experimente2/prune_thres00001_3.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres000001.sh 3 &> experimente2/prune_thres000001_3.txt
sleep 20
