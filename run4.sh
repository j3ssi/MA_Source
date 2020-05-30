#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_005.sh 4 &> experimente2/prune_lasso_005_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_01.sh 4 &> experimente2/prune_lasso_01_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_015.sh 4 &> experimente2/prune_lasso_015_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_02.sh 4 &> experimente2/prune_lasso_02_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_lasso_025.sh 4 &> experimente2/prune_lasso_025_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_reconf_2.sh 4 &> experimente2/prune_reconf_2_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_reconf_5.sh 4 &> experimente2/prune_reconf_5_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_reconf_10.sh 4 &> experimente2/prune_reconf_10_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres01.sh 4 &> experimente2/prune_thres01_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres001.sh 4 &> experimente2/prune_thres001_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres0001.sh 4 &> experimente2/prune_thres0001_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres00001.sh 4 &> experimente2/prune_thres00001_4.txt
sleep 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 ./run_thres000001.sh 4 &> experimente2/prune_thres000001_4.txt
sleep 20
