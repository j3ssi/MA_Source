#!/bin/sh

echo "j: 0 bis 5"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 0 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn 
sleep 20

echo "j: 6 bis 10"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 5 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 11 bis 15"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 10 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 16 bis 20"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 15 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 21 bis 25"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 20 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 26 bis 30"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 25 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 31 bis 35"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 30 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 36 bis 40"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 35 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 41 bis 45"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 40 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 46 bis 50"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 45 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 51 bis 55"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 50 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 56 bis 60"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 55 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 61 bis 65"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 60 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 66 bis 70"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 65 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 71 bis 75"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 70 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 76 bis 80"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 75 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 81 bis 85"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 80 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 86 bis 90"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 85 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 91 bis 95"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 90 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 96 bis 100"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 95 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 101 bis 105"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 100 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 106 bis 110"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 105 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 111 bis 115"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 110 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 116 bis 120"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 120 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 121 bis 125"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 125 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 126 bis 130"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 130 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 131 bis 135"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 135 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 136 bis 140"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 140 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 141 bis 145"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 145 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 146 bis 150"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 150 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 151 bis 155"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 155 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 156 bis 160"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 160 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 161 bis 165"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 165 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 166 bis 170"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 170 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 171 bis 175"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 175 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar
sleep 20

echo "j: 176 bis 180"
python3 main.py -j 6 --checkpoint ./output/experimente2/prune_LWOLARS_$1 --epochs 5 -s 3 -l 2 -n 5,5,5 --sparse_interval 5 --en_group_lasso --var_group_lasso_coeff 0.2 --threshold 0.1 --lr 0.1 --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 180 --pathToModell ./output/experimente2/prune_LWOLARS_$1/model.nn  --resume ./output/experimente2/prune_LWOLARS_$1/checkpoint.pth.tar --lastEpoch
