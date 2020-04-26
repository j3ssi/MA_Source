#!/bin/sh

echo "j: 0 bis 5"
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 0 --pathToModell ./output/prune1/model.nn
sleep 60

echo "j: 6 bis 10"
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 5 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar
sleep 60

echo "j: 11 bis 15"
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 10 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar
sleep 60

echo "j: 16 bis 20"
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 15 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar
sleep 60

echo "j: 21 bis 25"
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 20 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar
sleep 60

echo "j: 26 bis 30"
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 25 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar
sleep 60

echo "j: 31 bis 35"
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 30 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar
sleep 60

echo "j: 36 bis 40"
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 35 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar
sleep 60

echo "j: 41 bis 45"
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 40 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 46 bis 50
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 45 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 51 bis 55
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 50 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 56 bis 60
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 55 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 61 bis 65
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 60 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 66 bis 70
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 65 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 71 bis 75
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 70 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 76 bis 80
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 75 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 81 bis 85
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 80 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 86 bis 90
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 85 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 91 bis 95
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 90 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 96 bis 100
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 95 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 101 bis 105
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 100 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 106 bis 110
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 105 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 111 bis 115
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 110 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 116 bis 120
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 120 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 121 bis 125
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 125 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 126 bis 130
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 130 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 131 bis 135
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 135 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 136 bis 140
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 140 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 141 bis 145
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 145 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 146 bis 150
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 150 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 151 bis 155
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 155 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 156 bis 160
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 160 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 161 bis 165
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 165 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 166 bis 170
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 170 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 171 bis 175
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 175 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar

# j: 176 bis 180
python3 main.py -j 2 --checkpoint ./output/prune1 --epochs 5  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.001 --en_group_lasso --batchTrue --batch_size 64 --gpu_id 0  --var_group_lasso_coeff 0.2 --test_batch 200 --epochsFromBegin 180 --pathToModell ./output/prune1/model.nn --resume ./output/prune1/checkpoint.pth.tar --lastEpoch
