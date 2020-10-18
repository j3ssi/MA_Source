import os
import yaml
import argparse

os.system('python3 main.py -j 6 --checkpoint ./output/experimente4/net2netdeeperL1_$1 --epochs 35 --batchTrue --batch_size 256 -s 3 -l 1 -n 5,5,5 -dlr --sparse_interval 0 --widthOfAllLayers 16,32,64  --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 0 --pathToModell ./output/experimente4/net2netdeeperL1_$1/model.nn')

os.system('python3 main.py -j 6 --checkpoint ./output/experimente4/net2netdeeperL1_$1 --deepper --epochs 5 --batchTrue --batch_size 256 -s 3 -l 1 -n 5,5,5 -dlr --sparse_interval 0 --widthOfAllLayers 16,32,64  --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 5 --pathToModell ./output/experimente4/net2netdeeperL1_$1/model.nn  --resume ./output/experimente4/net2netdeeperL1_$1/checkpoint.pth.tar')

os.system('python3 main.py --reset -j 6 --checkpoint ./output/experimente4/net2netdeeperL1_$1 --epochs 180 --batchTrue --batch_size 256  -s 3 -l 1 -n 5,5,5 -dlr --sparse_interval 0 --widthOfAllLayers 16,32,64  --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 15 --pathToModell ./output/experimente4/net2netdeeperL1_$1/model.nn  --resume ./output/experimente4/net2netdeeperL1_$1/checkpoint.pth.tar')

