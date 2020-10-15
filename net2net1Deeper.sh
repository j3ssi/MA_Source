#!/bin/sh

echo "j: 1 bis 5"
python3 main.py -j 6 --deeper  --checkpoint ./output/experimente4/net2netdeeperL1_$1 --epochs 5 --batchTrue --batch_size 256 -s 3 -l 1 -n 5,5,5 -dlr --sparse_interval 0 --widthOfAllLayers 16,32,64  --cifar10 --test --saveModell --test_batch 200 --epochsFromBegin 0 --pathToModell ./output/experimente4/net2netdeeperL1_$1/model.nn
sleep 5

