import os


def main():


    s = 3
    for i in range(0,4):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 90 --test --cifar10'
        cmdLine = cmdLine + ' --learning-rate 0.2 --batchTrue --batch_size 128 '
        cmdLine = cmdLine + '  --test_batch 100 -s ' + str(s) + ' -n 5,5,5 -l 2'
        os.system(cmdLine)
