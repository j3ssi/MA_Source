import os


def main():


    s = 3
    for i in range(0,10):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 180 --test --cifar10'
        cmdLine = cmdLine + ' --learning-rate 0.1 --batchTrue --batch_size 2048 '
        cmdLine = cmdLine + '  --test_batch 100 -s ' + str(s) + ' -n 5,5,5 -l 2'
        os.system(cmdLine)



if __name__ == '__main__':
    main()
