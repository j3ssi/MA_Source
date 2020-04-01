import os


def main():


    s = 3

    for i in range(0,10):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 180 --test --cifar10'
        cmdLine = cmdLine + ' --learning-rate 0.1 '
        cmdLine = cmdLine + ' --batchTrue --batch_size 320 -b '
        cmdLine = cmdLine + ' --test_batch 100 -s ' + str(s) + ' -n 4,4,4 -l 3'
        print(cmdLine)
        os.system(cmdLine)


if __name__ == '__main__':
    main()
