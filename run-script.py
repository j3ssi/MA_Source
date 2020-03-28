import os


def main():
    # Test, wie wie sich die Accuracy in den verschiedenen Optimierungsstufen verhält
    # s = 3
    # n = 3
    #
    # for batch_size in range(1, 10):
    #     cmdLine = 'python3 main.py '
    #     cmdLine = cmdLine + '--workers 4 --epochs 50 --test'
    #     cmdLine = cmdLine + ' --learning-rate 0.1 '
    #     cmdLine = cmdLine + '--batchTrue --batch_size ' + str(batch_size * 100)
    #     cmdLine = cmdLine + ' --test_batch 100 -s ' + str(s) + ' -n ' + str(n) + ' -l 3'
    #     print(cmdLine)
    #     os.system(cmdLine)

    # s = 3
    # n = 3
    #
    # for batch_size in range(1, 20):
    #     cmdLine = 'python3 main.py '
    #     cmdLine = cmdLine + '--workers 4 --epochs 50 --O1 --test '
    #     cmdLine = cmdLine + ' --learning-rate 0.1 '
    #     cmdLine = cmdLine + '--batchTrue --batch_size ' + str(batch_size * 100)
    #     cmdLine = cmdLine + ' --test_batch 100 -s ' + str(s) + ' -n ' + str(n) + ' -l 3'
    #     print(cmdLine)
    #     os.system(cmdLine)

    # s = 3
    # n = 3
    #
    # for batch_size in range(1, 30):
    #     cmdLine = 'python3 main.py '
    #     cmdLine = cmdLine + '--workers 4 --epochs 50 --O3 --test'
    #     cmdLine = cmdLine + ' --learning-rate 0.1 '
    #     cmdLine = cmdLine + '--batchTrue --batch_size ' + str(batch_size * 100)
    #     cmdLine = cmdLine + ' --test_batch 100 -s ' + str(s) + ' -n ' + str(n) + ' -l 3'
    #     print(cmdLine)
    #     os.system(cmdLine)


    s = 3
    n = 3


    cmdLine = 'python3 main.py '
    cmdLine = cmdLine + '--workers 4 --epochs 50 --test --cifar100'
    cmdLine = cmdLine + ' --learning-rate 0.1 '
    cmdLine = cmdLine + ' --batchTrue --batch_size 1024 '
    cmdLine = cmdLine + ' --test_batch 100 -s ' + str(s) + ' -n 1,2,3,4 -l 3'
    print(cmdLine)
    os.system(cmdLine)


if __name__ == '__main__':
    main()
