import os


def main():
    s = 3
    n = 3

    for batch_size in range(1, 102):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 1 '
        cmdLine = cmdLine + '--learning-rate 0.1 --schedule 91 136 '
        cmdLine = cmdLine + '--batchTrue --batch_size ' + str(batch_size * 10)
        cmdLine = cmdLine + ' --test_batch 100 -s ' + str(s) + ' -n ' + str(n) + ' -l 3'
        print(cmdLine)
        os.system(cmdLine)

    s = 3
    n = 3

    for batch_size in range(1, 102):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 1 --fp16'
        cmdLine = cmdLine + '--learning-rate 0.1 --schedule 91 136 '
        cmdLine = cmdLine + '--batchTrue --batch_size ' + str(batch_size * 10)
        cmdLine = cmdLine + ' --test_batch 100 -s ' + str(s) + ' -n ' + str(n) + ' -l 3'
        print(cmdLine)
        os.system(cmdLine)


    s = 4
    n = 2

    for batch_size in range(1, 70):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 1 '
        cmdLine = cmdLine + '--learning-rate 0.1 --schedule 91 136 '
        cmdLine = cmdLine + '--batchTrue --batch_size ' + str(batch_size * 10)
        cmdLine = cmdLine + ' --test_batch 100 -s ' + str(s) + ' -n ' + str(n) + ' -l 3'
        print(cmdLine)
        os.system(cmdLine)
    s = 4
    n = 2

    for batch_size in range(1, 70):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 1 --fp16'
        cmdLine = cmdLine + ' --learning-rate 0.1 --schedule 91 136 '
        cmdLine = cmdLine + '--batchTrue --batch_size ' + str(batch_size * 10)
        cmdLine = cmdLine + ' --test_batch 100 -s ' + str(s) + ' -n ' + str(n) + ' -l 3'
        print(cmdLine)
        os.system(cmdLine)


if __name__ == '__main__':
    main()
