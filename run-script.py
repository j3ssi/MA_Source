import os

def main():
    s = 1
    n = 2

    for batch_size in range(1,881):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 1 '
        cmdLine = cmdLine + '--learning-rate 0.1 --schedule 91 136 '
        cmdLine = cmdLine + '--batchTrue --batch_size ' + str(batch_size*10)
        cmdLine = cmdLine + ' --en_group_lasso --test_batch 100 --sparse_interval 5'
        cmdLine = cmdLine + ' -s ' + str(s) + ' -n ' + str(n) +' -l 3'
        print (cmdLine)
        os.system(cmdLine)

    s = 2
    n = 3

    for batch_size in range(1,236):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 1 '
        cmdLine = cmdLine + '--learning-rate 0.1 --schedule 91 136 '
        cmdLine = cmdLine + '--batchTrue --batch_size ' + str(batch_size*10)
        cmdLine = cmdLine + ' --en_group_lasso --test_batch 100 --sparse_interval 5'
        cmdLine = cmdLine + ' -s ' + str(s) + ' -n ' + str(n) +' -l 3'
        print (cmdLine)
        os.system(cmdLine)



if __name__ == '__main__':
    main()