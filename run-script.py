import os

def main():
    s = [1,2,3,4,5,6,7,8,9,10]
    n = [2,3,4,5,6,7,8,9,10]

    for i in s:
        for j in n:
            cmdLine = 'python3 main.py '
            cmdLine = cmdLine + '--workers 4 --epochs 100 '
            cmdLine = cmdLine + '--learning-rate 0.1 --schedule 91 136 '
            cmdLine = cmdLine + '--gpu_id 2 --train_batch 128 '
            cmdLine = cmdLine + '--en_group_lasso --test_batch 100 --sparse_interval 5'
            cmdLine = cmdLine + ' -s ' + str(i) + ' -n ' + str(j) +' -l 3'
            print (cmdLine)
            os.system(cmdLine)
    for i in s:
        for j in n:
            print("\nohne PruneTrain")
            cmdLine = 'python3 main.py '
            cmdLine = cmdLine + '--workers 4 --epochs 100 '
            cmdLine = cmdLine + '--learning-rate 0.1 --schedule 91 136 '
            cmdLine = cmdLine + '--gpu_id 2 --train_batch 128 '
            cmdLine = cmdLine + ' --test_batch 100 --sparse_interval 5'
            cmdLine = cmdLine + ' -s ' + str(i) + ' -n ' + str(j) +' -l 3'
            print (cmdLine)
            os.system(cmdLine)


if __name__ == '__main__':
    main()