import os


def main():


    # s = 3
    #
    # for i in range(0,20):
    #     cmdLine = 'python3 main.py '
    #     cmdLine = cmdLine + '--workers 4 --epochs 90 --test --cifar10'
    #     cmdLine = cmdLine + ' --learning-rate 0.2 --sparse_interval 10 --threshold 0.05 --en_group_lasso --batchTrue --batch_size 128 '
    #     cmdLine = cmdLine + ' --var_group_lasso_coeff 0.2 --test_batch 100 -s ' + str(s) + ' -n 5,5,5 -l 2'
    #     print(cmdLine)
    #     os.system(cmdLine)
    #

    s = 3

    for i in range(0,20):
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 4 --epochs 90 --test --cifar10'
        cmdLine = cmdLine + ' --learning-rate 0.2 --batchTrue --batch_size 128 '
        cmdLine = cmdLine + '  --test_batch 100 -s ' + str(s) + ' -n 5,5,5 -l 2'
        print(cmdLine)
        os.system(cmdLine)

# python3 main.py -j 4 --epochs 180  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.01 --en_group_lasso --batchTrue --batch_size 3900



if __name__ == '__main__':
    main()
