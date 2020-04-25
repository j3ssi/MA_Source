import os

def main():

    s = 3
    for i in range(0,1):
        for j in range(0, 36):
            cmdLine = 'python3 main.py '
            cmdLine = cmdLine + '--workers 0 --checkpoint ./output/prune1 --test --epochs 5 --cifar10 --gpu_id 2 '
            cmdLine = cmdLine + ' --threshold 0.01  --en_group_lasso --batchTrue --batch_size 128 --sparse_interval 5 '
            cmdLine = cmdLine + ' --var_group_lasso_coeff 0.2 --test_batch 100 -s ' + str(s) + ' -n 5,5,5 -l 2 '
            cmdLine = cmdLine + ' --gpu_id 2 --cifar10 --learning-rate 0.1 --sparse_interval 5  --learning-rate 0.1 '
            cmdLine = cmdLine + ' --epochsFromBegin ' + str(j*5) + ' --pathToModell ./output/prune1/model.nn '
            if(j>0):
                cmdLine = cmdLine + ' --resume ./output/prune1/checkpoint.pth.tar'

            print(cmdLine)
            os.system(cmdLine)
        cmdLine = 'python3 main.py '
        cmdLine = cmdLine + '--workers 0 --checkpoint ./output/prune1 --test --epochs 5 --cifar10 --gpu_id 2 '
        cmdLine = cmdLine + ' --threshold 0.01  --en_group_lasso --batchTrue --batch_size 128 --sparse_interval 5 '
        cmdLine = cmdLine + ' --var_group_lasso_coeff 0.2 --test_batch 100 -s ' + str(s) + ' -n 5,5,5 -l 2 --lastEpoch'
        cmdLine = cmdLine + ' --gpu_id 2 --cifar10 --learning-rate 0.1 --sparse_interval 5  --learning-rate 0.1 '
        cmdLine = cmdLine + ' --epochsFromBegin ' + str(j * 5) + ' --pathToModell ./output/prune1/model.nn '
        print(cmdLine)
        os.system(cmdLine)

# python3 main.py -j 4 --epochs 180  -s 3 -l 2 -n 5,5,5 --cifar10 --test --sparse_interval 5 --threshold 0.01 --en_group_lasso --batchTrue --batch_size 3900

# python3 main.py -j 4 --epochs 180  -s 3 -l 2 -n 5,5,5 --cifar10 --test --batchTrue --batch_size 3900


if __name__ == '__main__':
    main()


#python3 main.py --workers 0 --checkpoint ./output/prune1 --test --epochs 5 --cifar10 --gpu_id 2 --threshold 0.01  --en_group_lasso --batchTrue --batch_size 128 --sparse_interval 5 --var_group_lasso_coeff 0.2 --test_batch 100 -s  3  -n 5,5,5 -l 2 --lastEpoch --gpu_id 2 --cifar10 --learning-rate 0.1 --sparse_interval 5  --learning-rate 0.1 --epochsFromBegin 30 --pathToModell ./output/prune1/model.nn