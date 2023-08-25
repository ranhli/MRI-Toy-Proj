import argparse
from utils.trainSolver import TrainSolver
import time
import warnings

warnings.filterwarnings("ignore")

def main(config):
    print('Training and testing on MRI')

    solver = TrainSolver(config)
    loss = solver.train()
    print('loss: {}\n'.format(loss))

def printArgs(args):
    print('-------------------args---------------------')
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))   # str, arg_type
    print('--------------------------------------------\n')

if __name__ == '__main__':
    print('Begin time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', dest='lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=20, help='Learning rate ratio for hyper networt')
    parser.add_argument('--epochs', dest='epochs', type=int, default=12, help='Epochs for training')
    parser.add_argument('--save_pkl_dir', dest='save_pkl_dir', type=str, default='/Users/henry/PycharmProjects/MRI/checkpoints')
    parser.add_argument('--test_index', dest='test_index', type=int,
                        default=[25, 119, 3, 7, 192, 23, 238, 215, 102, 218, 263, 156, 288, 221, 39, 127, 86, 359, 184, 175, 32, 2, 4, 247, 120, 36, 31, 284, 187, 150])
    config = parser.parse_args()
    printArgs(config)

    main(config)
    print('Complete time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
