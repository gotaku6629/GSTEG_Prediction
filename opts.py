""" Define and parse commandline arguments """
import argparse
import os


def parse():
    print('parsing arguments')
    parser = argparse.ArgumentParser(description='PyTorch Charades Training')
    parser.add_argument('--rgb-data', metavar='DIR', default='/scratch/gsigurds/Charades_v1_rgb/',
                        help='path to dataset')
    parser.add_argument('--dataset', metavar='DIR', default='charades',
                        help='name of dataset under datasets/')
    parser.add_argument('--train-file', default='./Charades_v1_train.csv')
    parser.add_argument('--val-file', default='./Charades_v1_test.csv')
    parser.add_argument('--groundtruth-lookup', default='./groudtruth.p')
    parser.add_argument('--rgb-arch', '-ra', metavar='ARCH', default='i3d',
                        help='model architecture: ')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', # default=8
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N', # default=20
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=10, type=int,     # default=25
                        metavar='N', help='mini-batch size (default: 50)')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-decay-rate', default=3, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--rgb-pretrained-weights', default='')
    parser.add_argument('--inputsize', default=224, type=int)
    parser.add_argument('--extract-feat-dim', default=1024, type=int)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--manual-seed', default=0, type=int)
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456',
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', help='distributed backend')
    parser.add_argument('--train-size', default=0.2, type=float)
    parser.add_argument('--val-size', default=0.2, type=float)
    parser.add_argument('--cache-dir', default='./cache/')
    parser.add_argument('--name', default='test')
    parser.add_argument('--s-class', default=16, type=int)
    parser.add_argument('--o-class', default=38, type=int)
    parser.add_argument('--v-class', default=33, type=int)
    parser.add_argument('--accum-grad', default=1, type=int)
    parser.add_argument('--future', default=1, type=int)   # time of predicting future flame 
    parser.add_argument('--temporal', default=1, type=int) # time of using ST-Graph
    args = parser.parse_args()
    args.distributed = args.world_size > 1
    args.cache = args.cache_dir + args.name + '/'
    print('args.cache')  # ./gsteg/cr_caches/test/
    print(args.cache)
    if not os.path.exists(args.cache):
        os.makedirs(args.cache)

    return args
