import argparse
import logging
import os
import random
import sys
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from datasets.dataset_ACDC import ACDC_dataset
from datasets.dataset_synapse import Synapse_dataset
from networks.lkca import LKCA
from trainer_synapse import trainer_synapse
from trainer_ACDC import trainer_ACDC
from utils import maybe_mkdir_p, get_logger

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/users/nieminxin/ACDC', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--volume_path', type=str,
                    default='/home/users/nieminxin/ACDC', help='root dir for data')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_ACDC', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir', default="./trained_ckpt")
parser.add_argument('--test_save_path', type=str,
                    default='./predictions', help='save directory of testing')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--dice_loss_weight', type=float,
                    default=0.7, help='loss balance factor for the dice loss')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--optimizer', type=str, default='AdamW',
                    help='the choice of optimizer')

parser.add_argument('--base_lr', type=float, default=1e-3,
                    help='segmentation network learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--clip_grad', type=float, default=8,
                    help='gradient norm')
parser.add_argument('--lr_scheduler', type=str, default='cosine',
                    help='the choice of learning rate scheduler')
parser.add_argument('--warmup_epochs', type=int,
                    default=20, help='learning rate warm up epochs')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')
parser.add_argument('--cfg', type=str, required=True,
                    metavar="FILE", help='path to config file', )
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--eval_interval', default=20, type=int, help='evaluation interval')
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    config = get_config(args.cfg)
    args.img_size = int(config.Params.img_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    args.test_save_path = join(args.test_save_path, args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.logger = get_logger(join(args.output_dir, "log.txt"))


    ######### save hyper parameters #########
    option = vars(args)  ## args is the argparsing

    file_name = os.path.join(args.output_dir, 'hyper.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    if args.dataset == "Synapse":
        args.root_path = os.path.join(args.root_path, "train_npz")

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config_train = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': args.num_classes,
        },
        'ACDC': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_ACDC',
            'num_classes': args.num_classes,
        }
    }
    dataset_config_test = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
        'ACDC': {
            'Dataset': ACDC_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.dataset_config_test = dataset_config_test
    args.dataset_config_train = dataset_config_train
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config_train[dataset_name]['num_classes']
    args.root_path = dataset_config_train[dataset_name]['root_path']
    args.list_dir_train = dataset_config_train[dataset_name]['list_dir']
    args.list_dir_test = dataset_config_test[dataset_name]['list_dir']
    net = LKCA(config).cuda()

    trainer = {'Synapse': trainer_synapse, 'ACDC': trainer_ACDC}
    trainer[dataset_name](args, net, args.output_dir)
