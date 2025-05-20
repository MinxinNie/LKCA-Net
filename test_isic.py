import argparse

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader


from config import get_config


from datasets.dataset_ISIC import NPY_datasets
from engine import *
import os
import sys

from isic_config import setting_config
from isic_utils import get_optimizer, get_scheduler, log_config_info, set_seed
from networks.lkca import LKCA

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" # "0, 1, 2, 3"

from utils import *

from config import get_config
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm




def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    config_cfg = get_config(config.cfg)
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)



    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    # gpu_ids = [0, 1]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    val_dataset = NPY_datasets(config.data_path, config.test_transformer, train=False)

    test_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Models----------#')
    # # AttentionUNet
    # model = AttU_Net(n_channels=3, n_classes=1, scale_factor=1).cuda()
    # model.load_state_dict(torch.load('/home/users/nieminxin/LKCA-NET/ISIC/ATTENTION_UNETbest-epoch127-loss0.3275.pth'))
    # DCSAUNet
    model = LKCA(config_cfg).cuda()

    msg = model.load_state_dict(torch.load(r"C:\Users\puff\Desktop\epoch_20.pth"))
    # # MALUNet
    # model_cfg = config.model_config
    # model = MALUNet(num_classes=model_cfg['num_classes'],
    #                 input_channels=model_cfg['input_channels'],
    #                 c_list=model_cfg['c_list'],
    #                 split_att=model_cfg['split_att'],
    #                 bridge=model_cfg['bridge']).cuda()
    # model.load_state_dict(torch.load('/home/users/nieminxin/LKCA-NET/ISIC/MALUNETbest-epoch107-loss0.3404.pth'))
    # # SANet
    # model = SANet().cuda()
    # model.load_state_dict(torch.load('model/pretrained_pth/isic17_models/SANet.pth'))
    # # SwinUNetV2
    # model = SwinTransformerSys(img_size=256, window_size=8, num_classes=1).cuda()
    # model.load_state_dict(torch.load('/home/users/nieminxin/LKCA-NET/ISIC/SWINUNET.pth'))
    # # TransFuse use test_one_epoch_deepsup
    # model = TransFuse_S().cuda()
    # model.load_state_dict(torch.load('model/pretrained_pth/isic17_models/TransFuse.pth'))
    # # TransUNet
    # config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    # config_vit.n_classes = 1
    # config_vit.n_skip = 3
    # model = VisionTransformer(config_vit, img_size=256, num_classes=config_vit.n_classes).cuda()
    # model.load_state_dict(torch.load('/home/users/nieminxin/LKCA-NET/ISIC/-TRANSUNETbest-epoch33-loss0.3007.pth'))
    # # UNetPlusPlus
    # model = ResNet34UnetPlus(num_channels=3, num_class=1).cuda()
    # model.load_state_dict(torch.load('model/pretrained_pth/isic17_models/UNetPlusPlus.pth'))
    # # UNet
    # model = Unet(in_channels=3, classes=1).cuda()
    # model.load_state_dict(torch.load('/home/users/nieminxin/LKCA-NET/ISIC/UNET-best-epoch132-loss0.3330.pth'))


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Testing----------#')
    # best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
    # model.module.load_state_dict(best_weight)
    # loss = test_one_epoch(
    loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            config,
        )
    print('#----------predicting over----------#')
    print('#----------prediction path = ' + config.work_dir + '----------#')


if __name__ == '__main__':
    config = setting_config
    print("networks = " + config.network)
    config.work_dir = "./predictions/" + config.datasets + "/" + config.network + "/"
    print("config.work_dir = " + config.work_dir)

    main(config)
