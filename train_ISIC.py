import argparse
import logging
import os
import random
from os.path import join
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from config import get_config
from datasets.dataset_ISIC import NPY_datasets
from lr_scheduler_factory import build_scheduler
from networks.lkca import LKCA
from optimizer_factory import build_optimizer
from utils import  get_logger, myResize, myRandomRotation, myRandomVerticalFlip, myRandomHorizontalFlip, \
    myToTensor, myNormalize

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/users/nieminxin/data_isic1718/isic2018/', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='ISIC2018', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir', default="./trained_ckpt")
parser.add_argument('--test_save_path', type=str,
                    default='./predictions', help='save directory of testing')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--dice_loss_weight', type=float,
                    default=0.6, help='loss balance factor for the dice loss')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--optimizer', type=str, default='AdamW',
                    help='the choice of optimizer')

parser.add_argument('--base_lr', type=float, default=0.001,
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
parser.add_argument('--eval_interval', default=1, type=int, help='evaluation interval')
args = parser.parse_args()
print(args)


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

def validation(model, validloader):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for data in tqdm(validloader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            P = model(img)
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            outputs = outputs.squeeze(1).cpu().detach().numpy()
            preds.append(outputs)
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds >= 0.5, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    return miou, f1_or_dsc, accuracy, specificity, sensitivity, confusion


if __name__ == '__main__':
    config = get_config(args.cfg)
    args.img_size = int(config.Params.img_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    output_dir = os.path.join(args.output_dir, args.dataset)
    test_save_path = join(args.test_save_path, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.logger = get_logger(join(output_dir, "log.txt"))

    option = vars(args)

    file_name = os.path.join(args.output_dir, 'hyper.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

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
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    model = LKCA(config).cuda()



    train_transformer = transforms.Compose([
        myNormalize(args.dataset, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(args.img_size, args.img_size)
    ])

    test_transformer = transforms.Compose([
        myNormalize(args.dataset, train=False),
        myToTensor(),
        myResize(args.img_size, args.img_size)
    ])

    train_dataset = NPY_datasets(args.root_path, train_transformer, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=8)
    val_dataset = NPY_datasets(args.root_path, test_transformer, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=2,
                            drop_last=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)


    optimizer = build_optimizer(args, model)
    if not args.lr_scheduler in ['const', 'exponential']:
        lr_scheduler = build_scheduler(args, optimizer, len(train_loader))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    lowest_train_loss = float('inf')
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    ce_loss = BCELoss()
    dice_loss = DiceLoss()

    for epoch_num in iterator:
        epoch_loss = 0.0
        model.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch
            image_batch, label_batch = image_batch.cuda(non_blocking=True).float(), label_batch.cuda(non_blocking=True).float()
            iout = model(image_batch)
            loss = 0.0
            lc1, lc2 = 0.4, 0.6  # 0.3, 0.7
            loss_ce = ce_loss(iout, label_batch)
            loss_dice = dice_loss(iout, label_batch)
            loss += (lc1 * loss_ce + lc2 * loss_dice)
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            if args.lr_scheduler == 'exponential':
                lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            elif args.lr_scheduler == 'const':
                lr_ = args.base_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_scheduler.step_update(epoch_num * len(train_loader) + i_batch)

            epoch_loss += loss.item()
            iter_num = iter_num + 1

            if iter_num % 20 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info('iteration %d : loss : %f, mem: %.0fMB' % (
                    iter_num, loss.item(), memory_used))

        if epoch_loss < lowest_train_loss:
            lowest_train_loss = epoch_loss
            save_mode_path = os.path.join(output_dir, 'best_train_model.pth')
            torch.save(model.state_dict(), save_mode_path)

        eval_interval = 1
        if (epoch_num + 1) % eval_interval == 0:
            if (epoch_num+1) % 1 == 0:
                save_mode_path = os.path.join(output_dir, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            miou, f1_or_dsc, accuracy, specificity, sensitivity, confusion = validation(model, val_loader)
            log_info = f'val epoch: {epoch_num}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                                    specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
            logging.info(log_info)
            model.train()

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(output_dir, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
