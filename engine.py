import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from isic_utils import save_imgs, save_msk_pred


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)

# switch to evaluate mode
def visualize_results_horizontal(img, gt, pred, save_path):
    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(gt, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(pred, cmap='gray')
    plt.axis('off')
    if save_path is not None:
        save_path = save_path + save_path + '_'
    plt.savefig(save_path)
    plt.close()

def test(model, validloader, path):
    save_dir = os.path.join(path, 'test_results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(validloader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            P = model(img)
            for b in range(img.size(0)):  # 处理batch中的每张图片
                img_np = img[b].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
                img_np = ((img_np - img_np.min()) * 255 / (img_np.max() - img_np.min())).astype(np.uint8)

                gt_np = msk[b].squeeze().cpu().numpy()
                gt_np = (gt_np * 255).astype(np.uint8)

                outputs = sum(P) if isinstance(P, list) else P
                outputs = outputs.squeeze(1).cpu().detach().numpy()

                pred_np = outputs[b]
                pred_np = (pred_np > 0.5).astype(np.uint8) * 255

                visualize_results_horizontal(
                    img_np,
                    gt_np,
                    pred_np,
                    os.path.join(save_dir, f'result_{i}_{b}.png'))

def test_one_epoch(test_loader,
                   model,
                   criterion,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)

            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i % config.save_interval == 0:
                save_msk_pred(out, i, config.work_dir, config.datasets, config.threshold, test_data_name=test_data_name)

