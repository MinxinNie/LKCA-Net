import logging
import os
import random
import sys
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from test import inference
from utils import DiceLoss, powerset
from torchvision import transforms
from optimizer_factory import build_optimizer
from lr_scheduler_factory import build_scheduler
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator


def trainer_synapse(args, model, snapshot_path):
    # from datasets.dataset_synapse import Synapse_dataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = build_optimizer(args, model)
    if not args.lr_scheduler in ['const', 'exponential']:
        lr_scheduler = build_scheduler(args, optimizer, len(trainloader))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    lowest_train_loss = float('inf')
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    l = [0, 1, 2, 3]
    ss = [x for x in powerset(l)]


    for epoch_num in iterator:
        epoch_loss = 0.0
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            P = model(image_batch)

            loss = 0.0
            lc1, lc2 = 0.3, 0.7  # 0.3, 0.7

            for s in ss:
                iout = 0.0
                # print(s)
                if (s == []):
                    continue
                for idx in range(len(s)):
                    iout += P[s[idx]]
                loss_ce = ce_loss(iout, label_batch[:].long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (lc1 * loss_ce + lc2 * loss_dice)

            optimizer.zero_grad()

            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            if args.lr_scheduler == 'exponential':
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            elif args.lr_scheduler == 'const':
                lr_ = base_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_scheduler.step_update(epoch_num * len(trainloader) + i_batch)

            epoch_loss += loss.item()
            iter_num = iter_num + 1

            if iter_num % 20 == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, mem: %.0fMB' % (
                iter_num, loss.item(), loss_ce.item(), loss_dice.item(), memory_used))

        if epoch_loss < lowest_train_loss:
            lowest_train_loss = epoch_loss
            save_mode_path = os.path.join(snapshot_path, 'best_train_model.pth')
            torch.save(model.state_dict(), save_mode_path)

        eval_interval = 1
        if epoch_num+1 >= 100 and (epoch_num+1) % 10 == 0:
            # save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            # torch.save(model.state_dict(), save_mode_path)
            # logging.info("save model to {}".format(save_mode_path))

            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            test_save_mode_path = os.path.join(args.test_save_path, 'epoch_' + str(epoch_num))
            if not os.path.exists(test_save_mode_path):
                os.makedirs(test_save_mode_path)
            inference(args, model, test_save_path=test_save_mode_path)
            model.train()

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            iterator.close()
            break

    # writer.close()
    return "Training Finished!"