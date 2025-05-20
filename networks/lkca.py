import torch
import torch.nn as nn

from networks.net import NET


def load_pretrained(ckpt_path, model):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    msg = model.load_pretrained(checkpoint['model'])
    # print(msg)
    del checkpoint
    torch.cuda.empty_cache()


class LKCA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = NET(**config.Params)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.net(x)
        return logits
    
    # def load_from(self, config):
    #     pretrained_path = config.MODEL.PRETRAIN_CKPT
    #     load_pretrained(pretrained_path, self.agile_former)
