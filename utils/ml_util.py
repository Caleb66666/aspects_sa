from torch import nn
import torch
from sklearn import metrics
import math
import random
import warnings

warnings.filterwarnings("ignore")


def init_network(model, method="xavier", exclude="embedding", seed=279):
    for name, w in model.named_parameters():
        if exclude not in name:
            if "weight" in name:
                if method == "xavier":
                    nn.init.xavier_normal_(w)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif "bias" in name:
                nn.init.constant_(w, 0)
            else:
                pass


def init_bert_net(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_lr(optimizer, scale, init_lr, lr_decay, min_lr=1e-4):
    cur_lr = init_lr * (lr_decay ** scale)
    if cur_lr < min_lr:
        cur_lr = min_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = cur_lr
    return cur_lr


def scale_lr(optimizer, scale=1):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= scale


def unwrap_to_tensors(*tensors):
    return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


def calc_accuracy(logits, label, dim=-1):
    pred_y = logits.argmax(dim=dim, keepdim=True).squeeze(dim=dim)
    true_y, pred_y = unwrap_to_tensors(label, pred_y)
    return metrics.accuracy_score(true_y, pred_y)


def calc_f1(logits, label, classes, average="micro", dim=-1):
    pred_y = logits.argmax(dim=dim, keepdim=True).squeeze(dim=dim)
    true_y, pred_y, classes = unwrap_to_tensors(label, pred_y, classes)
    return metrics.f1_score(true_y, pred_y, labels=classes, average=average)


def gen_random_mask(batch_size, max_seq, seed=279):
    random.seed(seed)
    seq_mask = torch.zeros(size=(batch_size, max_seq))
    seq_len = list()
    for idx in range(batch_size):
        random_one = random.randint(math.ceil(max_seq / 3), max_seq)
        seq_len.append(random_one)
        seq_mask[idx, 0:random_one] = 1
    return seq_mask.int(), torch.tensor(seq_len)
