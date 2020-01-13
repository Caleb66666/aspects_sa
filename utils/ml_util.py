from torch import nn
import torch
from sklearn import metrics


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


def unwrap_to_tensors(*tensors):
    return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


def calc_accuracy(logits, label, dim=-1):
    pred_y = logits.argmax(dim=dim, keepdim=True).squeeze(dim=dim)
    true_y, pred_y = unwrap_to_tensors(label, pred_y)
    return metrics.accuracy_score(true_y, pred_y)
