# @File: xlnet_multi_attn
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/1/13 16:54:15

import os
import torch
from torch import nn
from utils.path_util import abspath, keep_max_backup, newest_file
from utils.time_util import cur_time_stamp
from utils.ml_util import calc_f1
from data_loader import XlLoader
from transformers import AdamW, get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer

from logging import ERROR
from transformers.tokenization_utils import logger as tokenizer_logger
from transformers.file_utils import logger as file_logger
from transformers.configuration_utils import logger as config_logger
from transformers.modeling_utils import logger as model_logger

[logger.setLevel(ERROR) for logger in (tokenizer_logger, file_logger, config_logger, model_logger)]


class Config(object):
    def __init__(self):
        self.name = os.path.basename(__file__).split(".")[0]
        self.train_file = abspath("data/train.csv")
        self.valid_file = abspath("data/valid.csv")
        self.loader_cls = XlLoader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = None
        self.num_classes = None
        self.num_labels = None
        self.eval_per_batches = 200
        self.improve_require = 10000

        # 训练样本中，小于1024长度的样本数占据约98.3%，过长则截断
        self.max_seq = 1024
        self.epochs = 4
        # 更长的序列长度，减小batch大小
        self.batch_size = 32
        self.dropout = 0.5
        self.xlnet_hidden = 768
        self.attn_size = 256
        self.linear_size = 256

        # 梯度相关
        self.learning_rate = 1e-5
        self.weight_decay = 1e-2
        self.warm_up_steps = 40
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 5

        self.xlnet_path = "/data/wangqian/berts/xlnet-base-chinese"
        # self.xlnet_path = "/Users/Vander/Code/pytorch_col/xlnet-base-chinese"
        self.tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
        self.cls = self.tokenizer.cls_token
        self.sep = self.tokenizer.sep_token

        self.model_dir = abspath(f"checkpoints/{self.name}")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_ckpt = os.path.join(self.model_dir, "{}.%s.ckpt" % self.name)
        self.summary_dir = abspath(f"summary/{self.name}")
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

    def build_optimizer(self, model, t_total):
        opt_params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in opt_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in opt_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(grouped_params, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warm_up_steps,
                                                    num_training_steps=int(t_total / self.eval_per_batches) + 1)
        return optimizer, scheduler

    @staticmethod
    def scheduler_step(scheduler, loss):
        scheduler.step()

    def save_model(self, model, optimizer, epoch, best_loss, max_backup=2):
        save_dict = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "best_loss": best_loss
        }
        torch.save(save_dict, self.model_ckpt.format(cur_time_stamp()))
        keep_max_backup(self.model_dir, max_backup)

    def restore_model(self, model, optimizer, model_ckpt=None):
        if not model_ckpt:
            model_ckpt = newest_file(self.model_dir)
        save_dict = torch.load(model_ckpt)
        optimizer.load_state_dict(save_dict.get("optimizer"))
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        model.load_state_dict(save_dict.get("model"))
        return model, optimizer, save_dict["epoch"], save_dict["best_loss"]


class Attention(nn.Module):
    def __init__(self, hidden_size, attn_size):
        super(Attention, self).__init__()

        self.w = nn.Parameter(torch.zeros(hidden_size, attn_size), requires_grad=True)
        self.u = nn.Parameter(torch.zeros(attn_size, 1), requires_grad=True)
        [p.data.normal_(-0.001, 0.001) for p in (self.w, self.u)]

    def forward(self, input_x):
        x = torch.matmul(input_x, self.w)
        x = torch.tanh(x)
        x = torch.matmul(x, self.u)
        alpha = torch.softmax(x, dim=1)
        x = input_x * alpha
        return x


class MultiPool(nn.Module):
    def __init__(self):
        super(MultiPool, self).__init__()

    def forward(self, input_x):
        avg_p = torch.avg_pool1d(input_x.transpose(1, 2), input_x.size(1)).squeeze(-1)
        max_p = torch.max_pool1d(input_x.transpose(1, 2), input_x.size(1)).squeeze(-1)
        return torch.cat([avg_p, max_p], dim=1)


class SelfAttnWithMask(nn.Module):
    def __init__(self):
        super(SelfAttnWithMask, self).__init__()

    @staticmethod
    def zero_inf_mask(seq):
        mask = seq.eq(0)
        float_mask = mask.float()
        return torch.masked_fill(float_mask, mask, float("-inf"))

    def forward(self, encoded_seq, inf_mask):
        attn = torch.matmul(encoded_seq, encoded_seq.transpose(1, 2))
        soft_attn = torch.softmax(attn + inf_mask.unsqueeze(1), dim=-1)
        soft_align = torch.matmul(soft_attn, encoded_seq)
        return soft_align


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.xlnet = XLNetModel.from_pretrained(config.xlnet_path)
        list(map(lambda param: setattr(param, "requires_grad", True), self.xlnet.parameters()))

        self.device = config.device
        self.num_classes = config.num_classes
        self.num_labels = config.num_labels
        self.classes = config.classes

        self.self_attn = SelfAttnWithMask()
        self.units = list()
        for idx in range(self.num_labels):
            attn_fc_unit = nn.Sequential(
                # batch_size, seq, xlnet_hidden * 2
                Attention(config.xlnet_hidden * 2, config.attn_size),
                # batch_size, xlnet_hidden * 4
                MultiPool(),
                nn.BatchNorm1d(config.xlnet_hidden * 4),
                nn.Dropout(config.dropout),
                nn.Linear(config.xlnet_hidden * 4, config.num_classes),
            )
            self.units.append((attn_fc_unit, nn.CrossEntropyLoss().to(config.device)))

    def forward(self, inputs):
        # 输入数据解析
        seq_ids = inputs[0]
        labels = inputs[1:-2]
        inf_mask, seq_mask = inputs[-2:]
        assert len(labels) == self.num_labels, "number labels error!"

        # 获取xlnet的初始语意表征, [16, 1024, 768]
        outputs = self.xlnet(seq_ids, attention_mask=seq_mask)
        encoded_seq = outputs[0]

        # 使用自关注增强语意表征 [16, 1024, 768 * 2]
        soft_align = self.self_attn(encoded_seq, inf_mask)
        enhanced_seq = torch.cat([encoded_seq, soft_align], dim=-1)

        total_logits = list()
        total_loss, total_f1 = 0.0, 0.0
        for idx, (unit, criterion) in enumerate(self.units):
            logits = unit(enhanced_seq)
            total_logits.append(logits)
            if labels:
                label = labels[idx]
                loss = criterion(logits, label)
                f1 = calc_f1(logits, label, self.classes)
                total_loss += loss
                total_f1 += f1

        # 结果输出
        output_dict = {
            "logits": total_logits
        }
        if labels:
            output_dict.update({
                "f1": total_f1 / self.num_labels,
                "loss": total_loss / self.num_labels
            })
        return output_dict


if __name__ == '__main__':
    pass
