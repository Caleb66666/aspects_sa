# @File: base_xlnet
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/1/16 11:10:16

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
        self.dl_path = abspath(f"data/{self.name}.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = None
        self.num_classes = None
        self.num_labels = None
        self.classes = None
        self.eval_per_batches = 200
        self.improve_require = 20000

        # 训练样本中，小于1024长度的样本数占据约98.3%，过长则截断
        self.max_seq = 768
        self.first_half = int(self.max_seq / 2)
        self.latter_half = self.max_seq - self.first_half
        self.epochs = 4
        # 更长的序列长度，减小batch大小
        self.batch_size = 16
        self.dropout = 0.5
        self.xlnet_hidden = 768
        self.attn_size = 128
        self.linear_size = 128

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


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.w = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.w.data.normal_(-1e-4, 1e-4)

    def forward(self, h):
        m = torch.tanh(h)
        alpha = torch.softmax(torch.matmul(m, self.w), dim=1).unsqueeze(-1)
        out = h * alpha
        return torch.sum(out, 1)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.xlnet = XLNetModel.from_pretrained(config.xlnet_path)
        [setattr(param, "requires_grad", True) for param in self.xlnet.parameters()]

        self.num_classes = config.num_classes
        self.num_labels = config.num_labels
        self.classes = config.classes

        self.units, self.criterion_list = nn.ModuleList(), list()
        for _ in range(self.num_labels):
            unit = nn.Sequential(
                Attn(config.xlnet_hidden),  # batch_size, xlnet_hidden
                nn.Linear(config.xlnet_hidden, config.linear_size),
                nn.BatchNorm1d(config.linear_size),
                nn.ELU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(config.linear_size, config.num_classes)  # batch_size, classes
            ).to(config.device)
            self.units.append(unit)
            self.criterion_list.append(nn.CrossEntropyLoss().to(config.device))

    def forward(self, inputs):
        seq_ids, labels, (inf_mask, seq_mask) = inputs[0], inputs[1:-2], inputs[-2:]
        assert len(labels) == self.num_labels, "number labels error!"

        # batch_size, seq_len, xlnet_hidden
        encoded_seq = self.xlnet(seq_ids, attention_mask=seq_mask)[0]

        total_logits, total_loss, total_f1 = list(), 0.0, 0.0
        # 为每个标签label分别进行分类
        for idx, (unit, criterion) in enumerate(zip(self.units, self.criterion_list)):
            logits = unit(encoded_seq)
            total_logits.append(logits)
            if labels:
                label = labels[idx]
                loss = criterion(logits, label)
                f1 = calc_f1(logits, label, self.classes, average="micro")
                total_loss += loss
                total_f1 += f1

        output_dict = {
            "logits": total_logits
        }
        if labels:
            output_dict.update({
                "f1": total_f1 / self.num_labels,
                "loss": total_loss / self.num_labels
            })
        return output_dict
