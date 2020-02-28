# @File: base_model
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/18 18:14:54

import os
import torch
import random
import math
import numpy as np
from utils.time_util import cur_time_stamp
from utils.path_util import abspath, keep_max_backup, newest_file
from utils.ml_util import scale_lr, calc_f1
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
from allennlp.nn.util import masked_softmax


class BaseConfig(object):
    def __init__(self, name, debug):
        self.name = name
        self.debug = debug

        # 训练设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.data_cache = abspath(f"data/{name}")
        if not os.path.exists(self.data_cache):
            os.makedirs(self.data_cache)

        # data-loader，处理完的数据序列化的位置，不需要再次初始化、预处理、分词、word-piece、sentence-piece等操作
        self.dl_path = os.path.join(self.data_cache, "dl.pt")

        # 存储模型位置设置，使用时间戳进行模型的命名
        self.model_dir = abspath(f"checkpoints/{name}")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_ckpt = os.path.join(self.model_dir, "{}.ckpt")
        self.max_backup = 3  # 最多的模型保存

        # 训练损失记录设置
        self.summary_dir = abspath(f"summary/{name}")
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        # 日志记录
        self.logger_name = name
        self.logger_dir = abspath(f"log/{name}")
        if not os.path.exists(self.logger_dir):
            os.makedirs(self.logger_dir)
        self.logger_file = os.path.join(self.logger_dir, "{}.log")

        # restore模式
        self.restore_model = False
        self.default_scale = 1

        # 词嵌入相关
        self.w2v_path = os.path.join(self.data_cache, f"w2v.txt")

        self.epochs = None
        self.batch_size = None
        self.word_w2v = None
        self.char_w2v = None
        self.word_embed_path = None
        self.char_embed_path = None

    def debug_set(self):
        self.epochs = 1
        self.batch_size = 2
        self.dl_path = os.path.join(self.data_cache, "debug.dl.pt")
        self.logger_file = os.path.join(self.logger_dir, "{}.debug.log")
        self.model_ckpt = os.path.join(self.model_dir, "{}.debug.ckpt")
        self.w2v_path = os.path.join(self.data_cache, f"debug.w2v.txt")
        self.word_w2v = os.path.join(self.data_cache, f"debug.word.w2v.txt")
        self.char_w2v = os.path.join(self.data_cache, f"debug.char.w2v.txt")
        self.word_embed_path = os.path.join(self.data_cache, f"debug.word.embed.matrix")
        self.char_embed_path = os.path.join(self.data_cache, f"debug.char.embed.matrix")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def save_model(self, model, optimizer, scheduler, epoch, best_loss, last_improve):
        """
        存储模型，包括模型结构、优化器参数、调度器参数、当前epoch步数
        :param model:
        :param optimizer:
        :param scheduler:
        :param epoch:
        :param best_loss:
        :param last_improve:
        :return:
        """
        save_dict = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict(),
            "best_loss": best_loss,
            "last_improve": last_improve
        }
        torch.save(save_dict, self.model_ckpt.format(cur_time_stamp()))
        keep_max_backup(self.model_dir, self.max_backup)

    def restore_model(self, model, optimizer, scheduler, model_ckpt=None):
        if not model_ckpt:
            model_ckpt = newest_file(self.model_dir)
        save_dict = torch.load(model_ckpt)
        optimizer.load_state_dict(save_dict.get("optimizer"))
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        scheduler.load_state_dict(save_dict.get("scheduler"))
        model.load_state_dict(save_dict.get("model"))
        return model, optimizer, scheduler, save_dict["epoch"], save_dict["best_loss"], save_dict["last_improve"]

    @staticmethod
    def build_optimizer_scheduler(model, train_batches_len, weight_decay, lr, adam_epsilon, epochs,
                                  schedule_per_batches, warm_up_proportion):
        opt_params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in opt_params if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in opt_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(grouped_params, lr=lr, eps=adam_epsilon)

        schedule_steps = math.ceil(train_batches_len * epochs / schedule_per_batches)
        warm_up_steps = math.ceil(schedule_steps * warm_up_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=schedule_steps)
        return optimizer, scheduler

    def set_restore_lr(self, optimizer, scale=None):
        if self.restore_model:
            return
        self.restore_model = True
        scale = scale or self.default_scale
        scale_lr(optimizer, scale)


class AttnPool(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self._weight_vector = nn.Parameter(torch.zeros(size=(input_size,)), requires_grad=True)
        self._bias = nn.Parameter(torch.zeros(size=(1,)), requires_grad=True)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, input_x, x_mask):
        activated_x = torch.tanh(input_x)
        attn_value = torch.matmul(activated_x, self._weight_vector)
        normalized_attn = masked_softmax(attn_value, x_mask, dim=1)
        weighted_x = input_x * normalized_attn.unsqueeze(-1)
        pooled_x = torch.sum(weighted_x, dim=1)
        return pooled_x


class ExclusiveUnit(nn.Module):
    def __init__(self, input_dim, out_dim, linear_dim, dropout=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.attn_pool = AttnPool(input_dim)
        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(input_dim * 3, out_dim)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_dim * 3),
            nn.Linear(input_dim * 3, linear_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(dropout),
            nn.Linear(linear_dim, out_dim)
        )

    @staticmethod
    def avg_pool_on_seq(tensor):
        """
        一般tensor为三维矩阵，pool的层级一般是seq_len层级
        :param tensor:
        :return:
        """
        avg_p = torch.avg_pool1d(tensor.transpose(1, 2), tensor.size(1)).squeeze(-1)
        return avg_p

    @staticmethod
    def max_pool_on_seq(tensor):
        """
        一般tensor为三维矩阵，pool的层级一般是seq_len层级
        :param tensor:
        :return:
        """
        max_p = torch.max_pool1d(tensor.transpose(1, 2), tensor.size(1)).squeeze(-1)
        return max_p

    def forward(self, encoded_seq, seq_mask, label=None, classes=None, average=None):
        avg_pool = self.avg_pool_on_seq(encoded_seq)
        max_pool = self.max_pool_on_seq(encoded_seq)
        attn_pool = self.attn_pool(encoded_seq, seq_mask)
        concat_pool = torch.cat([avg_pool, max_pool, attn_pool], dim=1)
        logits = self.fc(concat_pool)
        if label is None:
            return logits, None, None
        return logits, self.criterion(logits, label), calc_f1(logits, label, classes, average=average)
