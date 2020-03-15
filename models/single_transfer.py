# @File: single_transfer
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/3/15 22:06:58

import os
from torch import nn
from models.base_model import BaseConfig
from data_loader.better_transfer_loader import TrainLoader
from utils.path_util import abspath
from transformers import AlbertModel
from utils.ml_util import calc_f1


class Config(BaseConfig):
    def __init__(self, debug):
        super(Config, self).__init__(os.path.basename(__file__).split(".")[0], debug)
        self.seed = 279
        self.set_seed(self.seed)

        # 需要并行时设置并行数
        self.nb_workers = 4

        # 读取、解析初始数据
        self.loader_cls = TrainLoader
        self.train_file = abspath("data/train.csv")
        self.valid_file = abspath("data/valid.csv")
        self.premise = "content"
        self.hypothesis = None
        self.shuffle = True
        self.max_seq = 1000
        self.header = 0
        self.sep = ","
        self.encoding = "utf-8"
        self.if_lower = True

        # 分词相关/向量化
        self.max_seq = 512
        self.truncate_method = "head"

        # batch化相关
        self.sort_within_batch = False
        self.batch_size = 64

        # 模型结构相关
        self.transfer_hidden = 2048
        self.linear_size = 256

        # 训练速率相关
        self.epochs = 50
        self.lr = 5e-5
        self.dropout = 0.5
        self.weight_decay = 5e-4
        self.warm_up_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 5
        self.schedule_per_batches = 200

        # 训练时验证相关
        self.improve_require = 50000
        self.eval_per_batches = 200
        self.f1_average = "macro"

        # 待计算赋值的全局变量
        self.classes = None
        self.num_classes = None
        self.feature_cols = None
        self.num_labels = None

        if self.debug:
            self.debug_set()

    def build_optimizer_scheduler(self, model, train_batches_len, **kwargs):
        return super(Config, self).build_optimizer_scheduler(
            model=model,
            train_batches_len=train_batches_len,
            weight_decay=self.weight_decay,
            lr=self.lr,
            adam_epsilon=self.adam_epsilon,
            epochs=self.epochs,
            schedule_per_batches=self.schedule_per_batches,
            warm_up_proportion=self.warm_up_proportion
        )


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._if_infer = False

        self.classes = config.classes
        self.num_classes = config.num_classes
        self.device = config.device
        self.f1_average = config.f1_average
        self.num_labels = config.num_labels
        self.feature_cols = config.feature_cols

        self.encoder = AlbertModel.from_pretrained(config.transfer_path)
        [setattr(param, "requires_grad", True) for param in self.encoder.parameters()]

        self.fc = nn.Sequential(
            nn.BatchNorm1d(config.transfer_hidden),
            nn.Linear(config.transfer_hidden, config.linear_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(config.linear_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.linear_size, self.num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = \
            inputs[:-len(self.feature_cols)], inputs[-len(self.feature_cols):]

        seq_out, pooled_out = self.encoder(seq_ids, attention_mask=seq_mask)
        logits = self.fc(pooled_out)

        if labels is None:
            return dict({"logits": logits})
        return dict({
            "logits": logits,
            "loss": self.criterion(logits, labels[0]),
            "f1": calc_f1(logits, labels[0], self.classes, average=self.f1_average)
        })
