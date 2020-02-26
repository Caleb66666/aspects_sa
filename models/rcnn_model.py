# @File: rcnn_model
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/18 18:29:56

import os
import torch
from torch import nn
from models.base_model import BaseConfig, ExclusiveUnit
from data_loader.char_loader import TrainLoader
from utils.path_util import abspath


class Config(BaseConfig):
    def __init__(self, debug=False):
        self.name = os.path.basename(__file__).split(".")[0]
        self.debug = debug
        self.seed = 279

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
        self.lower = True

        # 分词相关
        self.stop_dict = abspath("library/stopwords.dict")
        self.max_vocab = 20000
        self.pad_id = 0
        self.pad_token = "<pad>"
        self.unk_id = 1
        self.unk_token = "<unk>"

        # 词嵌入相关
        self.embed_dim = 128
        self.window = 10
        self.min_count = 1
        self.iterations = 20

        # batch化相关
        self.sort_within_batch = False
        self.batch_size = 64

        # 模型结构相关
        self.hidden_dim = 128
        self.bidirectional = True
        self.num_layers = 1
        self.linear_dim = 128

        # 训练速率相关
        self.epochs = 80
        self.lr = 8e-5
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
        self.num_labels = None
        self.embed_matrix = None

        super(Config, self).__init__(self.name, self.debug, self.seed)

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
        self.num_labels = config.num_labels
        self.device = config.device
        self.f1_average = config.f1_average

        self.tokens_embedding = nn.Embedding(config.embed_matrix.shape[0], config.embed_matrix.shape[1])
        self.tokens_embedding.from_pretrained(torch.from_numpy(config.embed_matrix))
        self.tokens_embedding.weight.requires_grad = True

        self.encoder = nn.LSTM(config.embed_dim, hidden_size=config.hidden_dim, bias=True, batch_first=True,
                               bidirectional=self.bidirectional, num_layers=self.num_layers)
        self.units = nn.ModuleList()
        for idx in range(self.num_labels):
            unit = ExclusiveUnit(
                config.embed_dim + config.hidden_dim * 2,
                config.num_classes,
                config.linear_dim,
                dropout=config.dropout
            )
            self.add_module(f"exclusive_unit_{idx}", unit)
            self.units.append(unit)

    def forward(self, inputs):
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = inputs[:-4], inputs[-4:]

        embed_seq = self.tokens_embedding(seq_ids)
        encoded_seq, _ = self.encoder(embed_seq)
        encoded_seq = torch.cat([embed_seq, encoded_seq], dim=-1)

        if labels is None:
            self._if_infer = True
            labels = [None] * self.num_labels

        total_logits, total_loss, total_f1 = list(), list(), list()
        for idx, (unit, label) in enumerate(zip(self.units, labels)):
            logits, criterion, f1 = unit(
                encoded_seq,
                seq_mask,
                label,
                self.classes,
                average=self.f1_average
            )
            total_logits.append(logits), total_loss.append(criterion), total_f1.append(f1)

        if self._if_infer:
            return dict({"logits": total_logits})

        return dict({
            "logits": total_logits,
            "loss": sum(total_loss) / float(self.num_labels),
            "f1": sum(total_f1) / float(self.num_labels)
        })
