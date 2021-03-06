# @File: transfer_sfu
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/3/5 10:27:39


import os
import torch
from torch import nn
from models.base_model import BaseConfig, ExclusiveUnit
from data_loader.transfer_loader import TrainLoader
from utils.path_util import abspath
from custom_modules.fusions import SfuCombiner
from custom_modules.albert import AlbertModel


class Config(BaseConfig):
    def __init__(self, debug=False):
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
        self.header = 0
        self.sep = ","
        self.encoding = "utf-8"
        self.if_lower = True

        # subword长度、维度、transfer_path
        self.max_seq = 1024
        self.subword_embed_dim = 128
        self.fine_tune_embed = True
        self.truncate_method = "head_tail"

        # batch化相关
        self.sort_within_batch = False
        self.batch_size = 32

        # 模型结构
        self.rnn_hidden_size = 256
        self.rnn_hidden_layers = 1
        self.unit_linear_size = 128

        # 训练速率相关
        self.epochs = 50
        self.lr = 5e-5
        self.dropout = 0.5
        self.weight_decay = 5e-4
        self.warm_up_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 5
        self.schedule_per_batches = 400

        # 训练时验证相关
        self.improve_require = 80000
        self.eval_per_batches = 400
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

        # 只取其词嵌入
        albert = AlbertModel.from_pretrained(config.transfer_path)
        [setattr(param, "requires_grad", False) for param in albert.parameters()]
        self.subword_embedding = albert.embeddings.word_embeddings
        if config.fine_tune_embed:
            [setattr(param, "requires_grad", True) for param in self.subword_embedding.parameters()]

        self.encoder = nn.LSTM(config.subword_embed_dim, hidden_size=config.rnn_hidden_size, bias=True,
                               batch_first=True, bidirectional=True, num_layers=config.rnn_hidden_layers)
        self.fusion_model = SfuCombiner(config.rnn_hidden_size * 2, config.rnn_hidden_size * 2)

        self.units = nn.ModuleList()
        for idx in range(self.num_labels):
            unit = ExclusiveUnit(
                config.rnn_hidden_size * 2,
                config.num_classes,
                config.unit_linear_size,
                dropout=config.dropout
            )
            self.add_module(f"exclusive_unit_{idx}", unit)
            self.units.append(unit)

    @staticmethod
    def self_soft_attn_align(encoded_text, inf_mask):
        attention = torch.matmul(encoded_text, encoded_text.transpose(1, 2))
        self_attn = torch.softmax(attention + inf_mask.unsqueeze(1), dim=-1)
        self_align = torch.matmul(self_attn, encoded_text)
        return self_align

    def forward(self, inputs):
        labels, (word_ids, seq_len, seq_mask, inf_mask) = \
            inputs[:-len(self.feature_cols)], inputs[-len(self.feature_cols):]

        embed_seq = self.subword_embedding(word_ids)
        encoded_seq, _ = self.encoder(embed_seq)
        self_attn_seq = self.self_soft_attn_align(encoded_seq, inf_mask)
        fusion_seq = self.fusion_model(encoded_seq, self_attn_seq)

        if labels is None:
            self._if_infer = True
            labels = [None] * self.num_labels

        total_logits, total_loss, total_f1 = list(), list(), list()
        for idx, (unit, label) in enumerate(zip(self.units, labels)):
            logits, criterion, f1 = unit(
                fusion_seq,
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
