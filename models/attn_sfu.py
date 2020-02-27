# @File: attn_sfu
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/26 14:37:59

import os
from torch import nn
from models.base_model import BaseConfig, ExclusiveUnit
from data_loader.transfer_loader import TrainLoader
from utils.path_util import abspath
from custom_modules.albert import AlbertModel
from custom_modules.attention import SelfAttnMatch
from custom_modules.fusions import BasicSfu, SfuCombiner


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
        self.if_lower = True

        # 分词、索引相关
        self.stop_dict = abspath("library/stop_symbols.txt")
        if debug:
            self.transfer_path = "/Users/Vander/Code/pytorch_col/albert-base-chinese"
        else:
            self.transfer_path = "/data/wangqian/berts/albert-base-chinese"
        self.fine_tune_embed = True
        self.truncate_method = "head"

        # 词嵌入相关
        self.embed_dim = 128

        # batch化相关
        self.sort_within_batch = False
        self.batch_size = 32

        # 模型结构相关
        self.hidden_dim = 256
        self.bidirectional = True
        self.num_layers = 1
        self.linear_dim = 256

        # 训练速率相关
        self.epochs = 80
        self.lr = 8e-5
        self.dropout = 0.5
        self.weight_decay = 5e-4
        self.warm_up_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 5
        self.schedule_per_batches = 400

        # 训练时验证相关
        self.improve_require = 50000
        self.eval_per_batches = 400
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

        # 只取其词嵌入
        albert = AlbertModel.from_pretrained(config.transfer_path)
        [setattr(param, "requires_grad", False) for param in albert.parameters()]
        self.embedding = albert.embeddings.word_embeddings
        if config.fine_tune_embed:
            [setattr(param, "requires_grad", True) for param in self.embedding.parameters()]

        self.encoder = nn.LSTM(config.embed_dim, hidden_size=config.hidden_dim, bias=True, batch_first=True,
                               bidirectional=config.bidirectional, num_layers=config.num_layers)

        # 特征表征结构
        self.self_attn = SelfAttnMatch(config.hidden_dim * 2)
        self.fusion_model = SfuCombiner(config.hidden_dim * 2, config.hidden_dim * 2)

        self.units = nn.ModuleList()
        for idx in range(self.num_labels):
            unit = ExclusiveUnit(
                config.hidden_dim * 2,
                config.num_classes,
                config.linear_dim,
                dropout=config.dropout
            )
            self.add_module(f"exclusive_unit_{idx}", unit)
            self.units.append(unit)

    def forward(self, inputs):
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = inputs[:-4], inputs[-4:]

        embed_seq = self.embedding(seq_ids)
        encoded_seq, _ = self.encoder(embed_seq)
        self_attn_seq = self.self_attn(encoded_seq, seq_mask)
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
