# @File: albert_attn_pool
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/1/18 00:54:32


import os
import torch
from models.base_config import BaseConfig
from torch import nn
from utils.path_util import abspath
from utils.ml_util import calc_f1
from data_loader import XlnetLoader as AlbertLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from albert_zh import AlbertTokenizer, AlbertModel

from logging import ERROR
from transformers.tokenization_utils import logger as tokenizer_logger
from transformers.file_utils import logger as file_logger
from transformers.configuration_utils import logger as config_logger
from transformers.modeling_utils import logger as model_logger

[logger.setLevel(ERROR) for logger in (tokenizer_logger, file_logger, config_logger, model_logger)]


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__(os.path.basename(__file__).split(".")[0])

        self.train_file = abspath("data/train.csv")
        self.valid_file = abspath("data/valid.csv")
        self.loader_cls = AlbertLoader
        self.dl_path = abspath(f"data/{self.name}.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = None
        self.num_classes = None
        self.num_labels = None
        self.classes = None
        self.eval_per_batches = 200
        self.improve_require = 40000

        # 训练样本中，小于1024长度的样本数占据约98.3%，过长则截断
        self.max_seq = 1024
        self.epochs = 20
        self.batch_size = 64
        self.dropout = 0.5
        self.embed_dim = 128
        self.encode_hidden = 512
        self.attn_size = 128
        self.linear_size = 128

        self.lr = 5e-5
        self.weight_decay = 1e-2
        self.warm_up_steps = 50
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 5

        self.albert_path = "/data/wangqian/berts/albert-base-chinese"
        # self.albert_path = "/Users/Vander/Code/pytorch_col/albert-base-chinese"
        self.tokenizer = AlbertTokenizer.from_pretrained(self.albert_path)
        self.cls = self.tokenizer.cls_token
        self.sep = self.tokenizer.sep_token

    def build_optimizer(self, model, t_total):
        opt_params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in opt_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in opt_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(grouped_params, lr=self.lr, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warm_up_steps,
                                                    num_training_steps=int(t_total / self.eval_per_batches) + 1)
        return optimizer, scheduler


class AttnPool(nn.Module):
    def __init__(self, hidden_size, attn_size):
        super().__init__()

        self.w = nn.Parameter(torch.zeros(hidden_size, attn_size), requires_grad=True)
        self.u = nn.Parameter(torch.zeros(attn_size), requires_grad=True)
        [p.data.normal_(-1e-4, 1e-4) for p in (self.w, self.u)]

    @staticmethod
    def avg_max_pool(tensor):
        """
        一般tensor为三维矩阵，pool的层级一般是seq_len层级
        :param tensor:
        :return:
        """
        avg_p = torch.avg_pool1d(tensor.transpose(1, 2), tensor.size(1)).squeeze(-1)
        max_p = torch.max_pool1d(tensor.transpose(1, 2), tensor.size(1)).squeeze(-1)
        return torch.cat([avg_p, max_p], dim=1)

    def forward(self, h):
        squeeze_h = torch.matmul(h, self.w)
        m = torch.tanh(squeeze_h)
        alpha = torch.softmax(torch.matmul(m, self.u), dim=1).unsqueeze(-1)
        out = h * alpha
        return self.avg_max_pool(out)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classes = config.classes
        self.num_classes = config.num_classes
        self.num_labels = config.num_labels

        # 只取其词嵌入
        albert = AlbertModel.from_pretrained(config.albert_path)
        [setattr(param, "requires_grad", False) for param in albert.parameters()]
        self.embedding = albert.embeddings.word_embeddings
        [setattr(param, "requires_grad", True) for param in self.embedding.parameters()]

        self.encoder = nn.LSTM(config.embed_dim, config.encode_hidden, batch_first=True, bidirectional=True)

        self.units, self.criterion_list = nn.ModuleList(), list()
        for _ in range(self.num_labels):
            unit = nn.Sequential(
                AttnPool(config.encode_hidden * 2, config.attn_size),  # bach_size, encode_hidden * 4
                nn.Linear(config.encode_hidden * 4, config.linear_size),
                nn.BatchNorm1d(config.linear_size),
                nn.ELU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(config.linear_size, self.num_classes)).to(config.device)
            self.units.append(unit)
            self.criterion_list.append(nn.CrossEntropyLoss().to(config.device))

    def forward(self, inputs):
        # 解析输入项
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = inputs[:-4], inputs[-4:]

        # test流程
        if not labels:
            embed_seq = self.embedding(seq_ids)
            encoded_seq, _ = self.encoder(embed_seq)
            return {"logits": [unit(encoded_seq) for unit in self.units]}

        # train流程
        assert len(labels) == self.num_labels, "number labels error!"
        embed_seq = self.embedding(seq_ids)  # batch_size, seq_len, embed_dim
        encoded_seq, _ = self.encoder(embed_seq)  # batch_size, seq_len, encode_hidden * 2
        total_logits, total_loss, total_f1 = list(), 0.0, 0.0
        for unit, criterion, label in zip(self.units, self.criterion_list, labels):
            logits = unit(encoded_seq)
            total_logits.append(logits)
            total_loss += criterion(logits, label)
            total_f1 += calc_f1(logits, label, self.classes, average="micro")

        return {
            "logits": total_logits,
            "f1": total_f1 / self.num_labels,
            "loss": total_loss / self.num_labels
        }
