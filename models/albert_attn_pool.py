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
        self.eval_per_batches = 100
        self.improve_require = 20000

        # 训练样本中，小于1024长度的样本数占据约98.3%，过长则截断
        self.max_seq = 1024
        self.epochs = 10
        self.batch_size = 2
        self.dropout = 0.5
        self.embed_dim = 128
        self.encode_hidden = 512
        self.linear_size = 512

        self.lr = 5e-5
        self.weight_decay = 1e-2
        self.warm_up_steps = 40
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


class OldModel(nn.Module):
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
                Attn(config.encode_hidden * 2),
                nn.Linear(config.encode_hidden * 2, config.linear_size),
                nn.BatchNorm1d(config.linear_size),
                nn.ELU(inplace=True),
                nn.Dropout(config.dropout),
                nn.Linear(config.linear_size, self.num_classes)).to(config.device)
            self.units.append(unit)
            self.criterion_list.append(nn.CrossEntropyLoss().to(config.device))

    def forward(self, inputs):
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = inputs[:-4], inputs[-4:]
        if labels:
            assert len(labels) == self.num_labels, "number labels error!"

        embed_seq = self.embedding(seq_ids)
        encoded_seq, _ = self.encoder(embed_seq)

        total_logits, total_loss, total_f1 = list(), 0.0, 0.0
        for idx, (unit, criterion) in enumerate(zip(self.units, self.criterion_list)):
            logits = unit(encoded_seq)
            total_logits.append(logits)
            if labels:
                total_loss += criterion(logits, labels[idx])
                total_f1 += calc_f1(logits, labels[idx], self.classes, average="macro")

        output_dict = {
            "logits": total_logits
        }
        if labels:
            output_dict.update({
                "f1": total_f1 / self.num_labels,
                "loss": total_loss / self.num_labels
            })
        return output_dict


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.name_classes = config.num_classes
        self.classes = config.classes

        # 只取其词嵌入
        albert = AlbertModel.from_pretrained(config.albert_path)
        [setattr(param, "requires_grad", False) for param in albert.parameters()]
        self.embedding = albert.embeddings.word_embeddings
        [setattr(param, "requires_grad", True) for param in self.embedding.parameters()]

        self.bn = nn.BatchNorm2d(config.max_seq, config.embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.linear_size),
            nn.BatchNorm1d(config.linear_size),
            nn.ELU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.linear_size, config.num_classes)
        )

        self.soft_max = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss().to(config.device)

    @staticmethod
    def avg_max_pooling(tensor):
        avg_p = torch.avg_pool1d(tensor.transpose(1, 2), tensor.size(1)).squeeze(-1)
        max_p = torch.max_pool1d(tensor.transpose(1, 2), tensor.size(1)).squeeze(-1)
        return torch.cat([avg_p, max_p], dim=1)

    def forward(self, inputs):
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = inputs[:-4], inputs[-4:]

        label = labels[0]
        embed_seq = self.embedding(seq_ids)
        print(embed_seq.size())

        pooled_seq = self.avg_max_pooling(embed_seq)
        print(pooled_seq.size())

        logits = self.fc(pooled_seq)
        print(logits.size())
        loss = self.criterion(logits, label)
        f1 = calc_f1(logits, label, classes=self.classes, average="macro")
        output_dict = {
            "logits": logits,
            "f1": f1,
            "loss": loss
        }
        return output_dict
