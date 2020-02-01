# @File: transformer_attn_pool
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/1 09:40:51


import os
import math
import torch
from torch import nn
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import \
    BidirectionalLanguageModelTransformer as TransformerEncoder
from models.base_config import BaseConfig
from utils.path_util import abspath
from utils.ml_util import calc_f1
from data_loader import XlnetLoader as TransformerLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from chinese_albert import AlbertTokenizer, AlbertModel


class Config(BaseConfig):
    def __init__(self, seed, debug=False):
        self.train_file = abspath("data/train.csv")
        self.valid_file = abspath("data/valid.csv")
        self.loader_cls = TransformerLoader

        self.num_classes = None
        self.num_labels = None
        self.classes = None
        self.improve_require = 40000
        self.eval_per_batches = 200
        self.schedule_per_batches = 200

        self.epochs = 30
        self.max_seq = 1024
        self.batch_size = 64
        self.embed_dim = 128
        self.hidden_size = 128
        self.linear_size = 128
        self.num_layers = 3

        self.lr = 6e-5
        self.dropout = 0.5
        self.weight_decay = 1e-2
        self.warm_up_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 5

        self.albert_path = "/data/wangqian/berts/albert-base-chinese"
        if debug:
            self.albert_path = "/Users/Vander/Code/pytorch_col/albert-base-chinese"
        self.tokenizer = AlbertTokenizer.from_pretrained(self.albert_path)
        self.cls = self.tokenizer.cls_token
        self.sep = self.tokenizer.sep_token

        super(Config, self).__init__(os.path.basename(__file__).split(".")[0], seed, debug)

    def build_optimizer_scheduler(self, model, train_batches_len):
        opt_params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
            {'params': [p for n, p in opt_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in opt_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(grouped_params, lr=self.lr, eps=self.adam_epsilon)

        schedule_steps = math.ceil(train_batches_len * self.epochs / self.schedule_per_batches)
        warm_up_steps = math.ceil(schedule_steps * self.warm_up_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=schedule_steps)
        return optimizer, scheduler


class AttnPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.w = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.w.data.normal_(-1e-4, 1e-4)

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
        m = torch.tanh(h)
        alpha = torch.softmax(torch.matmul(m, self.w), dim=1).unsqueeze(-1)
        out = h * alpha
        return torch.max_pool1d(out.transpose(1, 2), out.size(1)).squeeze(-1)
        # return self.avg_max_pool(out)


class Model(nn.Module):
    def __init__(self, config):
        """
        选择albert是因为其词表征输出维度较小，而且本身该模型为蒸馏模型，训练步骤及其长，结合其基于sub word的分词方法。不仅可以比较完美的
        解决oov问题，而且还拥有维度小，表征能力强的词嵌入，结果证明，后续接入一个比较简单的双向LSTM作为序列表征，分类器使用attention+max
        pool就能获得一个较好的基线结果。
        TODO: 直观上来说，改进encoder能获得跟更大的提升
        1. pool层改进：引入avg pool和max pool
        2. attention层改进：attention添加新的w，用于调整输入项
        3. encoder部分：使用简化版elmo模型，并逐级增加encoder层级
        4. encoder部分：使用可并行rnn结构单元sru代替
        5. embedding部分：目前使用的是base版，可以试用large版
        6. encoder部分：加入局部特征conv
        :param config:
        """
        super().__init__()
        self.classes = config.classes
        self.num_classes = config.num_classes
        self.num_labels = config.num_labels

        # 只取其词嵌入
        albert = AlbertModel.from_pretrained(config.albert_path)
        [setattr(param, "requires_grad", False) for param in albert.parameters()]
        self.embedding = albert.embeddings.word_embeddings
        [setattr(param, "requires_grad", True) for param in self.embedding.parameters()]

        # 简化版的elmo模型作为特征抽取
        self.encoder = TransformerEncoder(config.embed_dim, config.hidden_size, config.num_layers, config.dropout)

        self.units, self.criterion_list = nn.ModuleList(), list()
        for _ in range(self.num_labels):
            unit = nn.Sequential(
                AttnPool(config.embed_dim * 2),
                nn.Linear(config.embed_dim * 2, config.linear_size),
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
        encoded_seq = self.encoder(embed_seq, seq_mask)  # batch_size, seq_len, embed_dim * 2
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
