# @File: atae_lstm
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/1 17:27:09

import os
import torch
from torch import nn
from utils.ml_util import calc_f1
from layers.dynamic_lstm import DynamicLSTM
from layers.squeeze_embedding import Squeezer
from layers.attention import NoQueryAttention
from models.base_config import BaseConfig
from utils.path_util import abspath
from data_loader import XlnetLoader as AtaeLoader
from modules.albert import AlbertModel, AlbertTokenizer


class Config(BaseConfig):
    def __init__(self, seed, debug=False):
        self.train_file = abspath("data/train.csv")
        self.valid_file = abspath("data/valid.csv")
        self.loader_cls = AtaeLoader

        self.num_classes = None
        self.num_labels = None
        self.classes = None
        self.improve_require = 40000
        self.eval_per_batches = 200
        self.schedule_per_batches = 200

        self.epochs = 150
        self.max_seq = 1024
        self.batch_size = 64
        self.embed_dim = 128
        self.label_embed_dim = 16
        self.hidden_dim = 128
        self.num_layers = 1

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


class ExclusiveUnit(nn.Module):
    def __init__(self, attn_dim, score_fn, linear_dim, output_dim):
        super(ExclusiveUnit, self).__init__()

        self.attention = NoQueryAttention(attn_dim, score_fn=score_fn)
        self.dense = nn.Linear(linear_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, encoded_seq, label_embed, label=None, classes=None, average="micro"):
        label_cat_encoded = torch.cat((encoded_seq, label_embed), dim=-1)
        _, score = self.attention(label_cat_encoded)
        logits = torch.squeeze(torch.bmm(score, encoded_seq), dim=1)
        logits = self.dense(logits)
        if label is None:
            return logits, None, None
        return logits, self.criterion(logits, label), calc_f1(logits, label, classes, average=average)


class TransferEmbedding(nn.Module):
    def __init__(self, transfer_cls, transfer_path, fine_tune_embed=True):
        super().__init__()

        self.squeezer = Squeezer()
        transfer_model = transfer_cls.from_pretrained(transfer_path)
        [setattr(p, "requires_grad", False) for p in transfer_model.parameters()]
        self.embedding = transfer_model.embeddings.word_embeddings
        if fine_tune_embed:
            [setattr(p, "requires_grad", True) for p in self.embedding.parameters()]

    def forward(self, seq_ids, seq_len):
        embed_seq = self.embedding(seq_ids)
        return self.squeezer(embed_seq, seq_len)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.if_infer = False

        self.classes = config.classes
        self.num_classes = config.num_classes
        self.num_labels = config.num_labels
        self.device = config.device

        self.embedding = TransferEmbedding(AlbertModel, config.albert_path)
        self.encoder = DynamicLSTM(config.embed_dim, config.hidden_dim, config.num_layers, batch_first=True,
                                   sort_within_batch=config.sort_within_batch)
        self.label_embedding = nn.Embedding(self.num_labels, config.label_embed_dim)

        self.units = nn.ModuleList([
            ExclusiveUnit(
                attn_dim=config.hidden_dim * 2 + config.label_embed_dim,
                score_fn="bi_linear",
                linear_dim=config.hidden_dim * 2,
                output_dim=self.num_classes
            ) for _ in range(self.num_labels)
        ])

    def forward(self, inputs):
        # 解析输入项
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = inputs[:-4], inputs[-4:]
        bs, seq_max = seq_ids.size(0), torch.max(seq_len)

        embed_seq = self.embedding(seq_ids, seq_len)
        encoded_seq, _ = self.encoder(embed_seq, seq_len)

        # 判断是train or infer
        if labels is None:
            labels = [None] * self.num_labels
            self.if_infer = True

        total_logits, total_loss, total_f1 = list(), list(), list()
        for idx, (unit, label) in enumerate(zip(self.units, labels)):
            label_embed = self.label_embedding(torch.tensor(idx).long().to(self.device))
            label_embed = label_embed.unsqueeze(dim=0).expand(bs, -1).unsqueeze(1).expand(-1, seq_max.item(), -1)
            logits, criterion, f1 = unit(
                encoded_seq,
                label_embed,
                label,
                self.classes,
                average="micro"
            )
            total_logits.append(logits), total_loss.append(criterion), total_f1.append(f1)

        if self.if_infer:
            return dict({"logits": total_logits})

        return dict({
            "logits": total_logits,
            "loss": sum(total_loss) / float(self.num_labels),
            "f1": sum(total_f1) / float(self.num_labels)
        })
