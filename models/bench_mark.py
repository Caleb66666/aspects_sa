# @File: bench_mark
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/5 16:31:07

import os
import torch
from torch import nn
from base_config import BaseConfig
from utils.path_util import abspath
from utils.ml_util import calc_f1
from base_loader import XlnetLoader as BenchLoader
from self_modules.transfer_embedding import TransferEmbedding
from self_modules.albert import AlbertModel, AlbertTokenizer
from self_modules.dynamic_lstm import DynamicLSTM
from allennlp.modules.attention import BilinearAttention
from self_modules.attention import NoQueryAttention


class Config(BaseConfig):
    def __init__(self, seed, debug=False):
        self.train_file = abspath("data/train.csv")
        self.valid_file = abspath("data/valid.csv")
        self.loader_cls = BenchLoader
        self.average = "macro"

        self.num_classes = None
        self.num_labels = None
        self.classes = None
        self.improve_require = 40000
        self.eval_per_batches = 200
        self.schedule_per_batches = 200

        self.epochs = 100
        self.max_seq = 1024
        self.batch_size = 64
        self.embed_dim = 128
        self.hidden_dim = 128
        self.num_layers = 1
        self.bidirectional = True

        self.lr = 1e-4
        self.dropout = 0.5
        self.weight_decay = 1e-2
        self.warm_up_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 5

        if debug:
            self.transfer_path = "/Users/Vander/Code/pytorch_col/albert-base-chinese"
        else:
            self.transfer_path = "/data/wangqian/berts/albert-base-chinese"
        self.transfer_cls = AlbertModel
        self.embedding_attributes = ("embeddings", "word_embeddings")
        self.tokenizer = AlbertTokenizer.from_pretrained(self.transfer_path)
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


class SelfBiLinearAttentionWithPool(BilinearAttention):
    def __init__(self, input_dim):
        super().__init__(input_dim, input_dim)

    def forward(self, vector, **kwargs):
        similarity = super().forward(vector, vector)
        attended_vector = torch.bmm(similarity, vector)
        return torch.max_pool1d(attended_vector.transpose(1, 2), attended_vector.size(1)).squeeze(-1)

    def _forward_internal(self, vector, matrix):
        intermediate = torch.matmul(vector, self._weight_matrix)
        return self._activation(intermediate.bmm(matrix.transpose(1, 2)) + self._bias)


class SelfNoQueryAttention(NoQueryAttention):
    def __init__(self, input_dim, score_fn):
        super().__init__(input_dim, score_fn=score_fn)

    def forward(self, vector, **kwargs):
        _, similarity = super().forward(vector)
        attended_vector = torch.bmm(similarity, vector).squeeze()
        return attended_vector


class ExclusiveUnit(nn.Module):
    def __init__(self, input_dim, out_dim, dropout=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.dense = nn.Linear(input_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = SelfBiLinearAttentionWithPool(input_dim)

    def forward(self, encoded_seq, label=None, classes=None, average="micro"):
        attended_seq = self.attention(encoded_seq)
        dropped_seq = self.dropout(attended_seq)
        logits = self.dense(dropped_seq)
        if label is None:
            return logits
        return logits, self.criterion(logits, label), calc_f1(logits, label, classes, average=average)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.if_infer = False

        self.classes = config.classes
        self.num_classes = config.num_classes
        self.num_labels = config.num_labels
        self.device = config.device
        self.average = config.average

        self.embedding = TransferEmbedding(config.transfer_cls, config.transfer_path, config.embedding_attributes)
        self.encoder = DynamicLSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            sort_within_batch=config.sort_within_batch
        )
        if config.bidirectional:
            hidden_dim = config.hidden_dim * 2
        else:
            hidden_dim = config.hidden_dim
        self.units = nn.ModuleList()
        for idx in range(self.num_labels):
            unit = ExclusiveUnit(hidden_dim, self.num_classes, dropout=config.dropout)
            self.add_module(f"exclusive_unit_{idx}", unit)
            self.units.append(unit)

    def forward(self, inputs):
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = inputs[:-4], inputs[-4:]

        embed_seq = self.embedding(seq_ids, seq_len)
        encoded_seq, _ = self.encoder(embed_seq, seq_len)

        if labels is None:
            self.if_infer = True
            labels = [None] * self.num_labels

        total_logits, total_loss, total_f1 = list(), list(), list()
        for idx, (unit, label) in enumerate(zip(self.units, labels)):
            logits, criterion, f1 = unit(
                encoded_seq,
                label,
                self.classes,
                average=self.average
            )
            total_logits.append(logits), total_loss.append(criterion), total_f1.append(f1)

        if self.if_infer:
            return dict({"logits": total_logits})

        return dict({
            "logits": total_logits,
            "loss": sum(total_loss) / float(self.num_labels),
            "f1": sum(total_f1) / float(self.num_labels)
        })
