# @File: fix_len_model
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/12 10:50:06

import os
import torch
from torch import nn
import numpy as np
from base_config import BaseConfig
from base_loader import XlnetLoader as FixLenLoader
from utils.path_util import abspath
from utils.ml_util import calc_f1
from self_modules.albert import AlbertTokenizer, AlbertModel
from self_modules.transfer_embedding import TransferEmbedding
from torch.nn.parameter import Parameter
from overrides import overrides
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.nn import Activation


class Config(BaseConfig):
    def __init__(self, seed, debug=False):
        self.train_file = abspath("data/train.csv")
        self.valid_file = abspath("data/valid.csv")
        self.loader_cls = FixLenLoader

        self.improve_require = 400000
        self.eval_per_batches = 200
        self.schedule_per_batches = 200

        self.num_classes = None
        self.num_labels = None
        self.classes = None
        self.f1_average = "macro"

        self.epochs = 80
        self.max_seq = 1024
        self.batch_size = 64
        self.embed_dim = 128
        self.hidden_dim = 128
        self.aspects_dim = 32
        self.num_layers = 1
        self.bidirectional = True

        self.lr = 8e-5
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


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._if_infer = False

        self.classes = config.classes
        self.num_classes = config.num_classes
        self.num_labels = config.num_labels
        self.device = config.device
        self.f1_average = config.f1_average

        self.aspects_embedding = nn.Embedding(self.num_labels, config.aspects_dim)
        self.word_embedding = TransferEmbedding(config.transfer_cls, config.transfer_path, config.embedding_attributes)
        self.encoder = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional
        )
        if config.bidirectional:
            hidden_dim = config.hidden_dim * 2
        else:
            hidden_dim = config.hidden_dim

        self.units = nn.ModuleList()
        for idx in range(self.num_labels):
            unit = ExclusiveUnit(hidden_dim, self.num_classes, config.aspects_dim, config.max_seq,
                                 dropout=config.dropout)
            self.add_module(f"exclusive_unit_{idx}", unit)
            self.units.append(unit)

        self.reset_parameters(config)

    def reset_parameters(self, config):
        with torch.no_grad():
            self.aspects_embedding.weight.normal_(mean=0.0, std=1.0 / np.sqrt(config.aspects_dim))

    def forward(self, inputs):
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = inputs[:-4], inputs[-4:]

        embed_seq = self.word_embedding(seq_ids, seq_len)
        encoded_seq, _ = self.encoder(embed_seq)

        if labels is None:
            self._if_infer = True
            labels = [None] * self.num_labels

        total_logits, total_loss, total_f1 = list(), list(), list()
        for idx, (unit, label) in enumerate(zip(self.units, labels)):
            aspect_embed = self.aspects_embedding(torch.tensor(idx).long().to(self.device))
            logits, criterion, f1 = unit(
                aspect_embed,
                encoded_seq,
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


class ExclusiveUnit(nn.Module):
    def __init__(self, hidden_dim, out_dim, aspects_dim, seq_len, dropout=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.dense = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.similarity = BiLinearSimilarity(aspects_dim, seq_len)

    def forward(self, aspect_embed, encoded_seq, label=None, classes=None, average="macro"):
        aspect_embed = aspect_embed.unsqueeze(0).expand(encoded_seq.size(0), -1)
        sim_score = self.similarity(aspect_embed, encoded_seq)
        dropped_score = self.dropout(sim_score)
        logits = self.dense(dropped_score)
        if label is None:
            return logits, None, None
        return logits, self.criterion(logits, label), calc_f1(logits, label, classes, average=average)


@SimilarityFunction.register("my-bi-linear")
class BiLinearSimilarity(SimilarityFunction):
    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 activation: Activation = None) -> None:
        super(BiLinearSimilarity, self).__init__()
        self._weight_matrix = Parameter(torch.zeros(size=(tensor_1_dim, tensor_2_dim)), requires_grad=True)
        self._bias = Parameter(torch.zeros(size=(1,)), requires_grad=True)
        self._activation = activation or Activation.by_name('linear')()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        intermediate = torch.matmul(tensor_1, self._weight_matrix)
        result = torch.bmm(tensor_2.transpose(1, 2), intermediate.unsqueeze(-1)).squeeze()
        return self._activation(result + self._bias)
