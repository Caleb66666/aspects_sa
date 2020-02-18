# @File: rcnn_model
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/16 00:12:00

import os
import torch
from torch import nn
from base_config import BaseConfig
from base_loader import TrainLoader
from utils.path_util import abspath
from utils.ml_util import calc_f1
from allennlp.nn.util import masked_softmax


class Config(BaseConfig):
    def __init__(self, debug=False):
        # 随机种子
        self.seed = 279

        # pandas并行数
        self.nb_workers = 4

        # 读取、解析初始数据
        self.loader_cls = TrainLoader
        self.train_file = abspath("data/train.csv")
        self.valid_file = abspath("data/valid.csv")
        self.delimiter = ","
        self.premise = "content"
        self.hypothesis = None
        self.shuffle = True
        self.max_seq = 1200

        # 分词相关
        self.stop_dict = abspath("library/stopwords.dict")
        self.max_vocab = 20000
        self.pad_id = 0
        self.pad_token = "<pad>"
        self.unk_id = 1
        self.unk_token = "<unk>"

        # 词嵌入相关
        self.w2v_path = abspath("library/char_embeddings.txt")
        self.embed_matrix_path = abspath("library/embed_matrix.pt")
        self.embed_dim = 128

        # 待计算赋值的全局变量
        self.classes = None
        self.num_classes = None
        self.num_labels = None
        self.embedding_matrix = None

        # batch化相关
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sort_within_batch = False
        self.batch_size = 64

        # 模型结构相关
        self.hidden_dim = 128

        # 训练速率相关
        self.epochs = 80
        self.lr = 8e-5
        self.dropout = 0.5
        self.weight_decay = 1e-2
        self.warm_up_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 5
        self.schedule_per_batches = 200

        # 训练时验证相关
        self.improve_require = 50000
        self.eval_per_batches = 200
        self.f1_average = "macro"

        super(Config, self).__init__(os.path.basename(__file__).split(".")[0], debug)

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

        self.tokens_embedding = nn.Embedding(config.embedding_matrix.shape[0], config.embedding_matrix.shape[1])
        self.tokens_embedding.from_pretrained(torch.from_numpy(config.embedding_matrix))
        self.tokens_embedding.weight.requires_grad = True

        self.encoder = nn.LSTM(config.embed_dim, hidden_size=config.hidden_dim, bias=True, bidirectional=True,
                               batch_first=True)

        self.units = nn.ModuleList()
        for idx in range(self.num_labels):
            unit = ExclusiveUnit(config.embed_dim + config.hidden_dim * 2, self.num_classes, dropout=config.dropout)
            self.add_module(f"exclusive_unit_{idx}", unit)
            self.units.append(unit)

    def forward(self, inputs):
        labels, (seq_ids, seq_len, seq_mask, inf_mask) = inputs[:-4], inputs[-4:]

        embed_seq = self.tokens_embedding(seq_ids)
        encoded_seq, _ = self.encoder(embed_seq)
        encoded_seq = torch.cat([embed_seq, encoded_seq], dim=-1)
        encoded_seq = torch.relu(encoded_seq)

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


class ExclusiveUnit(nn.Module):
    def __init__(self, input_dim, out_dim, dropout=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.attn_pool = AttnPool(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_dim * 3, out_dim)

    @staticmethod
    def avg_pool_on_seq(tensor):
        """
        一般tensor为三维矩阵，pool的层级一般是seq_len层级
        :param tensor:
        :return:
        """
        avg_p = torch.avg_pool1d(tensor.transpose(1, 2), tensor.size(1)).squeeze(-1)
        return avg_p

    @staticmethod
    def max_pool_on_seq(tensor):
        """
        一般tensor为三维矩阵，pool的层级一般是seq_len层级
        :param tensor:
        :return:
        """
        max_p = torch.max_pool1d(tensor.transpose(1, 2), tensor.size(1)).squeeze(-1)
        return max_p

    def forward(self, encoded_seq, seq_mask, label=None, classes=None, average=None):
        avg_pool = self.avg_pool_on_seq(encoded_seq)
        max_pool = self.max_pool_on_seq(encoded_seq)
        attn_pool = self.attn_pool(encoded_seq, seq_mask)
        concat_pool = torch.cat([avg_pool, max_pool, attn_pool], dim=1)
        dropped_pool = self.dropout(concat_pool)
        logits = self.dense(dropped_pool)
        if label is None:
            return logits, None, None
        return logits, self.criterion(logits, label), calc_f1(logits, label, classes, average=average)


class AttnPool(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self._weight_vector = nn.Parameter(torch.zeros(size=(input_size,)), requires_grad=True)
        self._bias = nn.Parameter(torch.zeros(size=(1,)), requires_grad=True)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, input_x, x_mask):
        activated_x = torch.tanh(input_x)
        attn_value = torch.matmul(activated_x, self._weight_vector)
        normalized_attn = masked_softmax(attn_value, x_mask, dim=1)
        weighted_x = input_x * normalized_attn.unsqueeze(-1)
        pooled_x = torch.sum(weighted_x, dim=1)
        return pooled_x
