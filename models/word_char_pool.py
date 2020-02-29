# @File: word_char_pool
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/28 14:06:48


import os
from torch import nn
from models.base_model import BaseConfig, ExclusiveUnit
from data_loader.word_char_loader import TrainLoader
from utils.path_util import abspath
from custom_modules.embeddings import WordCharEmbeddingWithCnn
from custom_modules.attention import SelfAttnMatch
from custom_modules.fusions import BasicSfu, SfuCombiner


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

        # 分词相关、向量化
        self.max_seq = 512
        self.stop_dict = abspath("library/stop_symbols.dict")
        self.user_dict = None
        self.word_max_vocab = 40000
        self.word_pad_token = "<pad>"
        self.word_unk_token = "<unk>"
        self.char_max_vocab = 5000
        self.char_pad_token = "<pad>"
        self.char_unk_token = "<unk>"
        self.char_limit = 5
        self.truncate_method = "head"

        # 词嵌入相关
        self.w2v_path = os.path.join(self.data_cache, f"w2v.txt")
        self.word_w2v = os.path.join(self.data_cache, f"word.w2v.txt")
        self.word_embed_dim = 128
        self.word_window = 8
        self.word_min_count = 2
        self.word_iterations = 40
        self.word_embed_path = os.path.join(self.data_cache, f"word.embed.matrix")
        self.char_embed_path = os.path.join(self.data_cache, f"char.embed.matrix")
        self.char_w2v = os.path.join(self.data_cache, f"char.w2v.txt")
        self.char_embed_dim = 128
        self.char_window = 8
        self.char_min_count = 2
        self.char_iterations = 40
        self.char_channels = 64
        self.char_ks = [2, 3]
        self.highway_layers = 3
        self.positional_encoding = True

        # batch化相关
        self.sort_within_batch = False
        self.batch_size = 32

        # 模型结构相关
        self.hidden_dim = 256
        self.bidirectional = True
        self.hidden_layers = 1
        self.linear_dim = 128

        # 训练速率相关
        self.epochs = 60
        self.lr = 1e-5
        self.dropout = 0.5
        self.weight_decay = 1e-4
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
        self.word_embed = None
        self.char_embed = None
        self.word_vocab_size = None
        self.char_vocab_size = None
        self.feature_cols = None

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
        super(Model, self).__init__()
        self._if_infer = False

        self.classes = config.classes
        self.num_classes = config.num_classes
        self.device = config.device
        self.f1_average = config.f1_average
        self.num_labels = config.num_labels
        self.features_len = len(config.feature_cols)

        self.embedding = WordCharEmbeddingWithCnn(
            word_vocab_size=config.word_vocab_size,
            word_embed_size=config.word_embed_dim,
            char_vocab_size=config.char_vocab_size,
            char_embed_size=config.char_embed_dim,
            n_channel=config.char_channels,
            kernel_sizes=config.char_ks,
            highway_layers=config.highway_layers,
            positional_encoding=config.positional_encoding,
            max_seq=config.max_seq
        )
        self.embedding.load_pre_trained_embeddings(config.word_embed, config.char_embed)

        embed_out_size = config.word_embed_dim + config.char_channels * len(config.char_ks)
        self.encoder = nn.LSTM(embed_out_size, hidden_size=config.hidden_dim, bias=True, batch_first=True,
                               bidirectional=config.bidirectional, num_layers=config.hidden_layers)

        if config.bidirectional:
            hidden_size = config.hidden_dim * 2
        else:
            hidden_size = config.hidden_dim
        self.self_attn = SelfAttnMatch(hidden_size)
        self.fusion_model = SfuCombiner(hidden_size, hidden_size)

        self.units = nn.ModuleList()
        for idx in range(self.num_labels):
            unit = ExclusiveUnit(
                hidden_size,
                config.num_classes,
                config.linear_dim,
                dropout=config.dropout
            )
            self.add_module(f"exclusive_unit_{idx}", unit)
            self.units.append(unit)

    def forward(self, inputs):
        labels, (word_ids, char_ids, seq_len, seq_mask, inf_mask) = inputs[:-self.features_len], \
                                                                    inputs[-self.features_len:]

        embed_seq = self.embedding(word_ids, char_ids)
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
