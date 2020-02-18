# @File: data_loader
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/1/13 12:32:34

import os
import numpy as np
from utils.path_util import abspath
import pandas as pd
from itertools import islice
from utils.text_util import t2s, full2half
from utils.path_util import serialize, deserialize
from torchtext.data import Field, LabelField, Example, Dataset, BucketIterator
import torch
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from base_tokenizer import BaseTokenizer


class BatchWrapper(object):
    def __init__(self, batch_iter, columns, count):
        self.batch_iter = batch_iter
        self.columns = columns
        self.count = count

    def __iter__(self):
        for batch in self.batch_iter:
            yield [getattr(batch, column) for column in self.columns]

    def __len__(self):
        return len(self.batch_iter)


class BaseLoader(object):
    def __init__(self):
        # 我们的数据处理包括分词、映射转换均使用自己的方法，不依赖torchtext，故设置sequential=False
        # self.long_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
        # self.float_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        # self.label_field = LabelField(sequential=False, use_vocab=True, batch_first=True)
        pass

    def text_process(self, text, config):
        raise NotImplementedError

    def init_raw(self, config):
        raise NotImplementedError

    @staticmethod
    def truncate_single(iter_single, max_seq, method="head"):
        """
        假设当前某可迭代单条句子大于最大长度需要截断，可能从前往后，从后往前，从两端往中间
        :param iter_single:
        :param max_seq:
        :param method: 有往前，往后，两头往中间
        :return:
        """
        if method == "head":
            return iter_single[:max_seq]
        if method == "tail":
            return iter_single[-max_seq:]
        if method == "head_tail":
            head_len, tail_len = int(max_seq / 2), max_seq - int(max_seq / 2)
            return iter_single[:head_len] + iter_single[-tail_len:]
        raise ValueError(f"error method: {method}")

    @staticmethod
    def truncate_pair(iter_a, iter_b, max_seq):
        """
        假设是句子对，一般都是从头开始缩减
        :param iter_a:
        :param iter_b:
        :param max_seq:
        :return:
        """
        while True:
            if len(iter_a) + len(iter_b) <= max_seq:
                break
            if len(iter_a) > len(iter_b):
                iter_a.pop()
            else:
                iter_b.pop()
        return iter_a, iter_b

    @staticmethod
    def df2ds(df, fields, columns):
        examples = [Example.fromlist(record, fields) for record in zip(*[getattr(df, column) for column in columns])]
        return Dataset(examples, fields)

    @staticmethod
    def stat_df(df, n=10):
        print(f"df len: {len(df)}\ncolumns: {df.columns.tolist()}\n")
        for row in islice(df.itertuples(), 0, n):
            print(row.content)
            print("#############################################################################################\n")

    @staticmethod
    def stat_seq(df, max_seq, seq_column="content"):
        # 计算最大长度覆盖
        df['seq_len'] = df[seq_column].str.len()
        long_df = df[df['seq_len'] > max_seq]
        print(f"origin len: {len(df)}, long len: {len(long_df)}, ratio: {float(len(long_df)) / float(len(df))}")
        print(getattr(df.iloc[0:1], seq_column))

        # 画长度统计图
        plt.figure(figsize=(12, 8))
        plt.hist(df['seq_len'], bins=200, range=[200, 1800], color="green", density=True, label='train')
        plt.title('Normalized histogram of words count', fontsize=15)
        plt.legend()
        plt.xlabel('length of sequence', fontsize=15)
        plt.ylabel('Probability', fontsize=15)
        plt.grid()
        plt.show()

    @staticmethod
    def calc_inf_mask(ids):
        """
        在做attention with mask时解决soft max无法直接计算，引入该mask
        :param ids:
        :return:
        """
        if not torch.is_tensor(ids):
            return [float("-inf") if ele == 0 else 0.0 for ele in ids]
        mask = ids.eq(0)
        return torch.masked_fill(mask.float(), mask, float("-inf"))


class XlnetLoader(BaseLoader):
    def __init__(self, config):
        super().__init__()
        pandarallel.initialize(nb_workers=config.nb_workers, verbose=0)
        if not os.path.exists(config.dl_path):
            train_df, valid_df, fields, label_field, columns = self.init_raw(config)
        else:
            train_df, valid_df, fields, label_field, columns = deserialize(config.dl_path)
        train_ds, valid_ds = self.df2ds(train_df, fields, columns), self.df2ds(valid_df, fields, columns)
        label_field.build_vocab(train_ds)

        train_iter, valid_iter = BucketIterator.splits(
            (train_ds, valid_ds),
            batch_size=config.batch_size,
            device=config.device,
            sort_within_batch=config.sort_within_batch,
            sort=False,
            sort_key=lambda sample: getattr(sample, "seq_len"),
            repeat=False
        )
        self.train_batches = BatchWrapper(train_iter, columns, len(train_ds))
        self.valid_batches = BatchWrapper(valid_iter, columns, len(valid_ds))

        config.classes = list(label_field.vocab.stoi.values())
        config.num_classes = len(config.classes)
        config.num_labels = len(columns) - 4

    def text_process(self, text, config):
        text = t2s(text)
        text = full2half(text)
        text = text.strip('"').strip()
        text = text.replace("\n", "").replace("\r", "")

        tokens = config.tokenizer.tokenize_and_index(text)
        seq_ids = config.tokenizer.convert_tokens_to_ids(tokens)
        if len(seq_ids) < config.max_seq:
            seq_len = len(seq_ids)
            seq_mask = [1] * len(seq_ids) + [0] * (config.max_seq - len(seq_ids))
            seq_ids = seq_ids + [0] * (config.max_seq - len(seq_ids))
        else:
            seq_len = config.max_seq
            seq_mask = [1] * config.max_seq
            seq_ids = self.truncate_single(seq_ids, config.max_seq, method="head_tail")
        return seq_ids, seq_len, seq_mask, self.calc_inf_mask(seq_ids)

    def init_raw(self, config):
        long_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
        float_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        label_field = LabelField(sequential=False, use_vocab=True, batch_first=True)

        if not config.debug:
            train_df = pd.read_csv(config.train_file, header=0, sep=",")
            valid_df = pd.read_csv(config.valid_file, header=0, sep=",")
            for df in (train_df, valid_df):
                df["seq_ids"], df["seq_len"], df["seq_mask"], df["inf_mask"] = zip(
                    *df["content"].parallel_apply(lambda content: self.text_process(content, config)))
                df.drop(columns=["id", "content"], inplace=True)
        else:
            train_df = pd.read_csv(config.train_file, header=0, sep=",", nrows=4)
            valid_df = pd.read_csv(config.valid_file, header=0, sep=",", nrows=2)
            for df in (train_df, valid_df):
                df["seq_ids"], df["seq_len"], df["seq_mask"], df["inf_mask"] = zip(
                    *df["content"].apply(lambda content: self.text_process(content, config)))
                df.drop(columns=["id", "content"], inplace=True)

        fields, columns = list(), train_df.columns.tolist()
        for column in columns:
            if column in ("seq_ids", "seq_len", "seq_mask"):
                fields.append((column, long_field))
            elif column in ("inf_mask",):
                fields.append((column, float_field))
            else:
                fields.append((column, label_field))
        serialize(config.dl_path, [train_df, valid_df, fields, label_field, columns])
        return train_df, valid_df, fields, label_field, columns


class TrainLoader(object):
    def __init__(self, config):
        pandarallel.initialize(nb_workers=config.nb_workers, verbose=0)
        self.config = config

        if os.path.exists(self.config.dl_path):
            train_df, valid_df, fields, label_field, columns, embedding_matrix = deserialize(self.config.dl_path)
        else:
            train_df, valid_df, fields, label_field, columns, embedding_matrix = self.work()
        train_ds, valid_ds = self.df2ds(train_df, fields, columns), self.df2ds(valid_df, fields, columns)
        label_field.build_vocab(train_ds)
        train_iter, valid_iter = BucketIterator.splits(
            (train_ds, valid_ds),
            batch_size=config.batch_size,
            device=config.device,
            sort_within_batch=config.sort_within_batch,
            sort=False,
            sort_key=lambda sample: getattr(sample, "seq_len"),
            repeat=False
        )
        self.train_batches = BatchWrapper(train_iter, columns, len(train_ds))
        self.valid_batches = BatchWrapper(valid_iter, columns, len(valid_ds))

        config.classes = list(label_field.vocab.stoi.values())
        config.num_classes = len(config.classes)
        config.embedding_matrix = embedding_matrix
        config.num_labels = len(columns) - 4

    @staticmethod
    def df2ds(df, fields, columns):
        examples = [Example.fromlist(record, fields) for record in zip(*[getattr(df, column) for column in columns])]
        return Dataset(examples, fields)

    @staticmethod
    def calc_inf_mask(ids):
        """
        在做attention with mask时解决soft max无法直接计算，引入该mask
        :param ids:
        :return:
        """
        if not torch.is_tensor(ids):
            return [float("-inf") if ele == 0 else 0.0 for ele in ids]
        mask = ids.eq(0)
        return torch.masked_fill(mask.float(), mask, float("-inf"))

    @staticmethod
    def truncate_single(iter_single, max_seq, method="head"):
        """
        假设当前某可迭代单条句子大于最大长度需要截断，可能从前往后，从后往前，从两端往中间
        :param iter_single:
        :param max_seq:
        :param method: 有往前，往后，两头往中间
        :return:
        """
        if method == "head":
            return iter_single[:max_seq]
        if method == "tail":
            return iter_single[-max_seq:]
        if method == "head_tail":
            head_len, tail_len = int(max_seq / 2), max_seq - int(max_seq / 2)
            return iter_single[:head_len] + iter_single[-tail_len:]
        raise ValueError(f"error method: {method}")

    @staticmethod
    def truncate_pair(iter_a, iter_b, max_seq):
        """
        假设是句子对，一般都是从头开始缩减
        :param iter_a:
        :param iter_b:
        :param max_seq:
        :return:
        """
        while True:
            if len(iter_a) + len(iter_b) <= max_seq:
                break
            if len(iter_a) > len(iter_b):
                iter_a.pop()
            else:
                iter_b.pop()
        return iter_a, iter_b

    @staticmethod
    def _pretreatment(text):
        """
        主要是预处理文本，或者其他去除特殊字符的处理
        :param text:
        :return:
        """
        text = t2s(text)
        text = full2half(text)
        return text

    def read_raw(self):
        if self.config.debug:
            train_df = pd.read_csv(self.config.train_file, header=0, sep=self.config.delimiter, nrows=4)
            valid_df = pd.read_csv(self.config.valid_file, header=0, sep=self.config.delimiter, nrows=2)
        else:
            train_df = pd.read_csv(self.config.train_file, header=0, sep=self.config.delimiter)
            valid_df = pd.read_csv(self.config.valid_file, header=0, sep=self.config.delimiter)
        if self.config.shuffle:
            train_df = train_df.sample(frac=1)
            valid_df = valid_df.sample(frac=1)
        return train_df, valid_df

    def read_raw_single(self):
        if self.config.debug:
            df = pd.read_csv(self.config.data_file, header=0, sep=self.config.delimiter, nrows=6)
            train_df, valid_df = train_test_split(df, train_size=4, shuffle=self.config.shuffle,
                                                  random_state=self.config.seed)
        else:
            df = pd.read_csv(self.config.data_file, header=0, sep=self.config.delimiter)
            train_df, valid_df = train_test_split(df, train_size=0.7, shuffle=self.config.shuffle,
                                                  random_state=self.config.seed)
        return train_df, valid_df

    def pretreatment_text(self, train_df, valid_df):
        if self.config.debug:
            for df in (train_df, valid_df):
                df[self.config.premise] = df[self.config.premise].apply(self._pretreatment)
        else:
            for df in (train_df, valid_df):
                df[self.config.premise] = df[self.config.premise].parallel_apply(self._pretreatment)
        return train_df, valid_df

    def tokenize_and_index(self, train_df, valid_df):
        tokenizer = BaseTokenizer(
            max_vocab=self.config.max_vocab,
            pad_token=self.config.pad_token,
            unk_token=self.config.unk_token,
            pad_id=self.config.pad_id,
            unk_id=self.config.unk_id,
            stop_words_file=self.config.stop_dict
        )

        # 开始分词
        if self.config.debug:
            for df in (train_df, valid_df):
                df["tokens"] = df[self.config.premise].apply(tokenizer.tokenize)
        else:
            for df in (train_df, valid_df):
                df["tokens"] = df[self.config.premise].parallel_apply(tokenizer.tokenize)

        # 建立词典
        # tokenizer.build_vocab(train_df["tokens"], valid_df["tokens"])
        tokenizer.build_vocab(train_df["tokens"])

        # 转化为ids
        if self.config.debug:
            for df in (train_df, valid_df):
                df["seq_ids"] = df["tokens"].apply(tokenizer.convert_tokens_to_ids)
        else:
            for df in (train_df, valid_df):
                df["seq_ids"] = df["tokens"].parallel_apply(tokenizer.convert_tokens_to_ids)

        return train_df, valid_df, tokenizer

    def handle_embedding(self, train_df=None, tokenizer=None):
        # 如果没有已训练好的w2v模型，则重新生成，或者也可以用外部词向量-比如腾讯词向量代替
        if not os.path.exists(self.config.w2v_path):
            w2v_model = Word2Vec(train_df["tokens"], size=self.config.embed_dim, window=10, min_count=1, workers=4,
                                 iter=15, seed=self.config.seed, max_vocab_size=self.config.max_vocab)
            w2v_model.wv.save_word2vec_format(self.config.w2v_path, binary=False)
        else:
            w2v_model = KeyedVectors.load_word2vec_format(self.config.w2v_path, binary=False)
        embedding_matrix = np.zeros((len(tokenizer.word2index), self.config.embed_dim))
        for word, idx in tokenizer.word2index.items():
            if word == self.config.pad_token:
                embed_vector = np.zeros((1, self.config.embed_dim))
            elif word == self.config.unk_token:
                embed_vector = torch.normal(mean=0.0, std=1.0 / np.sqrt(self.config.embed_dim),
                                            size=(1, self.config.embed_dim)).numpy()
            elif word in w2v_model:
                embed_vector = w2v_model[word]
            else:
                embed_vector = torch.normal(mean=0.0, std=1.0 / np.sqrt(self.config.embed_dim),
                                            size=(1, self.config.embed_dim)).numpy()
            embedding_matrix[idx] = embed_vector
        return embedding_matrix

    def _pad_and_truncate(self, seq_ids):
        if len(seq_ids) < self.config.max_seq:
            seq_len = len(seq_ids)
            seq_mask = [1] * len(seq_ids) + [0] * (self.config.max_seq - len(seq_ids))
            seq_ids = seq_ids + [0] * (self.config.max_seq - len(seq_ids))
        else:
            seq_len = self.config.max_seq
            seq_mask = [1] * self.config.max_seq
            seq_ids = self.truncate_single(seq_ids, self.config.max_seq, method="head")
        return seq_ids, seq_len, seq_mask, self.calc_inf_mask(seq_ids)

    def pad_and_truncate(self, train_df, valid_df):
        # 将序列进行补长、截断、生成掩码等，drop掉不再需要的列
        if self.config.debug:
            for df in (train_df, valid_df):
                df["seq_ids"], df["seq_len"], df["seq_mask"], df["inf_mask"] = zip(
                    *df["seq_ids"].apply(self._pad_and_truncate))
                df.drop(columns=["id", "content", "tokens"], inplace=True)
        else:
            for df in (train_df, valid_df):
                df["seq_ids"], df["seq_len"], df["seq_mask"], df["inf_mask"] = zip(
                    *df["seq_ids"].parallel_apply(self._pad_and_truncate))
                df.drop(columns=["id", "content", "tokens"], inplace=True)
        return train_df, valid_df

    def serialize_data(self, train_df, valid_df, embedding_matrix):
        long_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
        float_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        label_field = LabelField(sequential=False, use_vocab=True, batch_first=True)
        fields, columns = list(), train_df.columns.tolist()
        for column in columns:
            if column in ("seq_ids", "seq_len", "seq_mask"):
                fields.append((column, long_field))
            elif column in ("inf_mask",):
                fields.append((column, float_field))
            else:
                fields.append((column, label_field))
        serialize(self.config.dl_path, (train_df, valid_df, fields, label_field, columns, embedding_matrix))
        return train_df, valid_df, fields, label_field, columns, embedding_matrix

    def work(self):
        train_df, valid_df = self.read_raw()
        train_df, valid_df = self.pretreatment_text(train_df, valid_df)
        train_df, valid_df, tokenizer = self.tokenize_and_index(train_df, valid_df)
        embedding_matrix = self.handle_embedding(train_df, tokenizer)
        train_df, valid_df = self.pad_and_truncate(train_df, valid_df)
        return self.serialize_data(train_df, valid_df, embedding_matrix)


if __name__ == '__main__':
    class Config(object):
        def __init__(self):
            # 随机种子
            self.seed = 279

            # pandas并行数
            self.nb_workers = 4

            # 是否为debug模式
            self.debug = True

            # 读取、解析初始数据
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

            # torch-text化相关
            self.dl_path = abspath("data/test_data.pt")

            # 待计算赋值的全局变量
            self.classes = None
            self.num_classes = None
            self.num_labels = None
            self.embedding_matrix = None

            # batch化相关
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.batch_size = 2
            self.sort_within_batch = False


    config_ = Config()
    loader = TrainLoader(config_)
