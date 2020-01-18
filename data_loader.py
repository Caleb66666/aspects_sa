# @File: data_loader
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/1/13 12:32:34

from utils.path_util import abspath
import pandas as pd
from itertools import islice
from utils.text_util import t2s, full2half
from utils.path_util import serialize, deserialize
import os
from torchtext.data import Field, LabelField, Example, Dataset, BucketIterator
import torch
import matplotlib.pyplot as plt


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
        # train_df = pd.read_csv(config.train_file, header=0, sep=",")
        # self.stat_seq(train_df, config.max_seq, "content")

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
            sort_within_batch=False,
            sort=False,
            sort_key=lambda sample: getattr(sample, "seq_len"),
            repeat=False
        )
        self.train_batches = BatchWrapper(train_iter, columns, len(train_ds))
        self.valid_batches = BatchWrapper(valid_iter, columns, len(valid_ds))

        config.classes = list(label_field.vocab.stoi.keys())
        print(config.classes)
        config.num_classes = len(config.classes)
        config.num_labels = len(columns) - 4

    def text_process(self, text, config):
        text = t2s(text)
        text = full2half(text)
        text = text.strip('"').strip()
        text = text.replace("\n", "").replace("\r", "")

        tokens = config.tokenizer.tokenize(text)
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

        train_df = pd.read_csv(config.train_file, header=0, sep=",")
        valid_df = pd.read_csv(config.valid_file, header=0, sep=",")
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


if __name__ == '__main__':
    from transformers import XLNetTokenizer
    from logging import ERROR
    from transformers.tokenization_utils import logger as tokenizer_logger
    from transformers.file_utils import logger as file_logger
    from transformers.configuration_utils import logger as config_logger
    from transformers.modeling_utils import logger as model_logger

    [logger.setLevel(ERROR) for logger in (tokenizer_logger, file_logger, config_logger, model_logger)]


    class Config(object):
        def __init__(self):
            self.name = "base_xlnet"
            self.train_file = abspath("data/train.csv")
            self.valid_file = abspath("data/valid.csv")
            self.dl_path = abspath(f"data/{self.name}.pt")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.seed = 279
            self.num_classes = None
            self.num_labels = None
            self.classes = None
            self.max_seq = 768
            self.batch_size = 32

            self.xlnet_path = "/Users/Vander/Code/pytorch_col/xlnet-base-chinese"
            self.tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
            self.bos = self.tokenizer.bos_token
            self.sep = self.tokenizer.sep_token


    config_ = Config()
    loader = XlnetLoader(config_)
