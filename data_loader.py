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
import numpy as np


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


class XlLoader(object):
    def __init__(self, config):
        save_path = abspath(f"data/{config.name}.dl.pt")
        if not os.path.exists(save_path):
            self.init_raw(config, save_path)
        text_field, label_field, train_df, valid_df, fields, columns = deserialize(save_path)
        train_ds, valid_ds = self.df2ds(train_df, fields, columns), self.df2ds(valid_df, fields, columns)
        label_field.build_vocab(valid_ds)

        train_iter, valid_iter = BucketIterator.splits(
            (train_ds, valid_ds),
            batch_size=config.batch_size,
            device=config.device,
            sort_within_batch=False,
            sort=False,
            sort_key=lambda sample: getattr(sample, "seq_len"),
            repeat=False
        )
        self.train_wrapper = BatchWrapper(train_iter, columns, len(train_ds))
        self.valid_wrapper = BatchWrapper(valid_iter, columns, len(valid_ds))
        self.classes = label_field.vocab
        config.num_classes = len(self.classes)

        print(f"train len: {len(train_ds)}, valid len: {len(valid_ds)}")

    @staticmethod
    def xl_process(config, text):
        text = t2s(text)
        text = full2half(text)
        text = text.strip('"').strip()
        text = text.replace("\n", "").replace("\r", "")

        tokens = [config.bos, ]
        tokens.extend(config.tokenizer.tokenize(text))
        token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < config.max_seq:
            mask = [1] * len(token_ids) + [0] * (config.max_seq - len(token_ids))
            token_ids += ([0] * (config.max_seq - len(token_ids)))
            seq_len = len(tokens)
        else:
            mask = [1] * config.max_seq
            token_ids = token_ids[:config.max_seq]
            seq_len = config.max_seq
        return token_ids, seq_len, np.array(mask)

    def init_raw(self, config, save_path):
        field = Field(sequential=False, use_vocab=False, batch_first=True)
        label_field = LabelField(sequential=False, use_vocab=True, batch_first=True)

        train_df = pd.read_csv(config.train_file, header=0, sep=",")
        valid_df = pd.read_csv(config.valid_file, header=0, sep=",")
        train_df["content"], train_df["seq_len"], train_df["seq_mask"] = zip(*train_df["content"].apply(
            lambda text: self.xl_process(config, text)))
        valid_df["content"], valid_df["seq_len"], valid_df["seq_mask"] = zip(*valid_df["content"].apply(
            lambda text: self.xl_process(config, text)))
        del train_df["id"], valid_df["id"]

        columns = valid_df.columns.tolist()
        fields = [(column, field) if column in ("content", "seq_len", "seq_mask") else (column, label_field) for
                  column in columns]
        serialize(save_path, [field, label_field, train_df, valid_df, fields, columns])

    @staticmethod
    def scan_df(df, n=60):
        print(f"df len: {len(df)}\ncolumns: {df.columns.tolist()}\n")
        for row in islice(df.itertuples(), 0, n):
            print(row.content)
            print("#############################################################################################\n")

    @staticmethod
    def stat_text(df):
        df['seq_len'] = df['content'].apply(lambda text: len(text))
        raw_len = len(df)
        df = df[df["seq_len"] > 1024]
        print(f"cur: {len(df)}, raw: {raw_len}, ratio:{float(len(df)) / float(raw_len)}")
        print(df.iloc[0:1].content)

    @staticmethod
    def df2ds(df, fields, columns):
        examples = [Example.fromlist(record, fields) for record in zip(*[getattr(df, column) for column in columns])]
        return Dataset(examples, fields)


if __name__ == '__main__':
    from transformers import XLNetTokenizer


    class Config(object):
        def __init__(self):
            self.name = "xlnet_multi_attn"
            self.train_file = abspath("data/train.csv")
            self.valid_file = abspath("data/valid.csv")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.seed = 279
            self.num_classes = None
            self.max_seq = 1024
            self.batch_size = 32

            self.xlnet_path = "/Users/Vander/Code/pytorch_col/xlnet-base-chinese"
            self.tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
            self.bos = self.tokenizer.bos_token
            self.sep = self.tokenizer.sep_token


    config_ = Config()
    loader = XlLoader(config_)
