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
        if not os.path.exists(config.dl_path):
            label_field, train_df, valid_df, fields, columns = self.init_raw(config)
        else:
            label_field, train_df, valid_df, fields, columns = deserialize(config.dl_path)

        train_ds, valid_ds = self.df2ds(train_df, fields, columns), self.df2ds(valid_df, fields, columns)
        label_field.build_vocab(valid_ds)

        train_iter, valid_iter = BucketIterator.splits(
            (train_ds, valid_ds),
            batch_size=config.batch_size,
            device=config.device,
            sort=False,
            sort_within_batch=False,
            sort_key=lambda sample: getattr(sample, "seq_len"),
            repeat=False
        )
        self.train_batches = BatchWrapper(train_iter, columns, len(train_ds))
        self.valid_batches = BatchWrapper(valid_iter, columns, len(valid_ds))
        self.classes = label_field.vocab
        config.classes = label_field.vocab
        config.num_classes = len(self.classes)
        config.num_labels = len(columns) - 3

    @staticmethod
    def df2ds(df, fields, columns):
        examples = [Example.fromlist(record, fields) for record in zip(*[getattr(df, column) for column in columns])]
        return Dataset(examples, fields)

    @staticmethod
    def zero_inf_mask(ids):
        if torch.is_tensor(ids):
            mask = ids.eq(0)
            float_mask = mask.float()
            return torch.masked_fill(float_mask, mask, float("-inf"))
        return [float("-inf") if ele == 0 else 0.0 for ele in ids]

    def xl_process(self, config, text):
        text = t2s(text)
        text = full2half(text)
        text = text.strip('"').strip()
        text = text.replace("\n", "").replace("\r", "")

        tokens = config.tokenizer.tokenize(text)
        token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < config.max_seq:
            seq_mask = [1] * len(token_ids) + [0] * (config.max_seq - len(token_ids))
            token_ids += ([0] * (config.max_seq - len(token_ids)))
        else:
            seq_mask = [1] * config.max_seq
            token_ids = token_ids[:config.first_half] + token_ids[config.latter_half:]
        return token_ids, self.zero_inf_mask(token_ids), seq_mask

    def init_raw(self, config):
        train_df = pd.read_csv(config.train_file, header=0, sep=",")
        valid_df = pd.read_csv(config.valid_file, header=0, sep=",")
        # train_df, valid_df = train_df[:4], valid_df[:2]
        for df in (train_df, valid_df):
            df["content"], df["inf_mask"], df["seq_mask"] = zip(
                *df["content"].apply(lambda text: self.xl_process(config, text)))
            df.drop(columns=["id"], inplace=True)

        long_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
        float_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        label_field = LabelField(sequential=False, use_vocab=True, batch_first=True)
        fields = list()

        columns = train_df.columns.tolist()
        for column in columns:
            if column in ("content", "seq_mask"):
                fields.append((column, long_field))
            elif column in ("inf_mask",):
                fields.append((column, float_field))
            else:
                fields.append((column, label_field))
        serialize(config.dl_path, [label_field, train_df, valid_df, fields, columns])
        return label_field, train_df, valid_df, fields, columns


class OldXlLoader(object):
    def __init__(self, config):
        save_path = abspath(f"data/{config.name}.dl.pt")
        if not os.path.exists(save_path):
            self.init_raw(config, save_path)
        label_field, train_df, valid_df, fields, columns = deserialize(save_path)
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
        self.train_batches = BatchWrapper(train_iter, columns, len(train_ds))
        self.valid_batches = BatchWrapper(valid_iter, columns, len(valid_ds))
        self.classes = label_field.vocab
        config.num_classes = len(self.classes)
        config.num_labels = len(columns) - 3
        config.classes = label_field.vocab

    @staticmethod
    def zero_inf_mask(ids):
        return [float("-inf") if ele == 0 else 0.0 for ele in ids]

    def xl_process(self, config, text):
        text = t2s(text)
        text = full2half(text)
        text = text.strip('"').strip()
        text = text.replace("\n", "").replace("\r", "")

        tokens = list()
        tokens.extend(config.tokenizer.tokenize(text))
        token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < config.max_seq:
            mask = [1] * len(token_ids) + [0] * (config.max_seq - len(token_ids))
            token_ids += ([0] * (config.max_seq - len(token_ids)))
        else:
            mask = [1] * config.max_seq
            token_ids = token_ids[:config.max_seq]
        return token_ids, self.zero_inf_mask(token_ids), mask

    def init_raw(self, config, save_path):
        field = Field(sequential=False, use_vocab=False, batch_first=True)
        inf_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        label_field = LabelField(sequential=False, use_vocab=True, batch_first=True)

        train_df = pd.read_csv(config.train_file, header=0, sep=",")
        valid_df = pd.read_csv(config.valid_file, header=0, sep=",")
        train_df["content"], train_df["inf_mask"], train_df["seq_mask"] = zip(*train_df["content"].apply(
            lambda text: self.xl_process(config, text)))
        valid_df["content"], valid_df["inf_mask"], valid_df["seq_mask"] = zip(*valid_df["content"].apply(
            lambda text: self.xl_process(config, text)))
        del train_df["id"], valid_df["id"]

        columns = train_df.columns.tolist()
        fields = list()
        for column in columns:
            if column == "inf_mask":
                fields.append((column, inf_field))
            elif column in ("content", "seq_mask"):
                fields.append((column, field))
            else:
                fields.append((column, label_field))

        serialize(save_path, [label_field, train_df, valid_df, fields, columns])

    @staticmethod
    def scan_df(df, n=6):
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
            self.num_labels = None
            self.classes = None
            self.max_seq = 1024
            self.batch_size = 32

            self.xlnet_path = "/Users/Vander/Code/pytorch_col/xlnet-base-chinese"
            self.tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
            self.bos = self.tokenizer.bos_token
            self.sep = self.tokenizer.sep_token


    config_ = Config()
    loader = OldXlLoader(config_)
