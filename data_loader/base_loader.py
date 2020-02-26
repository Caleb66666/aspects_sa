# @File: base_loader
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/18 11:10:34

import os
import codecs
import numpy as np
import pandas as pd
from itertools import islice
from utils.text_util import t2s, full2half
from torchtext.data import Field, LabelField, Example, Dataset, BucketIterator
import torch
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_data
from collections import Counter
import jieba_fast as jieba


class BaseTokenizer(object):
    """
    负责分词、建立词典、转换word/index、去停用词
    """

    def __init__(self, max_vocab=None, pad_token="<pad>", unk_token="<unk>", pad_id=0, unk_id=1, tokenize_method=None):
        jieba.setLogLevel(20)
        self.max_vocab = max_vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.word2index = {pad_token: pad_id, unk_token: unk_id}
        self.index2word = {pad_id: pad_token, unk_id: unk_token}

        if tokenize_method is None:
            self.tokenize_method = self.char_tokenize
        else:
            self.tokenize_method = tokenize_method

    @staticmethod
    def jieba_tokenize(sent):
        return jieba.lcut(sent)

    @staticmethod
    def char_tokenize(sent):
        return list(sent)

    def tokenize(self, sent):
        tokens = self.tokenize_method(sent)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        idx_list = list()
        for token in tokens:
            if token in self.word2index:
                idx_list.append(self.word2index[token])
            else:
                idx_list.append(self.unk_id)
        return idx_list

    def build_vocab(self, *args):
        tokens_counter = Counter()
        for tokens_list in args:
            [tokens_counter.update(tokens) for tokens in tokens_list]
        if self.max_vocab is not None:
            vocab = tokens_counter.most_common(self.max_vocab)
        else:
            vocab = tokens_counter.items()
        for idx, (token, freq) in enumerate(vocab):
            self.word2index[token] = idx + 2
            self.index2word[idx + 2] = token


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
    def __init__(self, nb_workers=4):
        pandarallel.initialize(nb_workers=nb_workers, verbose=0)

    @staticmethod
    def df2ds(df, fields, columns):
        examples = [Example.fromlist(record, fields) for record in zip(*[getattr(df, column) for column in columns])]
        return Dataset(examples, fields)

    @staticmethod
    def stat_df(df, seq_column="content", n=10):
        print(f"df len: {len(df)}\ncolumns: {df.columns.tolist()}\n")
        for row in islice(df.itertuples(), 0, n):
            print(getattr(row, seq_column))
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
    def _read_single_raw(single_file, header, sep, debug, shuffle, seed, encoding, train_size):
        if debug:
            df = pd.read_csv(single_file, header=header, sep=sep, nrows=6, encoding=encoding)
            train_df, valid_df = train_test_split(df, train_size=4, shuffle=shuffle, random_state=seed)
        else:
            df = pd.read_csv(single_file, header=header, sep=sep, encoding=encoding)
            train_df, valid_df = train_test_split(df, train_size=train_size, shuffle=shuffle, random_state=seed)
        return train_df, valid_df

    @staticmethod
    def _read_raw(train_file, valid_file, header, sep, debug, shuffle, seed, encoding):
        if debug:
            train_df = pd.read_csv(train_file, header=header, sep=sep, nrows=4, encoding=encoding)
            valid_df = pd.read_csv(valid_file, header=header, sep=sep, nrows=2, encoding=encoding)
        else:
            train_df = pd.read_csv(train_file, header=header, sep=sep, encoding=encoding)
            valid_df = pd.read_csv(valid_file, header=header, sep=sep, encoding=encoding)
        if shuffle:
            train_df = shuffle_data(train_df, random_state=seed)
            valid_df = shuffle_data(valid_df, random_state=seed)
        return train_df, valid_df

    def read_raw(self, files, header, sep, debug, shuffle, seed, encoding="utf-8", train_size=0.7):
        if isinstance(files, str):
            files = (files,)
        if len(files) == 1:
            return self._read_single_raw(files[0], header, sep, debug, shuffle, seed, encoding, train_size)
        if len(files) == 2:
            return self._read_raw(files[0], files[1], header, sep, debug, shuffle, seed, encoding)
        raise RuntimeError(f"files len: {len(files)} should be 1 or 2")

    @staticmethod
    def _pretreatment(text, if_lower=False):
        """
        主要是预处理文本，或者其他去除特殊字符的处理，如果有其他方面的需求，就在子类中将这个方法重载了
        :param text:
        :return:
        """
        text = t2s(text)
        text = full2half(text)
        if if_lower:
            return text.lower()
        return text

    def pretreatment_text(self, train_df, valid_df, premise, debug, if_lower):
        if debug:
            for df in (train_df, valid_df):
                df[premise] = df[premise].apply(lambda text: self._pretreatment(text, if_lower))
        else:
            for df in (train_df, valid_df):
                df[premise] = df[premise].parallel_apply(lambda text: self._pretreatment(text, if_lower))
        return train_df, valid_df

    @staticmethod
    def read_stop_words(sw_file):
        stop_words = set()
        with codecs.open(sw_file, "r", encoding="utf-8") as rfd:
            for line in rfd:
                stop_words.add(line.strip())
        stop_words.update([' ', '\xa0', '\n', '\ufeff', '\r'])
        return stop_words

    @staticmethod
    def _stop_filter(stop_words, tokens):
        return list(filter(lambda token: token not in stop_words, tokens))

    def tokenize_and_index(self, train_df, valid_df, debug, tokenizer=None, max_vocab=None, pad_token=None,
                           unk_token=None, pad_id=None, unk_id=None, stop_dict=None, premise=None,
                           tokens_column="tokens", ids_column="seq_ids"):
        if tokenizer is None:
            tokenizer = BaseTokenizer(
                max_vocab=max_vocab,
                pad_token=pad_token,
                unk_token=unk_token,
                pad_id=pad_id,
                unk_id=unk_id,
            )
            if_custom_tokenizer = True
        else:
            if_custom_tokenizer = False

        if stop_dict is not None:
            stop_words = self.read_stop_words(stop_dict)

        if debug:
            for df in (train_df, valid_df):
                df[tokens_column] = df[premise].apply(tokenizer.tokenize)
                if stop_dict is not None:
                    df[tokens_column] = df[tokens_column].apply(lambda tokens: self._stop_filter(stop_words, tokens))
        else:
            for df in (train_df, valid_df):
                df[tokens_column] = df[premise].parallel_apply(tokenizer.tokenize)
                if stop_dict is not None:
                    df[tokens_column] = df[tokens_column].parallel_apply(
                        lambda tokens: self._stop_filter(stop_words, tokens))

        # 自己定义的需要建立词典，比如迁移模型的直接存储有了，不需要
        if if_custom_tokenizer:
            tokenizer.build_vocab(train_df[tokens_column])

        if debug:
            for df in (train_df, valid_df):
                df[ids_column] = df[tokens_column].apply(tokenizer.convert_tokens_to_ids)
        else:
            for df in (train_df, valid_df):
                df[ids_column] = df[tokens_column].parallel_apply(tokenizer.convert_tokens_to_ids)

        return train_df, valid_df, if_custom_tokenizer, tokenizer

    @staticmethod
    def embed_token(w2v_path, train_df=None, tokens_column="tokens", seed=279, embed_dim=128, max_vocab=20000,
                    pad_token="<pad>", unk_token="<unk>", window=10, min_count=1, workers=4, iterations=15,
                    tokenizer=None,
                    word2index="word2index"):
        # 如果没有已训练好的w2v模型，则重新生成，或者也可以用外部词向量-比如腾讯词向量代替
        if not os.path.exists(w2v_path):
            w2v_model = Word2Vec(train_df[tokens_column], size=embed_dim, window=window, min_count=min_count,
                                 workers=workers, iter=iterations, seed=seed, max_vocab_size=max_vocab)
            w2v_model.wv.save_word2vec_format(w2v_path, binary=False)
        else:
            w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)

        word2index_vocab = getattr(tokenizer, word2index)
        embedding_matrix = np.zeros((len(word2index_vocab), embed_dim))
        for word, idx in word2index_vocab.items():
            if word == pad_token:
                embed_vector = np.zeros((1, embed_dim))
            elif word == unk_token:
                embed_vector = torch.normal(mean=0.0, std=1.0 / np.sqrt(embed_dim), size=(1, embed_dim)).numpy()
            elif word in w2v_model:
                embed_vector = w2v_model[word]
            else:
                embed_vector = torch.normal(mean=0.0, std=1.0 / np.sqrt(embed_dim), size=(1, embed_dim)).numpy()
            embedding_matrix[idx] = embed_vector
        return embedding_matrix

    def _pad_and_truncate(self, seq_ids, max_seq, truncate_method):
        if len(seq_ids) < max_seq:
            seq_len = len(seq_ids)
            seq_mask = [1] * len(seq_ids) + [0] * (max_seq - len(seq_ids))
            seq_ids = seq_ids + [0] * (max_seq - len(seq_ids))
        else:
            seq_len = max_seq
            seq_mask = [1] * max_seq
            seq_ids = self.truncate_single(seq_ids, max_seq, method=truncate_method)
        return seq_ids, seq_len, seq_mask, self.calc_inf_mask(seq_ids)

    def pad_and_truncate(self, max_seq, train_df, valid_df, debug, ids_column="seq_ids", len_column="seq_len",
                         mask_column="seq_mask", inf_mask_column="inf_mask", dropped_columns=None,
                         truncate_method="head"):
        if debug:
            for df in (train_df, valid_df):
                df[ids_column], df[len_column], df[mask_column], df[inf_mask_column] = zip(
                    *df[ids_column].apply(lambda seq_ids: self._pad_and_truncate(seq_ids, max_seq, truncate_method)))
                if dropped_columns is not None:
                    df.drop(columns=dropped_columns, inplace=True)

        else:
            for df in (train_df, valid_df):
                df[ids_column], df[len_column], df[mask_column], df[inf_mask_column] = zip(
                    *df[ids_column].parallel_apply(
                        lambda seq_ids: self._pad_and_truncate(seq_ids, max_seq, truncate_method)))
                if dropped_columns is not None:
                    df.drop(columns=dropped_columns, inplace=True)
        return train_df, valid_df

    @staticmethod
    def prepare_fields(columns, ids_column, len_column, mask_column, inf_mask_column):
        long_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
        float_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        label_field = LabelField(sequential=False, use_vocab=True, batch_first=True)
        fields = list()
        for column in columns:
            if column in (ids_column, len_column, mask_column):
                fields.append((column, long_field))
            elif column in (inf_mask_column,):
                fields.append((column, float_field))
            else:
                fields.append((column, label_field))
        return long_field, float_field, label_field, fields

    def batch_data(self, train_df, valid_df, fields, columns, batch_size, device, sort_within_batch, len_column,
                   build_vocab_field):
        train_ds, valid_ds = self.df2ds(train_df, fields, columns), self.df2ds(valid_df, fields, columns)
        build_vocab_field.build_vocab(train_ds)
        train_iter, valid_iter = BucketIterator.splits(
            (train_ds, valid_ds),
            batch_size=batch_size,
            device=device,
            sort_within_batch=sort_within_batch,
            sort=False,
            sort_key=lambda sample: getattr(sample, len_column),
            repeat=False
        )
        train_batches = BatchWrapper(train_iter, columns, len(train_ds))
        valid_batches = BatchWrapper(valid_iter, columns, len(valid_ds))
        return train_batches, valid_batches
