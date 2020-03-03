# @File: vocab
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/3/2 19:52:22

import os
import torch
import numpy as np
import jieba_fast as jieba
from collections import defaultdict, Counter
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from transformers.tokenization_bert import BertTokenizer
from functools import partial
from utils.path_util import serialize, deserialize


class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    UNK = '<unk>'

    def __init__(self, counter, max_size=None, min_freq=1, unk_token="<unk>", pad_token="<pad>"):
        self.specials = (unk_token, pad_token)
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list()  # tokens
        self.unk_index = None
        self.itos = list(self.specials)
        # only extend max size if specials are prepended
        max_size = None if max_size is None else max_size + len(self.specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for token in self.specials:
            del counter[token]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.unk_index = self.specials.index(Vocab.UNK)
        self.pad_index = self.itos.index(pad_token)
        self.stoi = defaultdict(self._default_unk_index)

        # stoi is simply a reverse dict for itos
        self.stoi.update({token: i for i, token in enumerate(self.itos)})

    def _default_unk_index(self):
        return self.unk_index

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __getstate__(self):
        # 序列化时defaultdict会出错，转化为常规dict
        attrs = dict(self.__dict__)
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        if state['unk_index'] is None:
            stoi = defaultdict()
        else:
            stoi = defaultdict(self._default_unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vectors(object):
    def __init__(self, model, unk_token, pad_token):
        self.model = model
        self.unk_token = unk_token
        self.pad_token = pad_token

    @classmethod
    def train_model(cls, w2v_path, sentences, embed_dim, window, min_count, workers, iterations, seed,
                    max_vocab_size, unk_token, pad_token):
        model = Word2Vec(
            sentences=sentences,
            size=embed_dim,
            window=window,
            min_count=min_count,
            max_vocab_size=max_vocab_size,
            seed=seed,
            workers=workers,
            iter=iterations
        )
        model.wv.save_word2vec_format(w2v_path, binary=False)
        return cls(model, unk_token, pad_token)

    @classmethod
    def load_model(cls, w2v_path, unk_token, pad_token):
        model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
        return cls(model, unk_token, pad_token)

    @classmethod
    def init_model(cls, w2v_path, unk_token, pad_token, sentences=None, embed_size=None, window=None, min_count=None,
                   workers=None, iterations=None, seed=None, max_vocab_size=None, force_train=False):
        if not os.path.isfile(w2v_path) or force_train:
            return cls.train_model(w2v_path, sentences, embed_size, window, min_count, workers, iterations, seed,
                                   max_vocab_size, unk_token, pad_token)
        return cls.load_model(w2v_path, unk_token, pad_token)

    def unk_init(self):
        # return np.zeros(shape=(1, self.vectors_dim))
        return torch.normal(mean=0.0, std=1.0 / np.sqrt(self.vectors_dim), size=(1, self.vectors_dim)).numpy()

    @property
    def vectors_dim(self):
        return self.model.vector_size

    @property
    def vectors_num(self):
        return len(self)

    @property
    def shape(self):
        return self.vectors_num, self.vectors_dim

    def __contains__(self, token):
        if isinstance(self.model, Word2Vec):
            return token in self.model.wv
        return token in self.model

    def __getitem__(self, token):
        if token == self.pad_token:
            return np.zeros((1, self.vectors_dim))
        if token not in self or token == self.unk_token:
            return self.unk_init()
        if isinstance(self.model, Word2Vec):
            return self.model.wv[token]
        return self.model[token]

    def __len__(self):
        if isinstance(self.model, Word2Vec):
            return len(self.model.wv.index2word)
        return len(self.model.index2word)


class Tokenizer(object):
    local_bert = "/Users/Vander/Code/pytorch_col/bert-base-chinese/vocab.txt"

    def __init__(self, split_type, user_dict=None, bert_path=None, vocab=None, max_vocab_size=None, min_count=1,
                 pad_token="<pad>", unk_token="<unk>"):
        self.split_type = split_type
        self.user_dict = user_dict
        self.bert_path = bert_path
        self.tokenize_method = self.gen_tokenize_method(split_type, user_dict, bert_path)

        self.vocab = vocab
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.pad_token = pad_token
        self.unk_token = unk_token

    @staticmethod
    def _word_split(sentence):
        return jieba.lcut(sentence)

    @staticmethod
    def _char_split(sentence):
        return list(sentence)

    @staticmethod
    def _piece_split(tokenizer, sentence):
        return tokenizer.tokenize(sentence)

    def gen_tokenize_method(self, split_type, user_dict=None, bert_vocab=None):
        lower_split_type = split_type.lower()

        if lower_split_type == "char":
            return self._char_split

        if lower_split_type == "word":
            jieba.setLogLevel(20)
            if user_dict is not None:
                jieba.load_userdict(user_dict)
            return self._word_split

        if lower_split_type == "word_piece":
            bert_vocab = bert_vocab or self.local_bert
            tokenizer = BertTokenizer.from_pretrained(bert_vocab)
            return partial(self._piece_split, tokenizer)

        raise TypeError(f"error tokenize type: {split_type}")

    def tokenize(self, sentence):
        return self.tokenize_method(sentence)

    def build_vocab(self, *corpus):
        tokens_counter = Counter()
        for tokens_list in corpus:
            [tokens_counter.update(tokens) for tokens in tokens_list]

        self.vocab = Vocab(
            counter=tokens_counter,
            max_size=self.max_vocab_size,
            min_freq=self.min_count,
            unk_token=self.unk_token,
            pad_token=self.pad_token
        )

    def convert_token_to_id(self, token):
        return self.vocab[token]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def contains_token(self, token):
        return token in self.vocab.stoi

    def eq_vocab(self, other):
        return self.vocab == other

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, save_path):
        serialize(save_path, (self.split_type, self.user_dict, self.bert_path, self.vocab, self.max_vocab_size,
                              self.min_count, self.unk_token, self.pad_token))

    @classmethod
    def load(cls, save_path):
        split_type, user_dict, bert_path, vocab, max_vocab_size, min_count, unk_token, pad_token = deserialize(
            save_path)
        return cls(
            split_type=split_type,
            user_dict=user_dict,
            bert_path=bert_path,
            vocab=vocab,
            # 下面四个参数主要作用于build_vocab, load之后基本用不上了，保持初始化的完整性
            max_vocab_size=max_vocab_size,
            min_count=min_count,
            unk_token=unk_token,
            pad_token=pad_token
        )
