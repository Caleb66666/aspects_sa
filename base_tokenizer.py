# @File: base_tokenizer
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/14 17:46:03

import codecs
from collections import Counter

import jieba_fast as jieba

jieba.setLogLevel(20)

"""
读取、预处理、分词(停用)、词嵌入(训练、加载)、torch化&分批
"""


class BaseTokenizer(object):
    """
    负责分词、建立词典、转换word/index、去停用词
    """

    def __init__(self, max_vocab=None, pad_token="<pad>", unk_token="<unk>", pad_id=0, unk_id=1,
                 stop_words_file=None):
        self.max_vocab = max_vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.word2index = {pad_token: pad_id, unk_token: unk_id}
        self.index2word = {pad_id: pad_token, unk_id: unk_token}
        if stop_words_file is None:
            self.stop_words = None
        else:
            self.stop_words = self.read_stop_words(stop_words_file)

    @staticmethod
    def read_stop_words(sw_file):
        stop_words = set()
        with codecs.open(sw_file, "r", encoding="utf-8") as rfd:
            for line in rfd:
                stop_words.add(line.strip())
        stop_words.update([' ', '\xa0', '\n', '\ufeff', '\r'])
        return stop_words

    @staticmethod
    def jieba_tokenize(sent):
        return jieba.lcut(sent)

    @staticmethod
    def char_tokenize(sent):
        return list(sent)

    def tokenize(self, sent):
        tokens = self.char_tokenize(sent)
        if self.stop_words is None:
            return tokens
        return list(filter(lambda token: token not in self.stop_words, tokens))

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
