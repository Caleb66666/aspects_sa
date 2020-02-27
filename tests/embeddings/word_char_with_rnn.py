# @File: embeddings
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/27 17:12:39

import unittest
from custom_modules.embeddings import WordCharEmbeddingWithRnn
import jieba
import numpy as np
from collections import Counter
import torch


class TestWordCharEmbeddingWithRnnClass(unittest.TestCase):
    def setUp(self):
        super(TestWordCharEmbeddingWithRnnClass, self).setUp()

        self.char_limit = 4
        self.max_seq = 25
        self.batch_size = 3
        self.word_embed_size = 17
        self.char_embed_size = 13
        self.rnn_hidden_size = 19
        self.fusion_method = "cat"
        self.char_bidirectional = True
        self.positional_encoding = True

        self.words_idx, self.chars_idx, self.words_vocab_size, self.chars_vocab_size = self.gen_test_env()
        self.embedding = WordCharEmbeddingWithRnn(
            word_vocab_size=self.words_vocab_size,
            word_embed_size=self.word_embed_size,
            char_vocab_size=self.chars_vocab_size,
            char_embed_size=self.char_embed_size,
            rnn_hidden_size=self.rnn_hidden_size,
            bidirectional=self.char_bidirectional,
            positional_encoding=self.positional_encoding,
            fusion_method=self.fusion_method,
            max_seq=self.max_seq
        )

    def gen_test_env(self):
        contents = ['这是一个很好的餐馆，菜很不好吃，我还想再去',
                    '这是一个很差的餐馆，菜很不好吃，我不想再去',
                    '这是一个很好的餐馆，菜很好吃，我还想再去',
                    '这是一个很好的餐馆，只是菜很难吃，我还想再去',
                    '这是一个很好的餐馆，只是菜很不好吃，我还想再去',
                    '好吃的！黑胡椒牛肉粒真的是挺咸的',
                    '不论是环境的宽敞度还是菜的味道上',
                    '烤鸭皮酥脆入口即化，总体还可以',
                    '烤鸭皮酥脆入口即化',
                    '软炸鲜菇据说是他家的优秀美味担当',
                    '环境挺好的，服务很到位',
                    '肉松的味道都不错，量一般',
                    '也不算便宜，不过吃着好吃',
                    '高大上的餐厅，一级棒的环境',
                    '比较硬，比较喜欢芝士和咖喱口味的',
                    '有嚼劲，很入味宫廷豌豆黄造型可爱',
                    '蔬菜中规中矩，解腻不错',
                    '女友生日菜有点贵架势不错味道就这样',
                    '相比其他兰州拉面粗旷的装饰风格，这家设计很小清新，座位宽敞，客人不多']

        char_counter = Counter()
        word_counter = Counter()
        for content in contents:
            chars_ = list(content)
            words_ = jieba.lcut(content)
            char_counter.update(chars_)
            word_counter.update(words_)

        words_vocab = dict()
        chars_vocab = dict()
        for idx, (word, freq) in enumerate(word_counter.items()):
            words_vocab.update({word: idx})
        for idx, (char, freq) in enumerate(char_counter.items()):
            chars_vocab.update({char: idx})

        words_idx = list()
        chars_idx = list()
        for idx in range(self.batch_size):
            words = jieba.lcut(contents[idx])
            chars = [list(word) for word in words]
            word_ids = [words_vocab.get(word) for word in words]
            if len(word_ids) < self.max_seq:
                word_ids.extend([0] * (self.max_seq - len(word_ids)))
            char_ids = np.zeros([len(word_ids), self.char_limit], dtype=np.int32)
            for i, tokens in enumerate(chars):
                for j, char in enumerate(tokens):
                    if j >= self.char_limit:
                        break
                    char_ids[i, j] = chars_vocab.get(char)

            word_ids = torch.tensor(word_ids).long().view(1, -1)
            char_ids = torch.from_numpy(char_ids).long().view(1, -1)

            words_idx.append(word_ids)
            chars_idx.append(char_ids)

        return torch.cat(words_idx, dim=0), torch.cat(chars_idx, dim=0), len(words_vocab), len(chars_vocab)

    def test_embed_output_size(self):
        calc_size = self.embedding(self.words_idx, self.chars_idx).size()
        if self.fusion_method == "sfu":
            predict_size = torch.Size([self.batch_size, self.max_seq, self.word_embed_size])
        else:
            if self.char_bidirectional:
                hidden_size = 2 * self.rnn_hidden_size
            else:
                hidden_size = self.rnn_hidden_size
            predict_size = torch.Size([self.batch_size, self.max_seq, self.word_embed_size + hidden_size])

        self.assertEqual(calc_size, predict_size)


if __name__ == '__main__':
    unittest.main()
