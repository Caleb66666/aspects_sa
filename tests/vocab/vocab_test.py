# @File: vocab_test
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/3/2 20:53:17

import unittest
from utils.vocab_util import Vocab
from collections import Counter


class TestVocab(unittest.TestCase):
    def setUp(self):
        super(TestVocab, self).setUp()

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
        for content in contents:
            char_counter.update(list(content))

        self.max_vocab_size = 50
        min_count = 1
        self.vocab = Vocab(
            counter=char_counter,
            max_size=self.max_vocab_size,
            min_freq=min_count
        )

    def test_vocab_size(self):
        self.assertEqual(self.max_vocab_size + len(self.vocab.specials), len(self.vocab))
