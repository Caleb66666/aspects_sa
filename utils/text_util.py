"""
主要是记录文字处理相关的函数：
    繁简转换
    全半角转换
    n-gram生成
    data_frame转化为data_set
"""
from hanziconv import HanziConv


def t2s(text):
    return HanziConv.toSimplified(text)


def full2half(text):
    def char2half(char):
        uchar = ord(char)
        if uchar == 12288:
            return chr(32)
        if uchar in range(65281, 65375):
            return chr(uchar - 65248)
        return char

    return "".join([char2half(char) for char in text])


def rm_blank(text):
    return "".join(text.split())


def remove_duplicate(text, max_dup=3):
    chars, start, end = [], 0, 0
    duplicates = []
    while start < len(text):
        while end < len(text) and text[start] == text[end]:
            duplicates.append(text[start])
            end += 1
        chars.append(''.join(duplicates[:max_dup]))
        duplicates = []
        start = end
    text = ''.join(chars)
    return text


def gen_ngrams(text, n=2):
    tokens = rm_blank(text)
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return list(ngrams)


if __name__ == '__main__':
    test_str = "双十一买什么了"
    print(gen_ngrams(test_str, 3))
