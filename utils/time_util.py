"""
主要涉及日期相关的函数，包括：
    日期格式转换
    计时器
"""

from functools import wraps
import time
from datetime import datetime


def str2date(time_str, time_format="%Y-%m-%d"):
    """
    根据不同的时间格式进行日期类型的转换
    :param time_str:
    :param time_format: 其他格式如："%a %b %d %Y %H:%M:%S GMT+0800 (China Standard Time)"
    :return:
    """
    return datetime.strptime(time_str, time_format)


def cur_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def cur_time_stamp(ms=False):
    """
    是否使用毫秒级
    :param ms:
    :return:
    """
    if ms:
        return int(round(time.time() * 1000))
    return int(time.time())


def ts_print(*args):
    # print("[%s] %s" % (cur_time_str(), " ".join([json.dumps(arg, ensure_ascii=False) for arg in args])))
    print("[%s] %s" % (cur_time_str(), " ".join([str(arg) for arg in args])))


def timer(func):
    @wraps(func)
    def with_timer(*args, **kwargs):
        tic = time.time()
        res = func(*args, **kwargs)
        toc = time.time()
        ts_print("%s costs: %.2f(s)" % (func.__name__, (toc - tic)))
        return res

    return with_timer


if __name__ == '__main__':
    pass
