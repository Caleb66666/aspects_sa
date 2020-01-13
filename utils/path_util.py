"""
路径相关的函数，包括：
    当前文件主目录
    文件绝对位置
    序列化、反序列化
"""

import os
import dill


def get_project_root(symbol="utils"):
    """
    获取项目的根目录
    :param symbol: 根目录下的一级目录，或者文件
    :return:
    """
    cur_dir = os.path.dirname(__file__)
    while len(cur_dir) > 1:
        if symbol in os.listdir(cur_dir):
            return cur_dir
        cur_dir = os.path.dirname(cur_dir)
    raise Exception(f"No such symbol: '{symbol}'")


pro_path = get_project_root()


def abspath(relative_file, parent_path=pro_path):
    if parent_path not in relative_file:
        return os.path.join(parent_path, relative_file)
    return relative_file


def serialize(save_path, py_obj):
    with open(save_path, "wb") as f:
        dill.dump(py_obj, f)


def deserialize(load_path):
    with open(load_path, "rb") as f:
        py_obj = dill.load(f)
    return py_obj


def keep_max_backup(target_dir, max_backup):
    files = os.listdir(target_dir)
    if len(files) <= max_backup:
        return
    time_sort_files = []
    for file_ in files:
        abspath_file = os.path.join(target_dir, file_)
        c_time = os.path.getctime(abspath_file)
        time_sort_files.append((abspath_file, c_time))
    time_sort_files = list(sorted(time_sort_files, key=lambda item: item[1], reverse=True))
    [os.remove(file_) for file_, c_time in time_sort_files[max_backup:]]


def newest_file(target_dir):
    time_sort_files = []
    for file_ in os.listdir(target_dir):
        abspath_file = os.path.join(target_dir, file_)
        c_time = os.path.getctime(abspath_file)
        time_sort_files.append((abspath_file, c_time))
    time_sort_files = list(sorted(time_sort_files, key=lambda item: item[1], reverse=True))
    return time_sort_files[0][0]


if __name__ == '__main__':
    test_path = abspath("summary/match")
    print(newest_file(test_path))
