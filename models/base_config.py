# @File: new_base_config
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/1/20 14:40:04

import os
import torch
import numpy as np
import random
from utils.path_util import abspath, keep_max_backup, newest_file
from utils.time_util import cur_time_stamp


class BaseConfig(object):
    def __init__(self, name, seed, debug):
        self.name = name

        # 训练设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        # 设置随机种子
        self.seed = seed
        self.set_seed()

        # data-loader，处理完的数据序列化的位置，不需要再次初始化、预处理、分词、word-piece、sentence-piece等操作
        self.dl_path = abspath(f"data/{self.name}.pt")

        # 存储模型位置设置，使用时间戳进行模型的命名
        self.model_dir = abspath(f"checkpoints/{self.name}")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_ckpt = os.path.join(self.model_dir, "{}.%s.ckpt" % self.name)
        self.max_backup = 3  # 最多的模型保存

        # 日志记录设置
        self.summary_dir = abspath(f"summary/{self.name}")
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        # 设置debug模式
        self.debug = debug
        if self.debug:
            self.epochs = 1
            self.batch_size = 2

    # 初始化优化器、调度器
    def build_optimizer_scheduler(self, model, train_batches_len):
        raise NotImplementedError

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)
        # torch.backends.cudnn.deterministic = true

    def save_model(self, model, optimizer, scheduler, epoch, best_loss):
        """
        存储模型，包括模型结构、优化器参数、调度器参数、当前epoch步数
        :param model:
        :param optimizer:
        :param scheduler:
        :param epoch:
        :param best_loss:
        :return:
        """
        save_dict = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict(),
            "best_loss": best_loss
        }
        torch.save(save_dict, self.model_ckpt.format(cur_time_stamp()))
        keep_max_backup(self.model_dir, self.max_backup)

    def restore_model(self, model, optimizer, scheduler, model_ckpt=None):
        if not model_ckpt:
            model_ckpt = newest_file(self.model_dir)
        save_dict = torch.load(model_ckpt)
        optimizer.load_state_dict(save_dict.get("optimizer"))
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        scheduler.load_state_dict(save_dict.get("scheduler"))
        # if torch.cuda.is_available():
        #     for state in scheduler.state.values():
        #         for k, v in state.items():
        #             if torch.is_tensor(v):
        #                 state[k] = v.cuda()
        model.load_state_dict(save_dict.get("model"))
        return model, optimizer, scheduler, save_dict["epoch"], save_dict["best_loss"]
