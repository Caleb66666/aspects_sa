# @File: base_config
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/1/17 16:33:52

import os
import torch
from utils.path_util import abspath, keep_max_backup, newest_file
from utils.time_util import cur_time_stamp


class BaseConfig(object):
    def __init__(self, name):
        self.name = name
        self.model_dir = abspath(f"checkpoints/{self.name}")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_ckpt = os.path.join(self.model_dir, "{}.%s.ckpt" % self.name)
        self.summary_dir = abspath(f"summary/{self.name}")
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

    def build_optimizer(self, model, t_total):
        raise NotImplementedError

    @staticmethod
    def scheduler_step(scheduler, loss):
        scheduler.step()

    def save_model(self, model, optimizer, scheduler, epoch, best_loss, max_backup=3):
        save_dict = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict(),
            "best_loss": best_loss
        }
        torch.save(save_dict, self.model_ckpt.format(cur_time_stamp()))
        keep_max_backup(self.model_dir, max_backup)

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
        if torch.cuda.is_available():
            for state in scheduler.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        model.load_state_dict(save_dict.get("model"))
        return model, optimizer, scheduler, save_dict["epoch"], save_dict["best_loss"]
