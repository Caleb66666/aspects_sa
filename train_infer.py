# @File: train_infer
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2019/12/27 14:49:22

import torch
import argparse
from importlib import import_module
from utils.time_util import timer, ts_print
from utils.ml_util import count_params
from torch.utils.tensorboard import SummaryWriter
from setproctitle import setproctitle


def test():
    pass


def evaluate(model, valid_wrapper):
    loss, correct = 0.0, 0.0

    model.eval()
    with torch.no_grad():
        for inputs in valid_wrapper:
            output_dict = model(inputs)
            loss += output_dict["loss"]
            batch_len = inputs[-1].size(0)
            correct += output_dict["f1"] * batch_len
    model.train()
    return correct / valid_wrapper.count, loss / len(valid_wrapper)


def batches_eval(model, config, optimizer, scheduler, valid_wrapper, output_dict, assist_params):
    # 决定是否进行evaluate
    if assist_params['batches'] % config.eval_per_batches == 0:
        train_f1, train_loss = output_dict.get("f1"), output_dict.get("loss")
        valid_f1, valid_loss = evaluate(model, valid_wrapper)
        config.scheduler_step(scheduler, valid_loss)
        if valid_loss < assist_params['best_loss']:
            assist_params.update({"best_loss": valid_loss, "last_improve": assist_params['batches']})
            loss_improved = "*"
            config.save_model(model, optimizer, assist_params['cur_epoch'], assist_params['best_loss'])
        else:
            loss_improved = ""
        cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        ts_print(
            f"Iter: {assist_params['batches']:>5}, Train Loss: {train_loss:>5.5f}, Train F1: {train_f1 * 100:6.2f}%, "
            f"Valid Loss: {valid_loss:>5.5f}, Valid F1: {valid_f1 * 100:6.2f}%, Current LR: {cur_lr:.9f} "
            f"{loss_improved}")
        writer = assist_params.get("writer")
        writer.add_scalar("train/loss", train_loss, assist_params["batches"])
        writer.add_scalar("train/f1", train_f1, assist_params["batches"])
        writer.add_scalar("valid/loss", valid_loss, assist_params["batches"])
        writer.add_scalar("valid/f1", valid_f1, assist_params["batches"])

    # 判断是否许久未改善valid_loss
    if assist_params["batches"] - assist_params["last_improve"] > config.improve_require:
        assist_params["stop_flag"] = True
        ts_print(f"train is stopped without improvement in {config.improve_require} batches, best valid loss:"
                 f"{assist_params['best_loss']:>5.4}, best batch: {assist_params['last_improve']:>5}")
    assist_params['batches'] += 1


@timer
def train():
    file_module = import_module(f"models.{args.model.lower()}")
    config = file_module.Config()
    config.seed = args.seed
    loader = config.loader_cls(config)

    # 辅助训练参数
    assist_params = {
        "batches": 0,  # 全局batch数
        "best_loss": float("inf"),  # 最佳valid loss记录
        "last_improve": 0,  # 当前最佳valid loss对应的batches数
        "writer": SummaryWriter(config.summary_dir),
        "stop_flag": False,  # 训练停止
        "cur_epoch": 0
    }

    # 初始化模型
    model = file_module.Model(config)
    optimizer, scheduler = config.build_optimizer(model, len(loader.train_wrapper) * config.epochs)
    ts_print(f"model: {args.model}, model params: {count_params(model)}, train samples: {loader.train_wrapper.count}, "
             f"valid samples: {loader.valid_wrapper.count}, train batches: {len(loader.train_wrapper)}, "
             f"valid batches: {len(loader.valid_wrapper)}, device: {config.device}, batch size: {config.batch_size}")

    # 是否加载旧模型
    if args.restore:
        model, optimizer, epoch, best_loss = config.restore_model(model, optimizer)
        assist_params.update({"cur_epoch": epoch, "best_loss": best_loss})
        ts_print(f"restore model, stopped epoch: {epoch}, best loss: {best_loss:>5.4}")

    model = model.to(config.device)
    model.train()
    for epoch in range(assist_params['cur_epoch'], config.epochs):
        ts_print(f"Epoch [{epoch + 1}/{config.epochs}]")
        assist_params.update({"cur_epoch": epoch})
        for inputs in loader.train_wrapper:
            optimizer.zero_grad()
            output_dict = model(inputs)
            output_dict.get("loss").backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            batches_eval(model, config, optimizer, scheduler, loader.valid_wrapper, output_dict, assist_params)
            if assist_params["stop_flag"]:
                break
        if assist_params["stop_flag"]:
            break
    assist_params["writer"].close()


if __name__ == '__main__':
    task_name = "text_match"
    setproctitle(task_name)
    parser = argparse.ArgumentParser(description=task_name)

    parser.add_argument("--pattern", default="train", type=str, help="train or test")
    parser.add_argument("--model", type=str, required=True, help="choose a model: esim, bert...")
    parser.add_argument("--seed", default=279, type=int, help="random seed")
    parser.add_argument("--restore", action="store_true", default=False, help="restore from ckpt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.pattern == "train":
        train()
    else:
        test()
