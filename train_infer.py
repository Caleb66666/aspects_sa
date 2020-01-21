# @File: new_train_infer
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/1/20 14:38:51

import argparse
from setproctitle import setproctitle
from utils.time_util import timer, ts_print
from utils.ml_util import count_params, scale_lr
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter
import torch


def test():
    pass


def evaluate(model, valid_batches):
    loss, f1 = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for inputs in valid_batches:
            outputs = model(inputs)
            loss += outputs["loss"]
            f1 += outputs["f1"]
    return f1 / len(valid_batches), loss / len(valid_batches)


def batches_eval(model, config, optimizer, scheduler, valid_batches, outputs, assist_params):
    if (assist_params['cur_batches'] + 1) % config.eval_per_batches == 0:
        train_f1, train_loss = outputs['f1'], outputs['loss']
        valid_f1, valid_loss = evaluate(model, valid_batches)
        if valid_loss < assist_params["best_loss"]:
            assist_params.update({"best_loss": valid_loss, "last_improve": assist_params['cur_batches']})
            loss_improved = "*"
            config.save_model(model, optimizer, scheduler, assist_params['cur_epoch'], assist_params['best_loss'])
        else:
            loss_improved = ""
        cur_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        ts_print(
            f"Iter: {assist_params['cur_batches'] + 1:>5}/{assist_params['total_batches']}, Train Loss: "
            f"{train_loss:>5.5f}, Train F1: {train_f1 * 100:6.2f}%, Valid Loss: {valid_loss:>5.5f}, Valid F1: "
            f"{valid_f1 * 100:6.2f}%, Current LR: {cur_lr:.9f} {loss_improved}")
        writer = assist_params.get("writer")
        writer.add_scalar("train/loss", train_loss, assist_params["cur_batches"])
        writer.add_scalar("train/f1", train_f1, assist_params["cur_batches"])
        writer.add_scalar("valid/loss", valid_loss, assist_params["cur_batches"])
        writer.add_scalar("valid/f1", valid_f1, assist_params["cur_batches"])

    if (assist_params['cur_batches'] + 1) % config.schedule_per_batches == 0:
        scheduler.step()

    # 判断是否许久未改善valid_loss
    if assist_params["cur_batches"] - assist_params["last_improve"] > config.improve_require:
        assist_params["stop_flag"] = True
        ts_print(f"train is stopped without improvement in {config.improve_require} batches, best valid loss:"
                 f"{assist_params['best_loss']:>5.4}, best batch: {assist_params['last_improve']:>5}")


@timer
def train():
    model_module = import_module(f"models.{args.model.lower()}")

    # config总管所有参数、dl总管原始数据的处理以及分成batches等、model代表建立的模型，optimizer优化器、scheduler学习率调度器
    config = model_module.Config(args.seed)
    dl = config.loader_cls(config)
    model = model_module.Model(config)
    optimizer, scheduler = config.build_optimizer_scheduler(model, len(dl.train_batches))

    # 辅助训练参数
    assist_params = {
        "cur_batches": 0,  # 全局batch数
        "total_batches": config.epochs * len(dl.train_batches),
        "best_loss": float("inf"),  # 最佳valid loss记录
        "last_improve": 0,  # 当前最佳valid loss对应的batches数
        "writer": SummaryWriter(config.summary_dir),
        "stop_flag": False,  # 训练停止
        "cur_epoch": 0,
    }

    ts_print(f"model: {args.model}, model params: {count_params(model)}, train samples: {dl.train_batches.count}, "
             f"valid samples: {dl.valid_batches.count}, train batches: {len(dl.train_batches)}, "
             f"valid batches: {len(dl.valid_batches)}, device: {config.device}, batch size: {config.batch_size}")

    # 是否加载旧模型
    if args.restore:
        model, optimizer, scheduler, epoch, best_loss = config.restore_model(model, optimizer, scheduler)
        # 有时候为了续接训练，需要适当的对原优化器lr进行调整
        scale_lr(optimizer, scale=5)
        assist_params.update({"cur_epoch": epoch, "best_loss": best_loss, "cur_batches": len(dl.train_batches) * epoch})
        ts_print(f"restore model, stopped epoch: {epoch}, best loss: {best_loss:>5.4}")

    # 开始训练
    model = model.to(config.device)
    model.zero_grad()
    for epoch in range(assist_params['cur_epoch'], config.epochs):
        ts_print(f"Epoch [{epoch + 1}/{config.epochs}]")
        assist_params.update({"cur_epoch": epoch})
        for inputs in dl.train_batches:
            model.train()
            outputs = model(inputs)
            outputs['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            if config.debug:
                break
            batches_eval(model, config, optimizer, scheduler, dl.valid_batches, outputs, assist_params)
            assist_params['cur_batches'] += 1
            if assist_params["stop_flag"]:
                break
        if assist_params["stop_flag"]:
            break
    assist_params["writer"].close()


if __name__ == '__main__':
    task_name = "aspects_sa"
    setproctitle(task_name)
    parser = argparse.ArgumentParser(description=task_name)

    parser.add_argument("--pattern", default="train", type=str, help="train or test")
    parser.add_argument("--model", type=str, required=True, help="choose a model: albert_attn_pool/xlnet_attn_pool")
    parser.add_argument("--seed", default=279, type=int, help="random seed")
    parser.add_argument("--restore", action="store_true", default=False, help="restore from ckpt")
    parser.add_argument("--debug", action="store_true", default=False, help="debug model for one batch")
    args = parser.parse_args()

    if args.pattern == "train":
        train()
    else:
        test()
