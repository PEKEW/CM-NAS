import logging
import sys

import numpy as np
import torch.backends.cudnn as cudnn
import torch.cuda
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable

import args
import utils
from architect import Architect
from model_build import NetWork

from torch.utils.tensorboard import SummaryWriter

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


# def abandon_func_info(func):
# 	def print_info(*args, **kw):
# 		print("警告: 函数{}即将废弃, 或后续的计划中此函数的功能将要被其他函数替代".format(func.__name__))
# 		return func(*args, **kw)
# 	return print_info

def main():
    arg = args.Args()
    tb_summary_writer = SummaryWriter(arg.path)
    assert torch.cuda.is_available(), logging.error('无可用GPU')
    torch.cuda.set_device(arg.gpu_id)
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info("GPU设备 = %d " % arg.gpu_id)
    criterion = nn.CrossEntropyLoss().cuda()
    model = NetWork(arg.init_channel_nums, arg.classes, arg.layers, criterion).cuda()

    logging.info("参数大小: = %fMB", utils.count_parameters(model))

    # 一般优化器 用于优化网络参数
    optimizer = torch.optim.SGD(
        model.parameters(),
        arg.learning_rate,
        momentum=arg.momentum,
        weight_decay=arg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, arg.epochs, eta_min=arg.min_learning_rate
    )

    train_transform, valid_transform = utils.data_transforms_cifar10(arg)
    train_data = dset.CIFAR10(root=arg.data_path, train=True, download=True, transform=train_transform)
    num_of_train_data = len(train_data)
    indices = list(range(num_of_train_data))
    split = int(np.floor(arg.train_portion * num_of_train_data))
    # 训练队列
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=arg.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0
    )
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=arg.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_of_train_data]),
        pin_memory=True, num_workers=0
    )

    # 实例化网络结构
    architect = Architect(model, arg)
    global_steps = 0
    for epoch in range(arg.epochs):
        ## TEST
        model.del_edge()
        print("{} th EPOCH".format(epoch))
        learning_rate = scheduler.get_last_lr()[0]
        model.accumulate_alpha()
        global_steps = train(train_queue, valid_queue, model, architect, criterion, optimizer, learning_rate, arg,
                             global_steps, tb_summary_writer, epoch)
        scheduler.step()
        new_model = None
        if epoch in [0, 5, 10, 15, 20, 25]:
            dct = {}
            for k in range(14):
                dct['beta' + str(k)] = model._betas['normal'][k].item()
            tb_summary_writer.add_scalars('betas', dct, epoch)

        # epoch > 5 -> begin search
        if epoch >= arg.warm_up:
            model.del_edge()
        # 逻辑需要重写  目前这个if会确保下面的代码块一定不会被执行到
        if False and arg.begin_epoch < epoch < arg.end_epoch and (not model.final_tag[0] or not model.final_tag[1]):
            if not model.final_tag[0]:
                need_regenerate_flag = model.decay()
                model.print_alpha_decay_times()
                if need_regenerate_flag and not model.final_tag[0]: new_model = model.new()
            elif not model.final_tag[1]:
                need_regenerate_flag_non_full = model.decay(non_full=True)
                if need_regenerate_flag_non_full: new_model = model.new(non_full=True)
                model.print_alpha_decay_times()
            if new_model is not None:
                utils.del_obj([model, architect, optimizer])
                model = new_model
                architect = Architect(model, arg)
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    scheduler.get_last_lr()[0],
                    momentum=arg.momentum,
                    weight_decay=arg.weight_decay
                )
                utils.del_obj([new_model])
                logging.info("已生成新模型")
    get_acc(valid_queue, model, criterion)


def get_acc(valid_queue, model, criterion):
    """
    @param valid_queue: 验证队列
    @param model: -
    @param criterion: -
    """
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    print("验证集准确率:", valid_acc)


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    for step, (train_input, train_target) in enumerate(valid_queue):
        with torch.no_grad():
            train_input = Variable(train_input).cuda()
            train_target = Variable(train_target).cuda(non_blocking=True)
        logits = model(train_input)
        loss = criterion(logits, train_target)
        prec1, prec5 = utils.accuracy(logits, train_target, topk=(1, 5))
        n = train_input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
    return top1.avg, objs.avg


def train(train_queue, valid_queue, model, architect, criterion, optimizer, learning_rate, arg, global_steps, writer,
          epoch):
    """
    训练过程
    :param writer: -
    :param global_steps: -
    :param train_queue: 训练队列
    :param valid_queue: 验证队列
    :param model: -
    :param architect: 动态结构
    :param criterion: -
    :param optimizer: -
    :param learning_rate: -
    :param arg: -
    :return: 返回top1的平均准确率和平均损失
    """
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (train_input, train_target) in enumerate(train_queue):
        # model.disp_arch_parameters()
        with torch.no_grad():
            train_input = Variable(train_input).cuda()
            train_target = Variable(train_target).cuda(non_blocking=True)
        valid_input, valid_target = next(iter(valid_queue))
        valid_input = Variable(valid_input, requires_grad=False).cuda()
        valid_target = Variable(valid_target, requires_grad=False).cuda(non_blocking=True)
        # model.copy_alphas()
        architect.step(train_input, train_target, valid_input, valid_target, learning_rate, optimizer, arg,
                       unrolled=arg.unrolled)
        # 正常方法更新网络参数
        optimizer.zero_grad()
        logits = model(train_input)
        loss = criterion(logits, train_target)
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), arg.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, train_target, topk=(1, 5))
        n = train_input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        global_steps += 1
        if step % arg.report_frequent == 0:
            logging.info('训练准确率 : %f', top1.avg)
    # print(global_steps)

    return global_steps


if __name__ == '__main__':
    main()
