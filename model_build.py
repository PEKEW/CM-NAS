import collections
import logging
import math
import re
import sys
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import args
from genotypes import PRIMITIVES
from opertaions import *
from utils import del_tensor_element, arg_sort

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
arg = args.Args()


# TODO 规范注释


def logic_rewrite(func):
    def printInfo(*args, **kw):
        print("函数{}的逻辑可能即将或已经不符合当前代码的整体逻辑结构, 需要重写".format(func.__name__))
        return func(*args, **kw)

    return printInfo


class MixedOP(nn.Module):
    """
    这个类作为计算图的出口，有两个作用：
        - 把alpha，mask和weight结合起来
        - 融合数据流之间的操作为MixedOP
    """

    def __init__(self, current_channel, stride_, primitive_dict=PRIMITIVES):
        """
        @param current_channel: 每个node的输出深度
        @param stride_: 卷积步长
        """
        super().__init__()
        self._ops = nn.ModuleList()
        # 从PRIMITIVES中遍历所有操作
        for primitive in primitive_dict:
            # OPS是匿名函数字典 以primitive为key
            op = OPS[primitive](current_channel, stride_, False)
            if 'pool' in primitive:
                # 把池化和BN绑定在一起
                op = nn.Sequential(op, nn.BatchNorm2d(current_channel, affine=False))
            self._ops.append(op)

    # 混合计算过程为所有操作加权求和
    def forward(self, x, weights, mask):
        if len(weights.shape) == 0:
            return weights * mask * self._ops[0](x)
        else:
            return sum(w * dw * op(x) for w, dw, op in zip(weights, mask, self._ops))


class Cell(nn.Module):
    """
    cell作为搭建网络的单位
    """

    def __init__(self, steps, multiplier, channels_pre_2, channels_pre_1, current_channels, reduction, is_pre_reduction,
                 primitive_dict_list, per_order_info):
        """
        @:param steps:               细胞中间不确定操作的节点个数
        @:param multiplier:          深度乘子
        @:param channels_pre_2:      第一个输入的深度
        @:param channels_pre_1:      第二个深入的深度
        @:param current_channels:    输出深度
        @:param reduction:           当前细胞的缩小指针
        @:param is_pre_reduction:    上一个细胞的缩小指针
        """
        super().__init__()
        self.reduction = reduction
        # 如果接入当前细胞的一个输入是缩小细胞 就对特征映射做缩小处理
        self._preprocess_0 = FactorizedReduce(channels_pre_2, current_channels, affine=False) \
            if is_pre_reduction else \
            ReLUConvBN(channels_pre_2, current_channels, 1, 1, 0, affine=False)
        # 第二个输入不受缩小状态的影响
        self._preprocess_1 = ReLUConvBN(channels_pre_1, current_channels, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        # 处理所有操作不确定的节点
        count = 0
        for i in range(self._steps):
            if not per_order_info:
                self.full = True
                for j in range(2 + i):
                    stride = 2 if reduction and j < 2 else 1
                    """
                        i表示不确定的节点编号 [0,1,2,3] 是针对不确定的4个节点的编号
                        j表示i的前驱节点编号 [0,1,2,..,5] 是针对cell中所有的节点而言的编号 
                        每个节点都和拓扑排序在自己之前的所有节点相连接
                        i = 0 => j = 0 , 1
                        i = 1 => j = 0 , 1 , 2
                    """
                    # 如果j是输入节点{0,1} 且当前细胞为缩小细胞 就放大步长以缩小输出映射
                    op = MixedOP(current_channels, stride, primitive_dict_list[count])
                    count += 1
                    self._ops.append(op)
            else:
                self.full = False
                # order_dct没有保存第一个node的输入节点 需要手动工造先
                # TODO normal和reduce应该抽象成一个函数
                _order = [[0, 1]]
                point = 0
                order_dct = {}
                for each in per_order_info['normal']:
                    _order.append([k for k in range(point) if k not in each])
                    point += 1
                order_dct['normal'] = _order
                _order = [[0, 1]]
                point = 0
                for each in per_order_info['reduce']:
                    _order.append([k for k in range(point) if k not in each])
                    point += 1
                order_dct['reduce'] = _order
                self.order_dct = order_dct
                """
                    per_order_info保存的是操作的前驱节点
                    最开始(强连通)的前驱节点列表应该是
                    [
                        [0,1],
                        [0,1,2],
                        [0,1,2,3],
                    ...]
                    经过删除一些操作后, 前驱节点可能变为
                    [
                        [0,1],
                        [0,2],
                        [0,2,3],
                    ...]
                """
                pre_idx_list = per_order_info['reduce'] if reduction else per_order_info['normal']
                for j in pre_idx_list[i]:
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOP(current_channels, stride, primitive_dict_list[count])
                    count += 1
                    self._ops.append(op)

    def forward(self, s0, s1, alpha_weights, beta_weights, mask):
        # 先处理细胞的两个输入节点
        s0, s1 = self._preprocess_0(s0), self._preprocess_1(s1)
        states = [s0, s1]
        if self.full:
            for i in range(self._steps):
                offset = 0
                """
                关于此生成式:
                    s0,s1分别是cell的输入
                    它们经过ops[0],_ops[1]得到node之间的中间状态 h1并添加到states
                    s0,s1,h1再过op[2],op[3]...等到h2
                    这样做是因为在ops.append时没有任何额外索引
                    所有的节点都和它拓扑排序之间的节点按顺序连接
                    权重也是按照这样的顺序解包
            """
                s = sum(
                    mask['edge'][offset + j] * beta_weights[offset + j] * self._ops[offset + j](h, alpha_weights[
                        offset + j], mask['op'][offset + j]) for
                    j, h in
                    enumerate(states))
                offset += len(states)
                states.append(s)
        else:
            # 当不是全连通时 normal和reduce的连接状态会因为删除操作导致不一致 要在此处判断normal/reduce
            order_list = self.order_dct['reduce'] if self.reduction else self.order_dct['normal']
            op_id = 0
            for each_list in order_list:
                """
                each_list 是第op_id个操作对应的前驱结点列表
                因为不是全连通 所以必须保存前驱节点的信息才能构造正确的数据流
                因而不需要使用拓扑排序的方法构造
                """
                s = 0
                for idx in each_list:
                    s += beta_weights[op_id] * self._ops[op_id](states[idx], alpha_weights[idx], mask[idx])
                    op_id += 1
                states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


class NetWork(nn.Module):
    def __init__(self, init_channels, classes, layers, criterion, per_order_info=None):
        """
        @:param init_channels:    卷积核个数
        @:param classes:          分类器个数
        @:param layers:           网络细胞个数
        @:param criterion:        优化器
        """
        super().__init__()
        """self
            _channels        ->  网络输入深度
            _classes_nums     ->  分类个数
            _layers          ->  cell的层数
            _steps           ->  不确定操作的节点个数
            _mutiplier       ->  网络深度乘子
            _stem_mutiplier  ->  主干网络乘子
        """
        self.is_attention = True
        self._channels = init_channels
        self._classes_nums = classes
        self._layers = layers
        self.criterion = criterion
        self._steps = 4
        self._mutiplier = 4
        self._stem_mutiplier = 3
        self.final_tag = [False, False]
        self._init_primitive_dict()
        self._init_alphas()
        self._init_betas()
        self._set_arch_parameters()
        # self._init_eva_times()
        self._init_masks()
        # self._init_accumulate_alpha()
        # self._init_alpha_copy_dct()
        # current_channels ->  当前(输出)深度
        current_channels = self._stem_mutiplier * init_channels
        # 构造主干网络
        self._stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=current_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(current_channels)
        )
        # 调整cell输出深度指示器
        """
            channels_pre_2   ->  前前层的深度
            channels_pre_1   ->  前层深度
        """
        channels_pre_2, channels_pre_1, current_channels = current_channels, current_channels, init_channels

        # cells是一个model_list 用来存储cell
        self._cells = nn.ModuleList()
        """
            is_pre_reduction  ->    缩小指针 用来表示前一层cell是不是reduction cell
            reduction         ->    缩小指针 用来表示当前cell是不是reduction cell
        """
        is_pre_reduction = False
        for i in range(layers):
            # 整个网络的第1/3，2/3处设置缩小细胞
            if i in [layers // 3, 2 * layers // 3]:
                current_channels *= 2
                reduction = True
                primitive_dict = self._operation_dict['reduce']
            else:
                reduction = False
                primitive_dict = self._operation_dict['normal']
            cell = Cell(self._steps, self._mutiplier, channels_pre_2, channels_pre_1, current_channels, reduction,
                        is_pre_reduction, primitive_dict, per_order_info)
            is_pre_reduction = reduction
            self._cells += [cell]
            channels_pre_2, channels_pre_1 = channels_pre_1, self._mutiplier * current_channels

        self._global_pooling = nn.AdaptiveAvgPool2d(1)
        self._classifier = nn.Linear(channels_pre_1, classes)

    def _init_alpha_copy_dct(self):
        _dct = {'normal': self._alphas['normal'].detach().clone(),
                'reduce': self._alphas['reduce'].detach().clone()}
        setattr(self, '_alphas_copy', _dct)

    @logic_rewrite
    def _init_accumulate_alpha(self):
        setattr(self, '_accumulate', {
            'normal': torch.zeros(self._alphas['normal'].shape, requires_grad=False).cuda(),
            'reduce': torch.zeros(self._alphas['reduce'].shape, requires_grad=False).cuda()
        })

    def _init_primitive_dict(self):
        def generate_dict():
            primitive_dict = []
            [primitive_dict.append(deepcopy(PRIMITIVES)) for _ in range(k)]
            return [primitive_dict, primitive_dict]

        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        normal, reduce = generate_dict()
        dict_ = {
            'normal': normal,
            'reduce': reduce
        }
        setattr(self, '_operation_dict', dict_)

    def _init_alphas(self):
        """
         初始化操作权重
         """
        # 每个node都和前面node相连接 依次有2，3，4，5个连接
        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        num_ops = len(PRIMITIVES)
        _alphas = {'normal': Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True),
                   'reduce': Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)}
        setattr(self, '_alphas', _alphas)

    def _init_masks(self):
        """
        初始化掩码
        掩码是操作掩码和边掩码组成的二元组 self._mask = (op_mask,edge_mask)
        其中每个掩码都是一个字典 分别和alpha/edge形状一致
        """
        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        num_ops = len(PRIMITIVES)
        op_masks = {'normal': torch.ones(k, num_ops, requires_grad=False).cuda(),
                    'reduce': torch.ones(k, num_ops, requires_grad=False).cuda()}
        edge_masks = {'normal': torch.ones(k, requires_grad=False).cuda(),
                      'reduce': torch.ones(k, requires_grad=False).cuda()}
        setattr(self, '_masks', {'op_masks': op_masks,
                                 'edge_masks': edge_masks}
                )

    def _init_betas(self):
        """
        初始化边注意力
        is_attention 决定使用哪种初始化方式
            若使用注意力则 使用randn初始化(与alphas一致)
            否则使用ones
        """
        # 边注意力只和结点尺度上的连接有关系
        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        if self.is_attention:
            _betas = {'normal': Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True),
                      'reduce': Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)}
        else:
            _betas = {'normal': Variable(torch.ones(k).cuda(), requires_grad=True),
                      'reduce': Variable(torch.ones(k).cuda(), requires_grad=True)}
        self._betas = _betas

    def _succ_betas(self, dct):
        self._betas = dct

    def _set_arch_parameters(self):
        """
        把alpha和beta放到_arch_parameters
        """
        _arch_parameters = [self._alphas['normal'], self._alphas['reduce'],
                            self._betas['normal'], self._betas['reduce']
                            ]
        setattr(self, '_arch_parameters', _arch_parameters)

    def forward(self, input_tensor):
        s0 = s1 = self._stem(input_tensor)
        for i, cell in enumerate(self._cells):
            _type = 'normal' if cell.reduction else 'reduce'
            alpha_weights = F.softmax(self._alphas[_type], dim=-1)
            n = 3
            start = 2
            if self.is_attention:
                beta_weights = F.softmax(self._betas[_type][0:2], dim=-1)
            else:
                beta_weights = F.relu(self._betas[_type][0:2])
            for _ in range(self._steps - 1):
                end = start + n
                if self.is_attention:
                    w = F.softmax(self._betas[_type][start:end], dim=-1)
                else:
                    w = F.relu(self._betas[_type][start:end])
                start = end
                n += 1
                beta_weights = torch.cat([beta_weights, w], dim=0)

            mask = getattr(self, '_masks')
            op_mask = mask['op_masks']
            edge_mask = mask['edge_masks']
            s0, s1 = s1, cell(s0, s1, alpha_weights, beta_weights, {'op': op_mask[_type],
                                                                    'edge': edge_mask[_type]})
        output = self._global_pooling(s1)
        logits = self._classifier(output.view(output.size(0), -1))
        return logits

    def loss(self, input_tensor, target):
        return self.criterion(self(input_tensor), target)

    def del_edge(self):
        """
         搜索开始阶段 直接删除掉一些多余的边 以迅速减少参数量
        """
        # 生成4个node的前驱节点范围的记录
        #     # [0,1], [0,2]...
        # 每个node只保留两个操作
        per_node_range_idx = []
        begin = 2
        for i in range(2, self._steps + 1):
            per_node_range_idx.append([begin, begin + i + 1])
            begin += i + 1
        res = 0
        normal_worst_edge_idx = []
        reduce_worst_edge_idx = []
        for per_nodes in per_node_range_idx:
            normal_worst_edge_idx.append(torch.argsort(
                (self._betas['normal'][per_nodes[0]:per_nodes[1]])[0:per_nodes[1] - per_nodes[0] - 2]
            )
            )
            reduce_worst_edge_idx.append(torch.argsort(
                (self._betas['reduce'][per_nodes[0]:per_nodes[1]])[0:per_nodes[1] - per_nodes[0] - 2]
            )
            )
        self.new(normal_worst_edge_idx, reduce_worst_edge_idx)

    @logic_rewrite
    def _init_eva_times(self):
        """
        初始化演化次数
        因为后续有除法操作 为了避免除以0 在初始化时每个操作都假设被演化过一次
        这样做并不会影响大小判断
        """
        _eva_times = {'normal': Variable(torch.ones(self._alphas['normal'].shape).cuda(), requires_grad=False),
                      'reduce': Variable(torch.ones(self._alphas['reduce'].shape).cuda(), requires_grad=False)}
        setattr(self, '_eva_times', _eva_times)

    @logic_rewrite
    # def _regenerate_operation_dict(self, normal, reduce):
    #     """
    #     重新生成参数字典
    #     @:return [normal, reduce]
    #         normal -> [需要丢弃列表, 操作字典]
    #     """
    #
    #     def gen_op_dct(type_string):
    #         obj = self._betas[type_string]
    #         dropped_list, dropped_list_no_reg = [], []
    #         operation_dict_copied = deepcopy(self._operation_dict)
    #         reg_num = [range(15)]
    #         """
    #             如果要从原始的操作字典中删除, 无论是在循环中使用remove还是pop都是不合适的:
    #                 - remove只会从前往后删除
    #                 - pop删除会打乱索引
    #             所以在循环中给需要被删除的位置打上标记, 随后再用统一删除
    #         """
    #         count = 0
    #         for worst_idx in normal:
    #             temp = obj[begin_idx: begin_idx + worst_idx]
    #             min_idx_no_reg = torch.argmin(temp).item()
    #             min_idx = min_idx_no_reg + reg_num
    #             dropped_list_no_reg.append(min_idx_no_reg)
    #             begin_idx += worst_idx
    #             reg_num = begin_idx
    #             dropped_list.append(min_idx)
    #             operation_dict_copied[type_string][min_idx] = 'to_be_removed'
    #             count += 1
    #         self.__gen_set_pre_idx_list(dropped_list_no_reg, type_string)
    #         for i in range(count):
    #             operation_dict_copied[type_string].remove('to_be_removed')
    #         return [dropped_list, operation_dict_copied[type_string]]
    #
    #     [normal, normal_primitive_dict] = gen_op_dct('normal')
    #     [reduce, reduce_primitive_dict] = gen_op_dct('reduce')
    #     return [[normal, normal_primitive_dict], [reduce, reduce_primitive_dict]]
    def new(self, normal, reduce):
        """
        生成新的模型
        """
        # 把列表内的节点索引映射到全局索引
        global_normal = []
        global_reduce = []
        reg_num = 2
        for idx in normal:
            for i in idx:
                global_normal.append(i + reg_num)
            reg_num += len(idx) + 2
        reg_num = 2
        for idx in reduce:
            for i in idx:
                global_reduce.append(i + reg_num)
            reg_num += len(idx) + 2
        normal_beta = np.array([])
        reduce_beta = np.array([])
        for i in range(14):
            if i in global_normal:
                continue
            normal_beta = np.append(normal_beta, self._betas['normal'][i].detach().cpu())
        for i in range(14):
            if i in global_reduce:
                continue
            reduce_beta = np.append(reduce_beta, self._betas['reduce'][i].detach().cpu())
        order_dct = {'normal': normal, 'reduce': reduce}
        new_model = NetWork(self._channels, self._classes_nums, self._layers, self.criterion,
                            per_order_info=order_dct).cuda()

        new_model.update_foward(order_dct)
        dct = {
            'normal': torch.from_numpy(normal_beta).cuda(),
            'reduce': torch.from_numpy(reduce_beta).cuda()
        }
        new_model._betas = dct
        new_state_dict = self._get_new_state_dict(global_normal, global_reduce, new_model.eval().state_dict())
        new_model.load_state_dict(new_state_dict)

        [alpha_dicts, beta_dicts, mask_dicts, mask_times_dicts, accumulate_alpha_dicts] = self.__get_parameters(
            normal, reduce)

        dict_ = {
            'alpha': alpha_dicts,
            'beta': beta_dicts,
            'mask': mask_dicts,
        }

        new_model.__set_parameters(dict_)
        return new_model

    def _get_new_state_dict(self, normal, reduce, new_dict):
        """
        从列表中找到state_dict对应的key
        :param new_dict: 新模型字典
        :return: 新字典的key
        """

        def generate_re_list(cell_list, dropped_idx_list):
            """
            :param cell_list: 细胞的索引列表:  [0,1,..]
            :param dropped_idx_list: 需要删除的操作索引列表: [0,1,..]
            :return: 需要删除的字典键的正则表达式
            """
            temp_list = []
            """
                _cells.1._ops.2._ops.3 <==>   第1个细胞的第2个MixedOp的第3个Op
            """
            # if not non_full:
            #     for idx in range(len(dropped_idx_list)):
            #         dropped_key_re = r'^_cells\.' + str(cell_list) + '\._ops.' + str(idx) + '\._ops\.' + str(
            #             dropped_idx_list[idx])
            #         temp_list.append(dropped_key_re)
            # 非完全连接删除阶段就不需要细分MixedOp了
            temp_list = [r'^_cells\.' + str(cell_list) + '\._ops\.' + str(idx) for idx in dropped_idx_list]
            return temp_list

        cell_nums = arg.layers
        reduce_cell_idx = [cell_nums // 3, 2 * cell_nums // 3]
        normal_cell_idx = [idx for idx in range(cell_nums)]
        normal_cell_idx = list(set(normal_cell_idx).difference(set(list(reduce_cell_idx))))
        dropped_key = []
        normal_dropped_key_re_list, reduce_dropped_key_re_list = \
            generate_re_list(normal_cell_idx, normal), \
            generate_re_list(reduce_cell_idx, reduce)
        old_dict = self.eval().state_dict()
        for each_key, _ in old_dict.items():
            for each_re in normal_dropped_key_re_list:
                if re.match(each_re, each_key) is not None: dropped_key.append(each_key)
            for each_re in reduce_dropped_key_re_list:
                if re.match(each_re, each_key) is not None: dropped_key.append(each_key)
        for each_dropped_key in dropped_key: del old_dict[each_dropped_key]
        new_dict = collections.OrderedDict([(k, v) for k, v in zip(new_dict.keys(), old_dict.values())])
        return new_dict

    @logic_rewrite
    def accumulate_alpha(self):
        """
        计算累计的alpha, 在计算时要除learning_rate, 恢复数量级
        """

        def delta(type_string, obj):
            return torch.abs(
                torch.sub(self._alphas_copy[type_string], obj.clone().detach())
            ) / arg.learning_rate

        delta_alpha = {
            'normal': delta('normal', self._alphas['normal']),
            'reduce': delta('reduce', self._alphas['reduce'])
        }

        self._accumulate['normal'].add_(delta_alpha['normal'])
        self._accumulate['reduce'].add_(delta_alpha['reduce'])

    @logic_rewrite
    def copy_alphas(self):
        """
        保留上一轮的alpha信息
        """
        setattr(self, '_alphas_copy', {
            'normal': self._alphas['normal'].detach().clone(),
            'reduce': self._alphas['reduce'].detach().clone(),
        })

    @logic_rewrite
    def print_dict(self):
        print(self._operation_dict)

    @logic_rewrite
    def print_eva_times(self):
        print(self._eva_times['normal'])

    @logic_rewrite
    def get_alpha_delta(self) -> [torch.tensor, torch.tensor]:
        """
            获取delta alpha的值
        """

        def __cal_delta(type_string):
            alpha_delta = self._alphas[type_string] - self._alphas_copy[type_string]
            # for line in range(lines): alpha_delta[line] = normalize(alpha_delta[line])
            return alpha_delta

        lines = self._accumulate['normal'].shape[0]
        normal_alpha_delta = __cal_delta('normal')
        reduce_alpha_delta = __cal_delta('reduce')

        return [normal_alpha_delta, reduce_alpha_delta]

    @logic_rewrite
    def _get_explore_delta(self):
        """
            计算explore delta 返回一个列表 分别是正常细胞的探索delta矩阵和缩小细胞的探索delta矩阵
            explore delta 是和由reduce time计算的
        """

        def get_explore_item(shape_, reduced_times):
            """
            返回探索项
            :param shape_: deltaAlpha矩阵的大小
            :param reduced_times: 演化次数矩阵
            """
            # 0初始化
            explore_delta = np.zeros(shape_)
            # 到非完全连接阶段可能传递向量 做一次特判 非完要全连接阶段不再探索
            if len(shape_) <= 1: return explore_delta
            # 每一行的长度
            explore_delta_lens_nums = shape_[0]
            explore_delta_len = shape_[1]

            """
                delta = sqrt( 2 * (n) / log(sum(n)))
            """
            each_row_sum = np.sum(reduced_times, 1)
            for each_line_idx in range(explore_delta_lens_nums):
                for each_row_idx in range(explore_delta_len):
                    explore_delta[each_line_idx][each_row_idx] = math.sqrt(
                        (2 * (reduced_times[each_line_idx][each_row_idx]) / (math.log(each_row_sum[each_line_idx]))))
            return explore_delta

        # lambda是探索系数

        lambda_ = arg.explore_lambda
        normal_explore_delta = lambda_ * get_explore_item(self._alphas['normal'].shape,
                                                          self._eva_times['normal'].cpu().numpy())
        reduce_explore_delta = lambda_ * get_explore_item(self._alphas['reduce'].shape,
                                                          self._eva_times['reduce'].cpu().numpy())
        return [torch.tensor(normal_explore_delta).cuda(), torch.tensor(reduce_explore_delta).cuda()]

    @logic_rewrite
    def _get_mask_arg_target(self, non_full=False) -> [list, list]:
        """
        计算需要被演化的操作的索引 返回两个列表(normal和reduce)
        """

        def generate_arg_target_list(explore_parameter) -> list:
            """
            :param explore_parameter: 探索参数 和alpha/mask的形状一致
            :return: 每两个结点之间需要被演化操作的索引的排序(升序)列表
            因为列表而不返回具体索引的原因是argmin操作可能会选取到mask已经是0的操作
            """
            lines = explore_parameter.shape[0]
            arg_target_list = []
            for idx in range(lines): arg_target_list.append(torch.argsort(explore_parameter[idx]))
            return arg_target_list

        def generate_arg_target_list_non_full(parameter):
            f"""
                功能类似于{generate_arg_target_list}, 是非完全连接阶段的版本
            """
            non_fulls = []
            lines = parameter.shape[0]
            begin_connections = 0
            """
                只有到非完全连接阶段的模型才会需要non_full_info属性
                non_full_info 保存非完全连接的长度信息
            """
            if not hasattr(self, 'non_full_info'):
                step_connections = [k + 2 for k in range(self._steps)]
                setattr(self, 'non_full_info', step_connections)
            else:
                step_connections = self.non_full_info
            non_full_info_idx = 0
            # 把.._delta切分保存在non_fulls
            while lines - step_connections[non_full_info_idx] >= 0:
                lines -= step_connections[non_full_info_idx]
                non_fulls.append(
                    parameter[begin_connections:begin_connections + step_connections[non_full_info_idx]].detach())
                begin_connections += step_connections[non_full_info_idx]
                non_full_info_idx = min(non_full_info_idx + 1, len(step_connections) - 1)
            arg_target_list = []
            reg_num = 0
            # 遍历non_fulls 保存最小的值
            for each in non_fulls:
                arg_target_list.append([idx + reg_num for idx in arg_sort(each)])
                reg_num = len(each)
            return arg_target_list

        normal_delta, reduce_delta = self.get_alpha_delta(non_full)
        if not non_full:
            normal_explore_delta, reduce_explore_delta = self._get_explore_delta()
            normal_arg_target_list = generate_arg_target_list(normal_explore_delta + normal_delta)
            reduce_arg_target_list = generate_arg_target_list(reduce_explore_delta + reduce_delta)
        else:
            normal_arg_target_list = generate_arg_target_list_non_full(normal_delta)
            reduce_arg_target_list = generate_arg_target_list_non_full(reduce_delta)

        return [normal_arg_target_list, reduce_arg_target_list]

    @logic_rewrite
    def __update_non_full_info(self):
        setattr(self, 'non_full_info', [max(2, k - 1) for k in self.non_full_info])

    @logic_rewrite
    def mask(self, non_full=False):
        """
        收缩操作
        """

        def __get_idx(arg_target_list, line, obj):
            idx = 0
            to_masked_idx = arg_target_list[line][idx]
            while obj[line][idx] == 0:
                idx += 1
                to_masked_idx = arg_target_list[line][idx]
                assert idx < inner_range, logging.error("错误: 找不到可以被演化的对象!")
            return to_masked_idx

        def __inner_change(obj, to_masked_idx, is_reduce):
            obj[line_idx][to_masked_idx].sub_(mask_rate)
            self._add_eva_time(line_idx, to_masked_idx, is_reduce)

        def __get_mask(g):
            for idx in g:
                if self._masks['normal'][idx] != 0: return idx
            return g[0]

        def __mask(arg_target_non_full, obj):
            flag_ = False
            for g in arg_target_non_full:
                g_idx = __get_mask(g)
                obj[g_idx].sub_(mask_rate)
                if obj[g_idx] <= 0: flag_ = True
            return flag_

        normal_arg_target_list, reduce_arg_target_list = self._get_mask_arg_target(non_full=non_full)
        mask_rate = 1
        flag = False
        if not non_full:
            lens = len(normal_arg_target_list)

            inner_range = self._masks['normal'].shape[1]
            for line_idx in range(lens):
                normal_to_masked_idx = __get_idx(normal_arg_target_list, line_idx, self._masks['normal'])
                reduce_to_masked_idx = __get_idx(reduce_arg_target_list, line_idx, self._masks['reduce'])

                __inner_change(self._masks['normal'], normal_to_masked_idx, False)
                __inner_change(self._masks['reduce'], reduce_to_masked_idx, True)

                if self._masks['normal'][line_idx][normal_to_masked_idx] <= 0 or self._masks['reduce'][line_idx][
                    reduce_to_masked_idx] <= 0:
                    flag = True
        else:
            flag = __mask(normal_arg_target_list, self._masks['normal']) or __mask(reduce_arg_target_list,
                                                                                   self._masks['reduce'])
        return flag

    @logic_rewrite
    def _add_eva_time(self, row_idx, column_idx, is_reduce):
        """
        演化次数加一
        :param row_idx: 连接索引
        :param column_idx: 操作索引
        :param is_reduce: 缩小细胞指针
        """
        if is_reduce:
            self._eva_times['reduce'][row_idx][column_idx].add_(1)
        else:
            self._eva_times['normal'][row_idx][column_idx].add_(1)

    @logic_rewrite
    def arch_parameters(self):
        return self._arch_parameters

    @logic_rewrite
    def __get_parameters(self, normal, reduce, alpha=True, beta=True) -> List[dict]:
        """
        返回参数
        :param normal:      normal dropped list
        :param reduce:      normal dropped list
        :param alpha:       if true     return alpha       else None
        :param mask:        if true     return mask        else None
        :param times:       if true     return mask times else None
        :param accumulate:  if true     return accumulate  else None
        """

        def _get_parameters_(obj: torch.tensor, dropped_list: list) -> torch.tensor:
            """
            从droppedList中取出应该删除的索引, 删除掉同样长度的obj中的元素
            :param obj: 待操作元素
            :param dropped_list: 删除元素索引
                droppedList.idx -> obj 行索引
                droppedList[idx] -> obj 列索引
            :return:
            """
            k = len(dropped_list)
            temp_tensor = torch.zeros((1, obj.shape[0])).cuda() if len(obj.shape) <= 1 else torch.zeros(
                (1, max(1, obj.shape[1] - 1))).cuda()
            # 当进入非完全连接阶段时
            if len(obj.shape) == 1 or obj.shape[1] == 1:
                temp = [0] + [each.item() for each in obj]
                for idx in dropped_list:
                    temp[idx] = 'X'
                for i in range(k):
                    temp.remove('X')
                temp_tensor = torch.tensor(np.array(temp), requires_grad=obj.requires_grad)
            else:
                # 这个方法不会因为删除索引而出错 因为每一行只会进行一次删除
                for i in range(k):
                    temp_tensor = torch.cat(
                        (temp_tensor, del_tensor_element(obj[i], dropped_list[i])
                         .resize(1, temp_tensor.shape[1])), 0)
            return temp_tensor[1::]

        if alpha:
            alpha_return = {'normal': _get_parameters_(self._alphas['normal'], normal),
                            'reduce': _get_parameters_(self._alphas['reduce'], reduce), }
        else:
            alpha_return = None
        if beta:
            beta_return = {'normal': _get_parameters_(self._betas['normal'], normal),
                           'reduce': _get_parameters_(self._betas['reduce'], reduce)}
        else:
            beta_return = None
        return [alpha_return, beta_return]

    @logic_rewrite
    def __set_parameters(self, dict_, alpha=True, mask=True, times=True, accumulate=True):
        """
        生成新模型后的关键参数设置
        :param dict_: 参数字典 keys = ['normal', 'reduce']
        :param type_string: 参数指示串 ['alpha', 'mask', 'times']
            python 3.7(当前环境) 没有switch 于是用了if return
        """
        if alpha:
            setattr(self, "_alphas_normal", dict_['alpha']['normal'].detach().clone().requires_grad_())
            setattr(self, "_alphas_reduce", dict_['alpha']['reduce'].detach().clone().requires_grad_())
            setattr(self, "_arch_parameters", [self._alphas['normal'], self._alphas['reduce']])
            self._init_alpha_copy_dct()
        if mask:
            setattr(self, "_mask_normal", dict_['mask']['normal'].detach().clone())
            setattr(self, "_mask_reduce", dict_['mask']['reduce'].detach().clone())
        if times:
            setattr(self, "_eva_times['normal']", dict_['times']['normal'].detach().clone())
            setattr(self, "_eva_times['reduce']", dict_['times']['reduce'].detach().clone())
        if accumulate:
            setattr(self, "_accumulate",
                    {'normal': dict_['accumulate']['normal'].detach().clone(),
                     'reduce': dict_['accumulate']['reduce'].detach().clone()})

    @logic_rewrite
    def __gen_set_pre_idx_list(self, dropped_list_not_reg, type_string):
        """
        生成并设置前驱节点 不是self的前驱节点
        而是self生成新模型的

        """
        attr_str = type_string + '_pre_idx'
        """
            如果此时已经有前驱节点列表 那就直接更新
            如果没有 说明是刚开始非完全连接阶段
        """
        if hasattr(self, attr_str):
            k = 0
            pre_idx = getattr(self, attr_str)[::-1]
            for dropped_reg_idx in dropped_list_not_reg[::-1]:
                pre_idx[k].pop(dropped_reg_idx)
                k += 1
            pre_idx = pre_idx[::-1]
        else:
            pre_idx = []
            for i in self.non_full_info:
                pre_idx.append([j for j in range(i)])
            idx, count = 0, 0
            for each in pre_idx:
                if len(each) <= 2:
                    count += 1
                else:
                    pre_idx[count].remove(dropped_list_not_reg[idx])
                    idx += 1
                    count += 1
        setattr(self, attr_str, pre_idx)

    @logic_rewrite
    def disp_arch_parameters(self, _print=False):
        if _print:
            print("Normal Alphas:{}".format(self._alphas['normal']))
            print("Reduce Alphas:{}".format(self._alphas['reduce']))
            print("Normal Betas:{}".format(self._betas['normal']))
            print("Reduce Betas:{}".format(self._betas['reduce']))
            return
        else:
            return [self._alphas['normal'], self._alphas['reduce'], self._betas['normal'], self._betas['reduce']]
