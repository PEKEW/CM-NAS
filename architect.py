import numpy as np
import torch.optim
from torch.autograd import Variable


def concat(x):
    """
        对输出进行维度重构再拼合
    """
    return torch.cat([each.view(-1) for each in x])


class Architect(object):
    def __init__(self, model, arg):
        self.networkMomentum = arg.momentum
        self.networkWeightDecay = arg.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=arg.arch_learning_rate,
                                          betas=(0.5, 0.999),
                                          weight_decay=arg.weight_decay
                                          )

    def _compute_unrolled_model(self, input_tensor, target, eta, networkOptimizer):
        """
            展开模型的计算图
        """
        loss = self.model.loss(input_tensor, target)
        theta = concat(self.model.parameters()).data
        # 动量梯度或非动量梯度
        try:
            moment = concat(networkOptimizer.state[v]['momentumBuffer'] for v in self.model.parameters()).mul_(
                self.networkMomentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = concat(torch.autograd.grad(loss, self.model.parameters())).data + self.networkWeightDecay * theta
        unrolledModel = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))
        return unrolledModel

    def _construct_model_from_theta(self, theta):
        modelNew = self.model.new()
        modelDict = self.model.stateDict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            vLength = np.prod(v.size())
            params[k] = theta[offset: offset + vLength].view(v.size())
            offset += vLength

        assert offset == len(theta)
        modelDict.update(params)
        modelNew.load_stateDict(modelDict)
        return modelNew.cuda()

    def step(self, trainInput, trainTarget, validInput, validTarget, eta, networkOptimizer, arg, unrolled):
        # 清除上一步更新的残余参数
        self.optimizer.zero_grad()
        # 损失展开计算或直接求导计算
        if unrolled:
            # 展开求导 二级优化
            self.backward_step_unrolled(arg, trainInput, trainTarget, validInput, validTarget, eta,
                                        networkOptimizer)
        else:
            # 不展开 直接求导
            self.backward_step(validInput, validTarget)
        self.optimizer.step()

    def backward_step(self, validInput, validTarget):
        # 正常的反向传播 不展开
        loss = self.model.loss(validInput, validTarget)
        loss.backward()

    def hessian_vector_product(self, vector, input_tensor, target, r=1e-2):
        R = r / concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = self.model.loss(input_tensor, target)
        gradsP = torch.autograd.grad(loss, self.model.arch_parameters())
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)
        loss = self.model.loss(input_tensor, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(gradsP, grads_n)]

    def backward_step_unrolled(self, arg, trainInput, trainTarget, validInput, validTarget, eta, networkOptimizer):
        # 展开的反向传播
        model = self._compute_unrolled_model(trainInput, trainTarget, eta, networkOptimizer)
        loss = arg.mix_lambda * model.loss(validInput, validTarget) + (1 - arg.mix_lambda) * model.loss(trainInput,
                                                                                                        trainTarget)
        # loss = model.loss(validInput, validTarget)
        loss.backward()
        dalpha = [v.grad for v in model.arch_parameters()]
        vector = [v.grad.data for v in model.parameters()]
        implicitGrads = self.hessian_vector_product(vector, trainInput, trainTarget)
        for g, ig in zip(dalpha, implicitGrads):
            g.data.sub_(ig.data, alpha=eta)
        # 等同于backward() 保存一步梯度信息
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
