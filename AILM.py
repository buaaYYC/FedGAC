import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple

import sys
import time


#进度迭代
def print_progress_bar(iteration, total=None, length=50, prefix='Progress:', suffix='Complete', decimals=1, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * min(iteration / float(total), 1)) if total else '?'
    filled_length = int(length * min(iteration / float(total), 1)) if total else 0
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s ' % (prefix, bar, percent, suffix))
    sys.stdout.flush()

def get_subset_loader(loader, subset_percentage=0.2):
    """
    获取数据加载器的子集加载器。

    参数:
    - loader: 要获取子集的数据加载器。
    - subset_percentage: 子集占原始数据集的百分比。

    返回:
    - subset_loader: 子集数据加载器。
    """

    # 获取加载器的数据集
    dataset = loader.dataset

    # 计算子集的大小
    subset_size = int(len(dataset) * subset_percentage)

    # 计算随机采样的数量
    rand_ratio = subset_percentage * 100
    rand_num = int(rand_ratio * len(dataset) / 100)
    print("采样数：",rand_num)

    # 生成随机起始索引
    rand_idx = random.randint(0, len(dataset) - rand_num)

    # 获取随机子集的索引
    subset_indices = range(rand_idx, rand_idx + rand_num)

    # 使用 DataLoader 的 collate_fn 参数来确保图像维度正确
    subset_loader = DataLoader(
        dataset,
        batch_size=loader.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=lambda x: x  # 保持单个图像的结构
    )

    return subset_loader


class AILM:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple], 
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 10) -> None:
        """
        Initialize AILM module

        Args:
            cid: Client ID. 
            loss: The loss function. 
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True

    def global_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:

        #===================================================
        # obtain the references of the parameters
        params_g = list(global_model.parameters()) #只有几层
        params = list(local_model.parameters())

        # deactivate AILM at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers（只更新全局前几层给--客户端参数）
        # 1. 局部模型前五层参数 更新为 全局模型（前5层）参数 
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()



    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.
        """
        #===================================================
        # randomly sample partial local training data
        # loader = self.train_data
        # 为了简化，直接用全部数据
        rand_loader = self.train_data
        dataset = rand_loader.dataset
        #选取80%做训练 w
        # rand_loader = DataLoader(dataset, batch_size=int(len(dataset)*self.rand_percent), shuffle=True)

        print("len(dataset):",len(rand_loader)) 
        # rand_loader = get_subset_loader(loader,self.rand_percent)
        #===================================================
        
        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate AILM at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers（只更新全局前几层给--客户端参数）
        # 1. 局部模型前五层参数 更新为 全局模型（前5层）参数 
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()


        # temp local model only for weight learning
        # 2. 定义一个临时局部模型，主要用来学习权重（确定合并中全局模型占比）
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:] # 局部的后几层模型参数
        params_gp = params_g[-self.layer_idx:] # 全局的后几层模型参数
        params_tp = params_t[-self.layer_idx:] # 临时局部的后几层模型参数

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]: #临时模型参数的前5层不用更新
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None: 
            #给所有参数初始化一个权重值（这个权重不是模型参数，是模型占的比例）
            # 使用torch.ones_like创建形状相同，所有元素值为1的张量
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p] 

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            # 初始化 临时模型的高层参数 = 局部模型参数 + （全局模型参数-局部模型参数）* W
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        max_e = 1000
        while True:
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y) # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    # 使用torch.clamp将元素限制在范围[0, 1]内
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                # 更新临时局部模型参数值
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1
            # print("epochs",cnt)
            time.sleep(0.1)
            # print(cnt)
            print_progress_bar(cnt, total=max_e)
            
            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold or cnt >= max_e:
                print("\n")
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                    '\tAILM epochs:', cnt)
                # print('Client:', '\tStd:', np.std(losses[-self.num_pre_loss:]),
                #     '\tAILM epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()