import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from torch.nn import init


class Memory(nn.Module):
    def __init__(self, arg, memory_init=None, logit=None):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = arg.num_cluster
        self.memory_dim = arg.feat_dim
        self.arg = arg
        self.memory_item = None
        self.global_memory_item = None

        if memory_init is not None:   # 加载现成的内存原型
            self.memory_item = memory_init
        else:                         # 否则初始化一个新的内存原型
            # 初始内存是一个归一化的随机值矩阵，self.memory_size行，self.memory_dim列
            # self.memory_item 就是内存原型mc
            self.memory_item = F.normalize(torch.rand((self.memory_size, self.memory_dim), dtype=torch.float, requires_grad=False), dim=1).cuda()
            self.global_memory_item = self.memory_item

        #     self.memory_center = torch.rand(self.memory_size, dtype=torch.float).cuda()
        # self.att = torch.rand((self.memory_size, self.memory_dim), dtype=torch.float).cuda()

    # 获取Wq和内存原型，计算^f
    def read(self, logit):
        W = F.softmax(logit, dim=1)
        read_item = torch.mm(W, self.global_memory_item)
        return read_item

    # 计算queue数据准备存入内存单元
    def write(self, queue, W_queue):

        V = torch.softmax(W_queue, dim=1)

        self.memory_item = torch.mm(V.t(), queue) #  .t() 表示对softmax（W_queue）进行矩阵转置，torch.mm()是矩阵乘法 Nf x Q  · Q x Nw  =  Nf x Nw
        self.memory_item = F.normalize(self.memory_item, dim=1)

    # # 平方筛选法
    # def square_screening(self, queue, W_queue):
    #     # 转化为np.array
    #     W_queue = np.array(W_queue)
    #     W_queue2 = []
    #     for i in range(len(W_queue)):
    #         sum = 0.0
    #         for j in range(len(W_queue[i])):
    #             sum = sum + (W_queue[i][j] * 10) ** 2  # 每个元素x10并平方求和
    #         W_queue2.append(sum)
    #     # 前80%被保留，W_queue和queue
    #     return

    # 遗忘操作
    def erase_attention(self, W_queue):
        W_queue = torch.softmax(W_queue, dim=1) # [4096,10]
        # torch.argmax(W_queue, dim=1)找到第二维中最大的张量值的索引（0~9之间的一个数）
        # 当unsqueeze()函数的参数是1的时候，该矩阵由（4096，10）变成了（4096,1,10）
        # expand(4096, 10)
        assignment = torch.argmax(W_queue, dim=1).unsqueeze(1).expand(W_queue.shape[0], self.memory_size)
        # torch.arange(self.memory_size)可视化
        # .long() 向下取整
        clusters = torch.arange(self.memory_size).long().cuda()
        # mask：tensor（4096，10）[[true，Flase...]]
        mask = assignment.eq(clusters.expand(W_queue.shape[0], self.memory_size))
        att = (1 - mask.sum(dim=0).float()/W_queue.shape[0]) # 高斯分布方差
        # 改变维度
        att = att.unsqueeze(1).expand(self.memory_size, self.memory_dim) # (10,700)
        # 归一化0
        self.memory_item = torch.normal(mean=self.memory_item, std=att*0.1)


# 计算queue数据准备存入内存单元
    def global_write(self, F_queue, W_queue, memory_queue):

        # 计算当前mini_batch记忆原型mc
        self.write(F_queue, W_queue)
        self.erase_attention(W_queue)
        # mc入队
        mc = torch.unsqueeze(self.memory_item, 0)
        memory_queue.M_Queue['Memory'] = memory_queue.queue_data(memory_queue.M_Queue['Memory'], mc)
        memory_queue.M_Queue['Memory'] = memory_queue.dequeue_data(memory_queue.M_Queue['Memory'], K=25)
        # 计算全局记忆原型
        self.global_memory_item = torch.sum(memory_queue.M_Queue['Memory'], dim=0)
        self.global_memory_item = F.normalize(self.global_memory_item, dim=1)

