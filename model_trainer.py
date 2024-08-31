import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import NTXentLoss, InfoNCE
# from nt_xent import NTXentLoss
from utils import momentum_update


def train(train_data, E, E_K, FC_Q, FC_K, optimizer_E, memory, queue, memory_queue, epoch, arg, viz=None, print_flag=False, scheduler=None, D=None, optimizer_D=None):
    NCE_loss = InfoNCE(arg)
    # adversarial_loss = torch.nn.BCELoss()
    # cross_entropy = nn.CrossEntropyLoss()
    # cosim = nn.CosineSimilarity()
    ntxent = NTXentLoss(arg.num_cluster, use_cosine_similarity=False, temperature=arg.temp_w)
    # l1_loss = nn.L1Loss()

    E.train()
    E_K.train()
    # 动量更新
    momentum_update(E, E_K, beta=arg.beta)

    # 开始训练，data[0]和data[1]分别为编码器q和k的输入
    encode_q = E(train_data[0].cuda())
    encode_k = E_K(train_data[1].cuda())
    q, w_q = FC_Q(encode_q.cuda())
    k, w_k = FC_K(encode_k.cuda())
    encode_k, k, w_k = encode_k.detach(), k.detach(), w_k.detach()

    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)

    # embedding NCE（计算Lz）
    loss1 = NCE_loss(q, k, queue.queue['embedding'])
    # cluster NCE（计算Lc和Lr）
    w_qn = F.normalize(w_q, dim=0).t()
    w_kn = F.normalize(w_k, dim=0).t()
    loss2 = ntxent(w_qn, w_kn)  # NTXentLoss
    loss3 = torch.sum(torch.sum(w_qn, dim=1) ** 2) / w_q.shape[0] * 0.05  # Lr

    queue.queue['feature'] = queue.queue_data(queue.queue['feature'], encode_k)
    queue.queue['feature'] = queue.dequeue_data(queue.queue['feature'], K=arg.queue_size)
    queue.queue['embedding'] = queue.queue_data(queue.queue['embedding'], k)
    queue.queue['embedding'] = queue.dequeue_data(queue.queue['embedding'], K=arg.queue_size)
    queue.queue['cluster'] = queue.queue_data(queue.queue['cluster'], w_k)
    queue.queue['cluster'] = queue.dequeue_data(queue.queue['cluster'], K=arg.queue_size)

    # 如果epoch >= self.arg.warmup_epoch，则计算Lm
    if epoch >= arg.warmup_epoch:

        # 从内存中读取内存原型
        read_item = memory.read(w_q)
        loss5 = (torch.sum((read_item - encode_q) ** 2)).mean()  # Lm
        loss_all = loss1 + loss2 + loss3 + loss5 * arg.loss5wight

        optimizer_E.zero_grad()
        loss_all.backward()
        optimizer_E.step()
        # 学习率变化
        scheduler.step()

        # 打印loss
        if print_flag:
            print("训练次数: {},Loss1: {},Loss2: {},Loss3: {},Loss5: {}".format(epoch, loss1, loss2, loss3, loss5))
            # 绘制loss5
            if arg.viz:
                viz.plot_lines('loss5', loss5.cpu().item())

        # 更新内存原型
        memory.global_write(queue.queue['feature'], queue.queue['cluster'], memory_queue)

    # 如果epoch < self.arg.warmup_epoch，则不计算Lm
    else:
        loss_all = loss1 + loss2 + loss3
        optimizer_E.zero_grad()
        loss_all.backward()
        optimizer_E.step()
        # 学习率变化
        scheduler.step()

        loss5 = 0

        if print_flag:
            print("训练次数: {},Loss1: {},Loss2: {},Loss3: {}".format(epoch, loss1, loss2, loss3))

    # viz.plot_lines('loss_all', loss_all.item())
    if arg.viz and print_flag:
        viz.plot_lines('loss1', loss1.item())
        viz.plot_lines('loss2', loss2.item())
        viz.plot_lines('loss3', loss3.item())

        if epoch < arg.warmup_epoch:
            viz.plot_lines('loss5', loss5)
        viz.plot_lines('lossc', loss1.item()+loss2.item()+loss3.item())

    return loss1, loss2, loss3, loss5
