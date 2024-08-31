import torch
import torch.nn.functional as F

class Queue:
    def __init__(self, encoder_k, fc_k, train_loader, arg):
        self.arg = arg
        self.queue = {'feature': [], 'embedding': [], 'cluster': []}

        self.queue['feature'] = torch.zeros((0, self.arg.feat_dim), dtype=torch.float, requires_grad=False).cuda()
        self.queue['embedding'] = torch.zeros((0, self.arg.low_dim), dtype=torch.float, requires_grad=False).cuda()
        self.queue['cluster'] = torch.zeros((0, self.arg.num_cluster), dtype=torch.float, requires_grad=False).cuda()

        for data in train_loader:
            enco_k = encoder_k(data[1].cuda())
            k, w_k = fc_k(enco_k.cuda())
            enco_k, k, w_k = enco_k.detach(), k.detach(), w_k.detach()
            k = F.normalize(k, dim=1)

            self.queue['feature'] = self.queue_data(self.queue['feature'], enco_k)
            self.queue['feature'] = self.dequeue_data(self.queue['feature'], K=10)
            self.queue['embedding'] = self.queue_data(self.queue['embedding'], k)
            self.queue['embedding'] = self.dequeue_data(self.queue['embedding'], K=10)
            self.queue['cluster'] = self.queue_data(self.queue['cluster'], w_k)
            self.queue['cluster'] = self.dequeue_data(self.queue['cluster'], K=10)

    # 新的一批f,z,w入队
    def queue_data(self, data, k):
        return torch.cat([data, k], dim=0)  # data和k按行拼接，将新的f，z，w入队（data是队列中的元素，k是当前新加入的）
    # 最老的一批f,z,w出队，队列总大小为K

    def dequeue_data(self, data, K):
        if len(data) > K:
            return data[-K:]  # 把面多余的元素剔除出列表
        else:
            return data


class MemoryQueue:
    def __init__(self, memory, Queue, arg, minibatch_num):
        self.arg = arg
        self.M_Queue = {'Memory': []}
        self.minibatch_num = minibatch_num

        self.M_Queue['Memory'] = torch.zeros((0, self.arg.num_cluster, self.arg.feat_dim), dtype=torch.float, requires_grad=False).cuda()  # 每一个mc大小为(num_cluster,2048)

        for i in range(self.minibatch_num):

            memory.write(Queue.queue['feature'], Queue.queue['cluster'])
            mc = torch.unsqueeze(memory.memory_item, 0)

            self.M_Queue['Memory'] = self.queue_data(self.M_Queue['Memory'], mc)
            self.M_Queue['Memory'] = self.dequeue_data(self.M_Queue['Memory'], K=self.minibatch_num)

    # 新的一批f,z,w入队
    def queue_data(self, data, k):
        return torch.cat([data, k], dim=0)  # data和k按行拼接，将新的f，z，w入队（data是队列中的元素，k是当前新加入的）
    # 最老的一批f,z,w出队，队列总大小为K

    def dequeue_data(self, data, K):
        if len(data) > K:
            return data[-K:]  # 把面多余的元素剔除出列表
        else:
            return data