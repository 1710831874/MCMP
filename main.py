import os
import logging
import random
import time

from torch.utils.data import DataLoader
from model import *
from model_loader import load_model, save_model
from model_trainer import train
from memory import *
from dataset import Dataset
from queue_k import Queue, MemoryQueue
import option
from tqdm import tqdm
from test import test
from utils import Visualizer, save_best_record
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR


if __name__ == '__main__':

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print(torch.backends.cudnn)
    # 设置随机数种子
    setup_seed(708)

    logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# 超参数
    arg = option.parser.parse_args()

# visdom可视化工具
    if arg.viz:
        viz = Visualizer(env='UCF-77复现', use_incoming_socket=False)
    else:
        viz = None

# 加载数据集
    train_loader = DataLoader(Dataset(arg, test_mode=False, unify=False),
                               batch_size=arg.batch_size, shuffle=True,
                               num_workers=arg.workers, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(arg, test_mode=True, unify=False),
                               batch_size=1, shuffle=False,
                               num_workers=arg.workers, pin_memory=False, drop_last=True)  # 69634能被1882整除 ucf_batch_size=1882  ，  8932能被2233整除，shanghai_batch_size=2233
# 实例化modle
    # ResNet模型
    Network_q = Aggregate(arg.feature_size).cuda()
    Network_FC_q = Linear_Module(arg).cuda()      # embedding和clusters层模型
    Network_k = Aggregate(arg.feature_size).cuda()
    Network_FC_k = Linear_Module(arg).cuda()
    # 实例化判别器
    # Network_d = Discriminator(arg.feat_dim, arg.num_cluster).cuda()

# 实例化优化器
    optimizer_E = torch.optim.AdamW([
        {'params': Network_q.parameters(), 'lr': arg.lr, 'betas': (0.9, 0.999), 'weight_decay': 0.0005},
        {'params': Network_FC_q.parameters(), 'lr': arg.lr, 'betas': (0.9, 0.999), 'weight_decay': 0.0005}
    ])
    # optimizer_D = torch.optim.Adam(Network_d.parameters(), lr=arg.lr, betas=(0.9, 0.999), weight_decay=0.0005)

# 余弦退火
    # scheduler = CosineAnnealingLR(optimizer_E, T_max=100, eta_min=0.0001)

# 阶梯下降
    scheduler = StepLR(optimizer_E, step_size=300*20, gamma=1)  # 每step_size轮变化一次，每次变化为：lr = lr * gamma

    # 实例化Memory模块
    memory = Memory(arg)

# 实例化queue
    queue = Queue(Network_k, Network_FC_k, train_loader, arg)

# 实例化MemoryQueue
    memory_queue = MemoryQueue(memory, queue, arg, minibatch_num=len(train_loader))


# 加载上次训练的数据or预训练数据
    load_model(Network_q, Network_k, Network_FC_q, Network_FC_k, memory, queue, optimizer=optimizer_E, dir='ckpt/XXX.pth', pre_dir='pre-ckpt/MCOD8-453-i3d.pth')

# 预先测试
    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = 'auc'  # put your own path here
    # auc = test(test_loader, Network_q, Network_FC_q, memory, arg, queue=queue, viz=viz)

    # save_model(Network_q, Network_k, Network_FC_q, Network_FC_k, memory, queue, optimizer=optimizer_E, save_dir='./ckpt/' + arg.model_name + '-star.pth')

    for step in tqdm(
            range(1, arg.epochs + 1),
            total=arg.epochs,
            dynamic_ncols=True
    ):

        time_start = time.time() #开始计时

        print_flag = True
        for i, data in enumerate(train_loader):

            loss1, loss2, loss3, loss5 = train(data, Network_q, Network_k, Network_FC_q, Network_FC_k, optimizer_E, memory, queue, memory_queue, step, arg, viz=viz, scheduler=scheduler, print_flag=print_flag)
            print_flag = False

            # Log the training loss
            logging.info('Epoch [{}][{}/{}], Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss5: {:.4f}'.format(i + 1, step, arg.epochs, loss1, loss2, loss3, loss5))

        cur_lr = optimizer_E.param_groups[-1]['lr']
        viz.plot_lines('lr', cur_lr)

        time_end = time.time()  # 结束计时
        time_c = time_end - time_start  # 运行所花时间
        print('time cost', time_c, 's')

        # 保存预热
        # if step == arg.warmup_epoch or step % 1000 == 0:
        #     save_model(Network_q, Network_k, Network_FC_q, Network_FC_k, memory, queue, D=Network_d, save_dir='./ckpt/' + arg.model_name + '-warmup{}-i3d.pth'.format(step))
        #     if step == arg.warmup_epoch:
        #         break

        # 训练大于预热轮数时，每50轮进行一次测试
        if step % 1 == 0 and step >= 200:
            auc = test(test_loader, Network_q, Network_FC_q, memory, arg, queue=queue, viz=viz)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                save_best_record(test_info, os.path.join(output_path, '{}-step-pre-auc.txt'.format(step)))
                save_model(Network_q, Network_k, Network_FC_q, Network_FC_k, memory, queue, optimizer=optimizer_E, save_dir='./ckpt/' + arg.model_name + '-{}-i3d.pth'.format(step))

            # 保存每一次的auc
            # save_best_record(test_info, os.path.join('auc-all', '{}-step-pre-auc.txt'.format(step)))

    save_model(Network_q, Network_k, Network_FC_q, Network_FC_k, memory, queue, optimizer=optimizer_E, save_dir='./ckpt/' + arg.model_name + '-final.pth')




