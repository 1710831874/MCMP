import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import torch.nn.functional as F
from sklearn.manifold import TSNE
from utils import evaluator

def test(testloader, Network_q, Network_FC_q, memory, args, queue=None, viz=None):
    with torch.no_grad():
        Network_q.eval()
        Network_FC_q.eval()

        pred = []
        encode_list = []
        cluster_list = []

        for i, data in enumerate(testloader):
            with torch.no_grad():  # 不需要反向传播（减少计算开销）

                encode = Network_q(data.cuda())
                embedding, cluster = Network_FC_q(encode.cuda())

                read_item = memory.read(cluster)
                predict = torch.sum((read_item - encode) ** 2, dim=1)

                for j in range(len(predict)):
                    pred.append(predict[j].cpu().detach())
                    encode_list.append(encode[j].cpu().detach())
                    cluster_list.append(cluster[j].cpu().detach())

        # 加载真实标签
        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        # pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        # # 打印分数和标签
        # for n in range(len(gt)):
        #     if gt[n] == 0:
        #         print("真实标签为：")
        #         print(gt[n])
        #         print("预测分数为：")
        #         print(pred[n])

        # 计算auc和roc曲线
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        # np.save('fpr.npy', fpr)
        # np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))
        # print('threshold : ' + str(threshold))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)

        # 绘图
        if args.viz:

            viz.plot_lines('pr_auc', pr_auc)
            viz.plot_lines('auc', rec_auc)
            viz.lines('score', pred)
            viz.lines('gt', gt)
            viz.lines('roc', tpr, fpr)  # 绘制roc曲线

            # y1 = pred[:2000]
            # y2 = gt[:2000]
            # viz.two_lines('pre&gt', y1, y2)

            # if args.draw:
            #     # 统计w的情况
            #     # 测试特征
            #     w_queue_np = torch.tensor([item.cpu().detach().numpy() for item in cluster_list])
            #     w_queue_np = w_queue_np.numpy()
            #
            #     def get_predictions_from_b(b):
            #         # 计算每个样本在b中的最大预测权重对应的索引
            #         predicted_classes = np.argmax(b, axis=1)
            #         # 将预测类别从0到15映射到1到16
            #         predicted_classes += 1
            #         return predicted_classes.tolist()
            #
            #     predicted_classes_1 = get_predictions_from_b(w_queue_np)
            #
            #     from collections import Counter
            #     # 使用Counter统计元素频率
            #     frequency_count = Counter(predicted_classes_1)
            #     # 循环打印1到16每个元素出现的次数
            #     for i in range(1, 21):
            #         print(f"第 {i} 个簇拥有 {frequency_count[i]} 个特征")
            #
            #     # 测试集散点图
            #     a = torch.tensor([item.cpu().detach().numpy() for item in encode_list])
            #     # 绘制散点图
            #     lable = []
            #     for i in range(int(len(gt) / 16)):
            #         for j in range(16):
            #             if gt[j + i * 16] == 1:
            #                 flag = 1
            #                 break
            #             else:
            #                 flag = 0
            #         if flag == 1:
            #             lable.append(2)
            #         else:
            #             lable.append(1)
            #
            #     b = memory.global_memory_item.cpu()
            #     # b = memory.memory_item.cpu()
            #     for i in range(len(b)):
            #         lable.append(3)
            #
            #     # 计算每个特征向量的L2范数
            #     norms = torch.norm(b, p=2, dim=1)
            #     # 对特征向量进行L2正则化
            #     b = b / norms.view(-1, 1)
            #     c = torch.cat((a, b), dim=0)
            #     c = F.normalize(c, dim=1)
            #
            #     tsne = TSNE(
            #         n_components=2,
            #         # init='pca',
            #         perplexity=20,
            #         metric="euclidean",
            #         n_jobs=8,
            #         random_state=10,
            #         verbose=True,
            #     )
            #     # 调用
            #     embedding_test = tsne.fit(c)
            #     # 绘制
            #     viz.scatter(name='测试特征散点图', datax=embedding_test.embedding_, datay=lable)

                # # 训练队列特征散点图
                # w = queue.queue['cluster'].cpu()
                # w = w.numpy()
                #
                # predicted_classes_1 = get_predictions_from_b(w)
                #
                # from collections import Counter
                # # 使用Counter统计元素频率
                # frequency_count = Counter(predicted_classes_1)
                # # 循环打印1到16每个元素出现的次数
                # for i in range(1, 21):
                #     print(f"第 {i} 个簇拥有 {frequency_count[i]} 个特征")
                #
                # # 队列散点图
                # a_train = queue.queue['feature'].cpu()
                # a_train = F.normalize(a_train, dim=1)
                # c_train = torch.cat((a_train, b), dim=0)
                # c_train = F.normalize(c_train, dim=1)
                #
                # lable2 = [1] * len(a_train)
                # for i in range(len(b)):
                #     lable2.append(2)
                #
                # # 调用
                # embedding_train = tsne.fit(c_train)
                # # 绘制
                # viz.scatter2(name='训练队列散点图', datax=embedding_train.embedding_, datay=lable2)

        return rec_auc
   

