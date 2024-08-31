import visdom
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# 绘图
class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(port=2333, env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))

    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name,
                          opts=dict(title=name)
                          )
        else:
            self.vis.line(X=X, Y=line, win=name,
                          opts=dict(title=name)
                          )

    def two_lines(self, name, y1, y2):
        x = [i for i in range(len(y1))]  # X轴坐标
        self.vis.line(X=x, Y=y1, win=name,
                      opts=dict(title='name',
                                legend=['pre'],
                                linecolor=np.array([[123, 190, 25]])))  # 绘制第一条线
        self.vis.line(X=x, Y=y2, win=name, update='append',
                      opts=dict(title='Line Plot',
                                legend=['pre', 'gt'],
                                linecolor=np.array([[30, 144, 255]])))  # 绘制第二条线（添加到原图中）

    def scatter(self, name, datax, datay):
        markercolor = np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255]])
        self.vis.scatter(X=datax,  # 值域0~1, 表示要展示的散点数据
                         Y=datay,  # 值域1~2, 每一个数据的类别，将以其对应的colors中的颜色来显示
                         win=name,
                         opts=dict(legend=['N', 'A', 'C'],
                                   markersymbol='dot',
                                   markersize=5,
                                   markercolor=markercolor
                                   )
                         )

    def scatter2(self, name, datax, datay):
        markercolor = np.array([[0, 255, 0], [0, 0, 255]])
        # shapes = {1:'dot', 2:'star'}
        # # 将标签值映射到对应的形状
        # marker_symbols = [shapes[label] for label in datay]
        # # 将 marker_symbols 转换为字符串类型
        # marker_symbols = [str(symbol) for symbol in marker_symbols]
        self.vis.scatter(X=datax,  # 值域0~1, 表示要展示的散点数据
                         Y=datay,  # 值域1~2, 每一个数据的类别，将以其对应的colors中的颜色来显示
                         win=name,
                         opts=dict(legend=['N', 'C'],
                                   markersymbol='dot',
                                   markersize=5,
                                   markercolor=markercolor
                                   )
                         )

    def scatter3(self, name, datax, datay):
        markercolor = np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0]])
        self.vis.scatter(X=datax,  # 值域0~1, 表示要展示的散点数据
                         Y=datay,  # 值域1~2, 每一个数据的类别，将以其对应的colors中的颜色来显示
                         win=name,
                         opts=dict(legend=['N', 'A', 'C', 'TRAIN'],
                                   markersymbol='dot',
                                   markersize=5,
                                   markercolor=markercolor
                                   )
                         )


# 保存当前取得最好AUC结果的模型参数
def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
    fo.write(str(test_info["test_AUC"][-1]))
    fo.close()


# 动量更新
def momentum_update(E, E_K, beta=0.999):
    param_k = E_K.state_dict()  # 获取当前k网络的参数
    param_q = E.named_parameters()  # 获取q网络参数
    for n, q in param_q:  # name, param
        if n in param_k:
            param_k[n].data.copy_(beta * param_k[n].data + (1 - beta) * q.data) #动量更新
    E_K.load_state_dict(param_k)


# 调整视频为32段
def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=np.int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


# 测试时计算AUC
def evaluator(predict, target):
    predict_pos = predict[target == 1]
    predict_neg = predict[target != 1]

    # calculate AUC
    truth = np.concatenate((np.zeros_like(predict_neg), np.ones_like(predict_pos)))
    predict = np.concatenate((predict_neg, predict_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, predict)
    roc_auc = auc(fpr, tpr)

    # # PR curve where "normal" is the positive class
    # precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, predict)
    # pr_auc_norm = auc(recall_norm, precision_norm)
    #
    # # PR curve where "anormal" is the positive class
    # precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -predict, pos_label=0)
    # pr_auc_anom = auc(recall_anom, precision_anom)

    # return roc_auc, pr_auc_norm, pr_auc_anom
    return roc_auc