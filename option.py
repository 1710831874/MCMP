import argparse

parser = argparse.ArgumentParser(description='MCOD8')
parser.add_argument('--model-name', default='MCOD8', help='name to save model')
parser.add_argument('--dataset', default='ucf', help='dataset to train on (default: )')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
parser.add_argument('--temp_f', type=float, default=0.1, help='Tempture parameters for cross_entropy,loss1')
parser.add_argument('--temp_w', type=float, default=0.1, help='Tempture parameters for NTXent,loss2')
# parser.add_argument('--p', default=0.1, type=float, help='feature dimension')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')

parser.add_argument('--warmup-epoch', default=100, type=int, help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--epochs', default=801, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=6400, type=int, metavar='N', help='mini-batch size')  # 280x20=5600SH  ,  1280x20=25600UCF
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate', dest='lr')  # 0.0001   0.000457
parser.add_argument('--loss5wight', default=0.000005, type=float, help='loss5-wight')  #0.000005

parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--feat-dim', default=2048, type=int, help='feature dimension')  # feature层输出特征的宽度，256,该维度同时也是内存插槽的列数（插槽的行数为cluster层的输出维度）,同时理论上feat_dim = 64*widen_factor
parser.add_argument('--low-dim', default=512, type=int, help='feature dimension')  # embedding层输出特征的宽度
parser.add_argument('--num-cluster', default=20, type=int, help='number of clusters')  # cluster层输出特征的维度（同时也是内存槽的行数）10  ,  13SH  ,  101UCF

parser.add_argument('--queue-size', default=102400, type=int, help='queue size')  # 4096,  4480SH  ,  20480UCF  ，102400 UCF-200
parser.add_argument('--beta', default=0.999, type=float, help='momentum_update')
parser.add_argument('--draw', default=False, type=bool, help='draw')  # 是否在测试时绘制散点图
parser.add_argument('--viz', default=True, type=bool, help='viz')  # 是否启用visdom

parser.add_argument('--num_clip', default=200, type=int, help='number of clip')
