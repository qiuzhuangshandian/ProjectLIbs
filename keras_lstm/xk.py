import utils
import model
import argparse
import mxnet as mx
import warnings
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import parse
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata
from mxnet.gluon.data import dataloader as gloader
import numpy as np
import os
import d2lzh as d2l

#用来接收参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train networks with to predict the network flow .')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Training mini-batch size')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=4096,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./yolo3_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,180',
                        help='epochs at which learning rate decays. default is 160,180.')
    parser.add_argument('--data-path',type=str,default='../traffic-matrices/',
                        help='the file path of your data.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    args = parser.parse_args()
    return args
data_path = '../traffic-matrices/'
num_epochs = 4096
batchsize = 4096
ctx = utils.try_all_gpus()

tms = utils.buildAllTMToATensor(xmlPath=data_path)
train_data = utils.train_dataloader(tms=tms)
test_data = utils.test_dataloader(tms=tms)
data_train = gdata.ArrayDataset(train_data[0],train_data[1])
data_test = gdata.ArrayDataset(test_data[0],test_data[1])
data_train_iter = gdata.DataLoader(dataset=data_train,batch_size=batchsize,shuffle=True)
data_test_iter = gdata.DataLoader(dataset=data_test,batch_size=batchsize,shuffle=False)
# for x, y  in data_test_iter:
#     print(x,y)
#     break
# for x, y  in data_train_iter:
#     print(x,y)
#     break

# ctx = mx.gpu(0)
net = utils.Foward_all_fc(layer_number=10)
net.initialize(init.Xavier())
net.collect_params().reset_ctx(ctx)
net.collect_params()
net.hybridize()
loss = gloss.L1Loss()
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03,'wd':0.001})

def test_net(net):
    test_l_sum, n = 0.0, 0
    for X,y in data_test_iter:
        Xs, ys, batch_size = utils._get_batch(X, y, ctx)
        y_hats = [net(Xx) for Xx in Xs]
        ls = [loss(y_hat,y) for y_hat,y in zip(y_hats,ys)]
        test_l_sum += sum([l.sum().asscalar() for l in ls])
        n += sum([l.size for l in ls])
    return test_l_sum/(n*23*23)

def train_net():
    for epoch in range(1, num_epochs + 1 ):
        train_l_sum, n = 0.0, 0
        for X,y in data_train_iter:
            Xs, ys, batch_size = utils._get_batch(X, y, ctx)
            with autograd.record():
                y_hats = [net(Xx) for Xx in Xs]
                ls = [loss(y_hat,y) for y_hat,y in zip(y_hats,ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size=batchsize)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
        test_loss = test_net(net=net)
        # l = loss(net(train_data[0]),train_data[1])
        # print(l)
        print('epoch%d,\ttrain_loss:%f,\ttest_loss:%f'%(epoch,train_l_sum/(n*23*23),test_loss))
if __name__=="__main__":
    train_net()
