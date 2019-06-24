import xml.etree.cElementTree as ET
from xml.etree.ElementTree import parse
# from mxnet import gluon,init,nd
# from mxnet.gluon import nn
# from mxnet.gluon import utils as gutils
# import mxnet as mx
import numpy as np
import os
import pandas as pd


def GetTMFromSingleXMLFile(fileName):
    doc = parse(fileName)      #解析文档
    root = doc.getroot()    #获取xml根节点
    num_id = 23            #设置总id数目，用于下面的for循环
    mat_dict = {}          #声明一个字典用于存放 所有src，dsts

    for data in root.findall('IntraTM'):
        for ii, item in enumerate(data.findall("src")):

            # srcs.append(item.attrib["id"])
            src_id = item.attrib["id"]
            # tmp_dict = {src_id:[]}

            tmp_dict = {}
            for iii, dst in enumerate(item.findall("dst")):
                value = float(dst.text)        #将str类型转换成float类型
                Id = dst.attrib["id"]          #抽取出目的地id号
                tmp_dict[Id] = value           #存入临时字典

            tmp_list = []
            # print(len(tmp_dict))
            #*************将速率按照id号排序并装入一个list**************
            for k in range(num_id):
                try:
                    tmp_list.append(tmp_dict[str(k+1)])
                except:
                    tmp_list.append(0)           #存在很多缺失值，暂时用0代替
                    # continue
                    # pass

            #*********************************************************
            mat_dict[src_id] = tmp_list

    TM = []
    for i in range(num_id):
        TM.append(mat_dict[str(i+1)])

    TM = np.array(TM)

    return TM


def buildAllTMToATensor(xmlPath = "../traffic-matrices/"):

    year = "2005"
    target_vs = []
    cnt = 0

    fileNames = os.listdir(xmlPath)
    num_files = len(fileNames)     # 确保数据原始文件夹中没有添加多余文件
    tm_tensor = np.zeros(shape=(num_files,23,23))
    # 月
    for month in range(4):
        if month <9:
            s_month = "0{}".format(month+1)
        else:
            s_month = str(month+1)
        #天
        for day in range(31):
            if day < 9:
                s_day = "0{}".format(day+1)
            else:
                s_day = str(day+1)
            # 小时
            for hour in range(24):
                if hour <= 9:
                    s_hour = "0{}".format(hour)
                else:
                    s_hour = str(hour)
                # 分钟
                for minute in range(0,59,15):
                    if minute <=9:
                        s_minute = "0{}".format(minute)
                    else:
                        s_minute = str(minute)
                    # 构建文件名字
                    file = xmlPath+"IntraTM-"+year+"-"+s_month+"-"+ s_day+"-"+s_hour+"-"+s_minute+".xml"
                    try:
                        tm = GetTMFromSingleXMLFile(file)
                        tm_tensor[cnt,:,:] = tm
                        cnt+=1
                        print("\r{}/{}".format(cnt,num_files),end="")
                    except: # 若是无法读取说明文件不存在， 则不做任何操作
                        # print(file)
                        pass
    print("cnt:",cnt)
    return tm_tensor    #返回全部的交通矩阵


def remove_outliers(vs):
    pd_data = pd.DataFrame(vs)
    m, n = pd_data.shape
    describe = pd_data.describe()
    # print(describe)
    Q1 = describe.ix["25%"]
    # 3rd quartile (75%)
    Q3 = describe.ix["75%"]
    # Interquartile range (IQR)
    IQR = Q3 - Q1
    # outlier step
    outlier_step = 1.5 * IQR

    downLimit = Q1 - outlier_step
    upLimit = Q3 + outlier_step
    # print(downLimit,upLimit)
    for col in [0]:
        for i in range(m):
            if pd_data[col][i] > upLimit.ix[col] or pd_data[col][i] < downLimit.ix[col]:
                # print(col,i,upLimit.ix[col],downLimit.ix[col])

                try:
                    pd_data[col][i] = pd_data[col][i - 1]
                    # pd_data[col][i] = pd_data.mean
                    # del pd_data[col][i]
                except:
                    try:
                        pd_data[col][i] = pd_data[col][i - 1]
                    except:
                        pd_data[col][i] = pd_data[col][i + 1]
    # print()
    return pd_data.values.reshape((-1)).tolist()









#
#
# def train_dataloader(tms,data_path = '../traffic-matrices/',train_numbers=20, test_numbers=1,batch_size=20):
#     train_data = tms[:10000,:,:]
#     train_x,train_y = [],[]
#     for i in range(train_data.shape[0]-20):
#         train_x.append(tms[i:i+20,:,:].reshape(23*23*20))
#         train_y.append(tms[i+20,:,:].reshape(23*23))
#     train_x = np.array(train_x)
#     train_y = np.array(train_y)
#     print("the shape of train_x:\t",train_x.shape)
#     print("the shape of train_y:\t",train_y.shape)
#     return nd.array(train_x),nd.array(train_y)
#     # for i in range(train_data.shape[0]-21):
#     #     yield nd.ndarray(train_data[i:i+20,:,:]), nd.ndarray(train_data[i+20,:,:])
#
# def test_dataloader(tms,data_path = '../traffic-matrices/',train_numbers=20, test_numbers=1,batch_size=20):
#     test_data = tms[10000:,:,:]
#     test_x,test_y = [],[]
#     for i in range(test_data.shape[0]-20):
#         test_x.append(tms[i:i+20,:,:].reshape(23*23*20))
#         test_y.append(tms[i+20,:,:].reshape(23*23))
#     test_x = np.array(test_x)
#     test_y = np.array(test_y)
#     print("the shape of test_x:\t",test_x.shape)
#     print("the shape of test_y:\t",test_y.shape)
#     return nd.array(test_x),nd.array(test_y)
#
#
# def Foward_all_fc(layer_number):
#     net = nn.Sequential()
#     assert layer_number-1 > 0, 'the layer Number must bigger than 1'
#     for i in range(layer_number-1):
#         net.add(nn.Dense(1024,activation='relu'),nn.Dropout(0.5),nn.BatchNorm())
#     net.add(nn.Dense(units=529))
#     return net
#
# def try_all_gpus():
#     """Return all available GPUs, or [mx.cpu()] if there is no GPU."""
#     ctxes = []
#     try:
#         for i in range(16):
#             ctx = mx.gpu(i)
#             _ = nd.array([0], ctx=ctx)
#             ctxes.append(ctx)
#     except mx.base.MXNetError:
#         pass
#     if not ctxes:
#         ctxes = [mx.cpu()]
#     return ctxes
#
#
# def try_gpu():
#     """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
#     try:
#         ctx = mx.gpu()
#         _ = nd.array([0], ctx=ctx)
#     except mx.base.MXNetError:
#         ctx = mx.cpu()
#     return ctx
#
# def _get_batch(data,label, ctx):
#     """Return features and labels on ctx."""
#     features, labels = data,label
#     if labels.dtype != features.dtype:
#         labels = labels.astype(features.dtype)
#     return (gutils.split_and_load(features, ctx),
#             gutils.split_and_load(labels, ctx), features.shape[0])

if __name__=="__main__":
    # file = "../traffic-matrices/IntraTM-2005-01-01-09-45.xml"
    # m = GetTMFromSingleXMLFile(file)
    # print(m)
    # print(m.shape)

    tms = buildAllTMToATensor(xmlPath = "../traffic-matrices/")
    aa = dataloader(tms)
    aa.iter_for_train(ds_list = [0,1,2,2,1])
    aa.iter_for_test(ds_list = [0,1,2,2,1])
