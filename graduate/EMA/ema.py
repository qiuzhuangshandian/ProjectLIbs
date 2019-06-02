# -*- coding: utf-8 -*-
'''
Exponential smooth(指数平滑)的手工实现(无第三方库)
Author : Kabuto_hui
Date   : 2018.04.19
'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter  #设置科学计数
from color import cnames
#指数平滑算法
def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    s_temp = [0 for i in range(len(s))]
    s_temp[0] = ( s[0] + s[1] + s[2] ) / 3
    # s_temp[0] = 0

    for i in range(1, len(s)):
        s_temp[i] = alpha * s[i] + (1 - alpha) * s_temp[i-1]
    return s_temp

def compute_single(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    return exponential_smoothing(alpha, s)

def compute_double(alpha, s):
    '''
    二次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回二次指数平滑模型参数a, b， list
    '''
    s_single = compute_single(alpha, s)
    s_double = compute_single(alpha, s_single)

    a_double = [0 for i in range(len(s))]
    b_double = [0 for i in range(len(s))]

    for i in range(len(s)):
        a_double[i] = 2 * s_single[i] - s_double[i]                    #计算二次指数平滑的a
        b_double[i] = (alpha / (1 - alpha)) * (s_single[i] - s_double[i])  #计算二次指数平滑的b

    return a_double, b_double

def compute_triple(alpha, s):
    '''
    三次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回三次指数平滑模型参数a, b, c， list
    '''
    s_single = compute_single(alpha, s)
    s_double = compute_single(alpha, s_single)
    s_triple = exponential_smoothing(alpha, s_double)

    a_triple = [0 for i in range(len(s))]
    b_triple = [0 for i in range(len(s))]
    c_triple = [0 for i in range(len(s))]

    for i in range(len(s)):
        a_triple[i] = 3 * s_single[i] - 3 * s_double[i] + s_triple[i]
        b_triple[i] = (alpha / (2 * ((1 - alpha) ** 2))) * ((6 - 5 * alpha) * s_single[i] - 2 * ((5 - 4 * alpha) * s_double[i]) + (4 - 3 * alpha) * s_triple[i])
        c_triple[i] = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single[i] - 2 * s_double[i] + s_triple[i])

    return a_triple, b_triple, c_triple

def Enose_single_exp_avg(alpha,s):

    s_temp = [0 for i in range(len(s))]
    # s_temp[0] = ( s[0] + s[1] + s[2] ) / 3
    s_temp[0] = 0
    # s_temp[1] = 0
    for i in range(1, len(s)):
        s_temp[i] = alpha * (s[i]-s[i-1]) + (1 - alpha) * s_temp[i-1]
    return s_temp
if __name__ == "__main__":
    txtFile = "E:/papers/for_journal/my_target/codes/data1/B1_GEy_F030_R3.txt"
    data =  pd.read_csv(txtFile,sep="\t").values
    v = data[4700:,2]
    x = data[4700:,0]
    
    # data = [i for i in range(100)]
    figsize = (5.5,2)
    plt.figure(figsize=figsize)
    plt.plot(x,v)
    # plt.xlabel("Raw curve")
    plt.ylabel("R(Ω)")

    sigle = compute_single(0.01, v)
    plt.figure(figsize=figsize)
    plt.plot(x,sigle)
    # plt.xlabel("Curve after filtering")
    plt.ylabel("R(Ω)")

    alpha = 0.1
    a= Enose_single_exp_avg(alpha,sigle)
    plt.figure(figsize=figsize)
    plt.plot(x,a,c=cnames["blue"])
    # plt.xlabel(r"$\alpha = 0.1$")
    plt.ylabel("R(Ω)")
    
    alpha = 0.01
    a= Enose_single_exp_avg(alpha,sigle)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    # def formatnum(x, pos):
    #     return '$%.1f$x$10^{-3}$' % (x/1e-3)
    # formatter = FuncFormatter(formatnum)
    ax.plot(x,a,c=cnames["green"])
    # ax.yaxis.set_major_formatter(formatter)
    # plt.xlabel(r"$\alpha = 0.01$")
    ax.set_ylabel("R(Ω)")
    
    

    alpha = 0.001
    a= Enose_single_exp_avg(alpha,sigle)
    plt.figure(figsize=figsize)
    plt.plot(x,a,c=cnames["hotpink"])
    # plt.xlabel(r"$\alpha = 0.001$")
    plt.ylabel("R(Ω)")
    

    plt.show()
