# *************************************************************** #
# 代码说明：画各种论文插图
# author: Haien Zhang
# *************************************************************** #
import pandas as pd
import numpy as np
import os
from scipy.fftpack import fft
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import *
import multiprocessing as mp
import scipy.io as sio
import matplotlib.spines as spine
#####
sns.set(style="white", palette="muted", color_codes=True)
matplotlib.rcParams['xtick.direction'] = 'in'   #set the direction of the ticks
matplotlib.rcParams['ytick.direction'] = 'in'

####
TargetName = 'B1_GEy_F070_R2'  #不需要加后缀,只需文件名字
# TargetName = 'B1_GEy_F020_R1'  #不需要加后缀,只需文件名字
# TargetName = 'B4_GEa_F050_R2'  #fft
n_sensor = 3
# n_sensor = 7    #fft
font = {'family': 'serif',
        'color': 'black',
        'weight': 'bold',
        'size': 20,
        }
class config_parameter():
    #定义查找位置
    read_path = [r'data1/unit1',r'data1/unit2',r'data1/unit3',r'data1/unit4']

Global_config = config_parameter()

def find_target_file(path,filename):
    #在文件夹中查找目标文件
    namelist = os.listdir(path)
    filename = filename+'.txt'
    if filename in namelist:
        readpath = path + '/' + filename
        data = pd.read_csv(readpath, sep='\t', header=None)
        return data
    else:
        return None

def drawFigure(df):
    data_ = pd.DataFrame(df.values[:, 1:])
    x = df[0].values
    print(x)
    # def save_mat_file():
    #     sio.savemat('mat_files/sellected_measure.mat', {'sel_axis_x': x.values, 'sel_data': data_.values[:,1:]})
    # save_mat_file()
    pic = plt.figure()
    ax = pic.add_subplot(111)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)
    ax.spines['top'].set_linewidth(2.5)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # ax.axvline(56.00, linestyle='--',c = 'thistle')   #图中竖线位置在此调整
    # ax.axvline(60.00, linestyle='--',c = 'thistle')   #图中竖线位置在此调整
    ax.plot(x, data_[0], label='S1', linewidth=1,c='red')
    ax.plot(x, data_[1], label='S2', linewidth=1,c='blue')
    ax.plot(x, data_[2], label='S3', linewidth=1,c='m')
    ax.plot(x, data_[3], label='S4', linewidth=1,c='yellowgreen')
    ax.plot(x, data_[4], label='S5', linewidth=1,c='g')
    ax.plot(x, data_[5], label='S6', linewidth=1,c='y')
    ax.plot(x, data_[6], label='S7', linewidth=1,c='orange')
    ax.plot(x, data_[7], label='S8', linewidth=1,c='pink')
    # ax.set_xticks(range(0, 601, 100))
    ax.set_yticks(range(0, 100, 10))
    plt.xlabel("Time(s)",fontdict = font)
    plt.ylabel(r"$R(KΩ)$",fontdict = font)
    # plt.title(TargetName)
    legend = ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.3), ncol=1,fontsize = 14,frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    plt.show()
def draw_figure_single_1(df,sensor):
    data_ = pd.DataFrame(df.values[5000:10000, 1:])
    x = df[0][5000:10000]
    pic = plt.figure(figsize=(6.7, 6))
    ax = pic.add_subplot(111)
    ax.plot(x, data_[sensor], label='S'+str(sensor), linewidth=2)
    # ax.set_xticks(range(0, 601, 100))
    # ax.set_yticks(range(0, 121, 20))
    plt.xlabel("Time(s)",fontdict = font)
    plt.ylabel(r"$R(KΩ)$",fontdict = font)
    ax.axvline(56.00, linestyle='--', c='red')  # 图中竖线位置在此调整
    ax.axvline(60.00, linestyle='--', c='red')  # 图中竖线位置在此调整
    plt.title(TargetName+'_sensor'+str(sensor))
    # ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.9), ncol=1, fancybox=True, shadow=True)
    plt.show()
def draw_figure_sensitivity(df,sensor):
    data_ = pd.DataFrame(df.values[5000:10000, 1:])
    x = df[0][5000:10000]
    def save_mat_file():
        sio.savemat('mat_files/sensitivity.mat',{'axis_x':x.values,'axis_y': data_[sensor].values})
    save_mat_file()
    # pic = plt.figure(figsize=(6.7, 6))
    pic = plt.figure()
    ax = pic.add_subplot(111)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)
    ax.spines['top'].set_linewidth(2.5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.plot(x, data_[sensor],c = 'red',label='S' + str(sensor), linewidth=2)
    ##画横线
    up = np.ones(shape=[len(x)])*np.mean((data_[sensor])[0:200])
    down =np.ones(shape = [len(x)])*np.mean((data_[sensor])[-1000:])
    ax.plot(x[:-500], up[:-500], label='b', linewidth=1.5,c = 'black',linestyle = '--')
    ax.plot(x[-2000:], down[-2000:], label='a', linewidth=1.5,c='black',linestyle = '--')
    point = x.values[-600]
    plt.annotate("",(point,up[0]),(point,(up[0]+down[0])/2),fontsize=20,arrowprops=dict(facecolor='green', shrink=0))
    plt.annotate("",(point,down[0]),(point,(up[0]+down[0])/2),fontsize=0.5,arrowprops=dict(facecolor='green', shrink=0))
    ax.text(point,(up[0]+down[0])/2,r'$\Delta$R',fontdict = font)
    ax.text(point+1,up[0],r'R$_b$',fontdict = font)
    plt.xlabel("Time(s)",fontdict = font)
    plt.ylabel(r"R(KΩ)",fontdict = font)
    # plt.title(TargetName + '_SENSOR' + str(sensor))
    # ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.9), ncol=1, fancybox=True, shadow=True)
    plt.show()
def draw_fft_figure(df,sensor):
    data_ = pd.DataFrame(df.values[:, 1:])
    y = data_[sensor][5000:10000]
    L = len(y)
    n_choose = [2 ** (x) for x in range(21)]
    for i in n_choose:
        if i >= L:
            n = i
            break
    fft_ = abs(fft(y,n=n))/L
    x = np.linspace(0,100,n)
    def save_mat_file():
        sio.savemat('mat_files/fftdata_with0Hz.mat',{'axis_f_0':x,'axis_fft_0':fft_})
        sio.savemat('mat_files/fftdata.mat', {'axis_f':x[1:], 'axis_fft': fft_[1:]})
    save_mat_file()
    pic1 = plt.figure() #figsize=(6.7, 6)
    cut = 50           #截取一部分用于画图
    ax1 = pic1.add_subplot(111)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax1.spines['left'].set_linewidth(2.5)
    ax1.spines['bottom'].set_linewidth(2.5)
    ax1.spines['right'].set_linewidth(2.5)
    ax1.spines['top'].set_linewidth(2.5)
    ax1.stem(x[0:cut],fft_[0:cut],'-.')
    plt.ylabel(r'|R(f)|',fontdict = font)
    plt.xlabel('Frequence(Hz)',fontdict = font)
    # plt.title(TargetName+r'_SENSOR'+ str(sensor)+r'_FFT',fontsize=10)
    pic2 = plt.figure()
    # pic2 = plt.figure(figsize=(6.7, 6))
    ax2 = pic2.add_subplot(111)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax2.spines['left'].set_linewidth(2.5)
    ax2.spines['bottom'].set_linewidth(2.5)
    ax2.spines['right'].set_linewidth(2.5)
    ax2.spines['top'].set_linewidth(2.5)
    ax2.stem(x[1:cut],fft_[1:cut],'-.')
    plt.ylabel(r'|R(f)|',fontdict = font)
    plt.xlabel(r'Frequence(Hz)',fontdict = font)
    # plt.title(TargetName + r'_SENSOR' + str(sensor) + r'_FFT(remove 0 Hz)',fontsize=10)
    plt.show()

if __name__=="__main__":
    print('\n', os.getcwd())
    for path in Global_config.read_path:
        print('search in ',path)
        data = find_target_file(path= path,filename=TargetName)
        if data is not None:
            p = mp.Process(target=drawFigure,args=(data,))
            p.start()
               # p.daemon = True
            p2 = mp.Process(target=draw_figure_single_1,args=(data,n_sensor,))
            p2.start()
            p3 = mp.Process(target = draw_figure_sensitivity,args = (data,n_sensor,))
            p3.start()
            p4 = mp.Process(target = draw_fft_figure,args = (data,n_sensor,))
            p4.start()
            break
    # time.sleep(10)