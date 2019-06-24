from utils import GetTMFromSingleXMLFile
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os
from myema import compute_single

def drawTMmat(filename,show = True,save = True):
    tm = GetTMFromSingleXMLFile(filename)
    # tm = tm.astype('float') / tm.sum(axis=1)[:, np.newaxis]
    tm = tm.astype('float') / tm.sum()
    cmap=plt.cm.Blues

    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(tm, interpolation='nearest', cmap=cmap)

    fmt = '.2f'
    thresh = tm.max() / 2.
    # thresh = tm.max()
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            ax.text(j, i, format(tm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if tm[i, j] > thresh else "black")
    ax.set_title(filename.split("/")[-1].split(".")[0],fontsize=12,color='r')
    # fig.tight_layout()
    if not os.path.exists("../figures/TMat/"):
        os.makedirs("../figures/TMat/")
    if save:
        plt.savefig("../figures/TMat/"+filename.split("/")[-1].split(".")[0]+".png")
    if show:
        plt.show()
    plt.close()

def drawAllTmats():
    xmlPath = "../traffic-matrices/"
    fileNames = os.listdir(xmlPath)
    num_files = len(fileNames)
    for i,file in enumerate(fileNames):
        print("\r{}/{}".format(i,num_files),end="")
        # assert file.split(".")[-1]=="xml"
        if file.split(".")[-1]=="xml":
            drawTMmat(xmlPath+file,show=False,save=True)
    print("all tm images have been saved in ../figures/TMat")

def remove_outliers(vs):
    pd_data = pd.DataFrame(vs)
    m,n = pd_data.shape
    describe = pd_data.describe()
    # print(describe)
    Q1 = describe.ix["25%"]
    # 3rd quartile (75%)
    Q3 = describe.ix["75%"]
    # Interquartile range (IQR)
    IQR = Q3 - Q1
    # outlier step
    outlier_step = 1.5 * IQR
    
    downLimit = Q1-outlier_step
    upLimit = Q3+outlier_step
    # print(downLimit,upLimit)
    for col in [0]:
        for i in range(m):
            if pd_data[col][i] > upLimit.ix[col] or pd_data[col][i] < downLimit.ix[col]:
                # print(col,i,upLimit.ix[col],downLimit.ix[col])

                try:
                    pd_data[col][i] = (pd_data[col][i-1]+pd_data[col][i+1])/2
                    # del pd_data[col][i]
                except:
                    try:
                        pd_data[col][i] = pd_data[col][i - 1]
                    except:
                        pd_data[col][i] = pd_data[col][i + 1]
    # print()
    return pd_data.values.reshape((-1)).tolist()


def drawsingleCurves(idPair = [1,2],ema=True,alpha = 0.9,show=False):
    xmlPath = "../traffic-matrices/"
    year = "2005"
    target_vs = []
    for month in range(4):
        if month < 9:
            s_month = "0{}".format(month+1)
        else:
            s_month = str(month+1)

        for day in range(31):
            if day < 9:
                s_day = "0{}".format(day+1)
            else:
                s_day = str(day+1)

            for hour in range(24):
                if hour <= 9:
                    s_hour = "0{}".format(hour)
                else:
                    s_hour = str(hour)

                for minute in range(0,46,15):
                    if minute <= 9:
                        s_minute = "0{}".format(minute)
                    else:
                        s_minute = str(minute)

                    file = xmlPath+"IntraTM-"+year+"-"+s_month+"-"+ s_day+"-"+s_hour+"-"+s_minute+".xml"
                   
                    try:
                        
                        tm = GetTMFromSingleXMLFile(file)
                        
                        target_v = tm[idPair[0]-1,idPair[1]-1]
                        
                        target_vs.append(target_v)
                    except:
                        pass
    

    if not os.path.exists("../figures/pair_cures/"):
        os.makedirs("../figures/pair_cures/")

    plt.figure()
    plt.plot(target_vs,c="g")
    plt.title(str(idPair[0])+"-"+str(idPair[1])+"raw data")
    plt.savefig("../figures/pair_cures/"+"{}_{} raw".format(idPair[0],idPair[1])+".png")
    plt.close()

    target_vs = remove_outliers(target_vs)
    plt.figure()
    plt.plot(target_vs,c="blue")
    plt.title(str(idPair[0])+"-"+str(idPair[1])+"box data")
    plt.savefig("../figures/pair_cures/"+"{}_{} box".format(idPair[0],idPair[1])+".png")
    plt.close()

    if ema:
        target_vs = compute_single(alpha,target_vs)   #指数平滑滤波（ema）

    # print(sum(target_vs)/len(target_vs))
    plt.figure()
    plt.plot(target_vs,c="r")
    plt.title(str(idPair[0])+"-"+str(idPair[1])+"ema data")
    plt.savefig("../figures/pair_cures/"+"{}_{} ema".format(idPair[0],idPair[1])+".png")
    if show:
        plt.show()
    plt.close()


if __name__=="__main__":

    #*********第一部分 画交通举证图***************#
    # drawAllTmats()     # 画交通矩阵图时取消注释


    #**********第二部分 画id对曲线图（不用时自行注释）**************#
    for i in range(9,23):
        for j in range(23):
            drawsingleCurves(idPair = [i+1,j+1],ema = True,alpha=0.5,show=False)
            print(i,j,"ok!")
    #*******************************************#