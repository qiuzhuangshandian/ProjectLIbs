
import matplotlib.pyplot as plt
import matplotlib



def myNewStyleFiure():
    pic = plt.figure(figsize=[8,7])
    ax = pic.add_subplot(111)
    # *************四条边线宽度******************
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)
    ax.spines['top'].set_linewidth(2.5)
    # ************刻度属性**************
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    return ax