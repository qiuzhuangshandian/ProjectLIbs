

import math
import plotSetting as ps
import matplotlib
import matplotlib.pyplot as plt
font = {'family': 'serif',
        'color': 'black',
        'weight': 'bold',
        'size': 24,
        }
'''set the direction of the ticks
'''
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'



def sigmoid(x):
    return 1/(1+math.exp(-x))
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def relu(x):
    return 0 if x<0 else x

def draw_sigmoid(x):
    y = [0 for i in range(len(x))]

    for i in range(len(x)):
        y[i] = sigmoid(x[i])
    
    
    ax = ps.myNewStyleFiure()
    ax.plot(x,y,linewidth=2.5)



def draw_tanh(x):
    y = [0 for i in range(len(x))]

    for i in range(len(x)):
        y[i] = tanh(x[i])
    
    ax = ps.myNewStyleFiure()
    ax.plot(x,y,linewidth=2.5)


def draw_relu(x):
    y = [0 for i in range(len(x))]

    for i in range(len(x)):
        y[i] = relu(x[i])
    
    ax = ps.myNewStyleFiure()
    ax.plot(x,y,linewidth=2.5)


if __name__=="__main__":
    x = [i*0.01 for i in range(-500,500)]
    draw_sigmoid(x)

    draw_tanh(x)

    draw_relu(x)

    plt.show()



