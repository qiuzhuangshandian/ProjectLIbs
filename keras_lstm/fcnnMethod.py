import tensorflow as tf

import time,os
import utils,json
from dataLoader import dataloader
import numpy as np
import datetime
# from matplotlib import pyplot
import matplotlib.pyplot as pyplot

print("version of tensorflow:",tf.__version__)

# tf.enable_eager_execution()

# ds_1_end =list(range(1,10))
# ds_list=[0]
ds_list=list(range(0,100,2))
target_pos = 1
batchsize = 64
#**************prepare data*********************
tms = utils.buildAllTMToATensor(xmlPath = "../traffic-matrices/")
# data_loader = dataloader(tms,EMA=False,alpha=0.9)

data_loader = dataloader(tms,ratio_for_train=0.85,EMA=False,alpha=0.1,BOX=True,normalization=1)

train_x,train_y = [],[]
# gen = data_loader.iter_for_train(ds_list = ds_list,batchsize=1,target_pos=1 )
for x,y in data_loader.iter_for_train(ds_list = ds_list,batchsize=1,target_pos=target_pos ):
    b, l, m, n = x.shape
    b_, m_, n_ = y.shape
    # tmp_x =
    train_x.append(x.reshape([l*m*n]))

    train_y.append(y.reshape([m_*n_]))
train_x = np.array(train_x)
train_y = np.array(train_y)
print("train x shape:",train_x.shape)
print("train y shape:",train_y.shape)
test_x,test_y = [],[]
for x,y in data_loader.iter_for_test(ds_list = ds_list,batchsize=1,target_pos=target_pos ):
    b, l, m, n = x.shape
    b_, m_, n_ = y.shape
    test_x.append(x.reshape([l* m * n]))
    test_y.append(y.reshape([m_ * n_]))
test_x = np.array(test_x)
test_y = np.array(test_y)
print("test_x shape:",test_x.shape)
print("test_y shape:",test_y.shape)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(150, input_shape=[len(ds_list)*23*23], kernel_initializer='uniform', activation='relu'))
# model.add(tf.keras.layers.Dense(500, kernel_initializer='uniform', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(150, kernel_initializer='uniform', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dense(500, kernel_initializer='uniform', activation='relu'))
model.add(tf.keras.layers.Dense(23*23, kernel_initializer='uniform', activation="sigmoid"))


# optimizer = tf.keras.optimizers.SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
# model.compile(loss='mean_absolute_error', optimizer=optimizer,metrics=['mean_absolute_error'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])#, 'mean_squared_error'
model.summary()
history = model.fit(train_x,train_y,batch_size=batchsize,epochs=70,validation_data=(test_x, test_y),verbose=1)


fw = open("../output_data/record_ann.json","w")
save_dict = {"train_loss":history.history['loss']
             ,"test_loss":history.history['val_loss']}
#画训练和测试的损失值曲线图
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

max_back = data_loader.max_train_x.reshape([-1])[53]
yhat = model.predict(train_x)
pyplot.figure()
pyplot.plot(yhat[:,53]*max_back, label='predict')
pyplot.plot(train_y[:,53]*max_back, label='true')
pyplot.legend()
pyplot.title("train set validation itself")

yhat = model.predict(test_x)
save_dict["test_pres"] = yhat.tolist()
save_dict["test_true"] = test_y.tolist()
# print(yhat.shape)
pyplot.figure()
pyplot.plot(yhat[:,53]*max_back, label='predict',marker="*",markersize = 5)
pyplot.plot(test_y[:,53]*max_back, label='true',marker="*",markersize = 5)
pyplot.legend()
pyplot.title("test set validation")


yhat = model.predict(test_x[-200:])
# print(yhat.shape)
pyplot.figure()
pyplot.plot(yhat[:,53], label='predict',marker="*",markersize = 5)
pyplot.plot(test_y[-200:,53], label='true',marker="*",markersize = 5)
pyplot.legend()
pyplot.title("test2 set validation ann")
json.dump(save_dict,fw)
fw.close()
pyplot.show()
print("ok！")
