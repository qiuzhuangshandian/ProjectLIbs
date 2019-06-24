from matplotlib import pyplot
import tensorflow as tf

import os,json
import utils
from dataLoader import dataloader
import numpy as np
# import datetime



#打印tensorflow 版本号
print("version of tensorflow:",tf.__version__)



# 自定义取值距离
# ds_1_end =list(range())
ds_list=list(range(0,100,2))
target_pos = 1
batchsize = 64
#**************prepare data*********************
tms = utils.buildAllTMToATensor(xmlPath = "../traffic-matrices/")
data_loader = dataloader(tms,ratio_for_train=0.85,EMA=False,alpha=0.1,BOX=True,normalization=1)
train_x,train_y = [],[]
# gen = data_loader.iter_for_train(ds_list = ds_list,batchsize=1,target_pos=1 )
for x,y in data_loader.iter_for_train(ds_list = ds_list,batchsize=1,target_pos=target_pos ):
    b, l, m, n = x.shape
    b_, m_, n_ = y.shape
    train_x.append(x.reshape([l,m*n]))
    train_y.append(y.reshape([m_*n_]))
train_x = np.array(train_x)
train_y = np.array(train_y)

test_x,test_y = [],[]
for x,y in data_loader.iter_for_test(ds_list = ds_list,batchsize=1,target_pos=target_pos ):
    b, l, m, n = x.shape
    b_, m_, n_ = y.shape
    test_x.append(x.reshape([l, m * n]))
    test_y.append(y.reshape([m_ * n_]))
test_x = np.array(test_x)
test_y = np.array(test_y)

# 构建模型
model = tf.keras.Sequential()
# lstm
model.add(tf.keras.layers.LSTM(150, input_shape=(len(ds_list),23*23),dropout=0.1,recurrent_dropout=0.5,return_sequences=True))  #,return_sequences=True,stateful=False
# 全连接
# model.add(tf.keras.layers.Dense(1500,kernel_initializer='uniform', activation='relu'))
# # flatten
# model.add((tf.keras.layers.LSTM(250,return_sequences=True)))
# model.add((tf.keras.layers.LSTM(100,return_sequences=True)))
# model.add((tf.keras.layers.LSTM(100,return_sequences=True)))
model.add(tf.keras.layers.Flatten())
# #dropout
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(2500,kernel_initializer='uniform', activation='relu'))
# # batchnormalization
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(500,kernel_initializer='uniform', activation='relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(23*23,activation="sigmoid"))

#定义损失函数 和 优化器
# optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
# optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0,clipnorm=1.)
optimizer = tf.keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004,clipnorm=0.5)
# loss = tf.losses.huber_loss(delta=1.0,)
model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['mean_absolute_error'])#metrics=['mean_absolute_error', 'mean_squared_error']
# model.compile(loss=loss, optimizer=optimizer,metrics=['mean_absolute_error'])#metrics=['mean_absolute_error', 'mean_squared_error']

# model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['mean_absolute_error'])#metrics=['mean_absolute_error', 'mean_squared_error']

# 定义模型存储位置
checkpoint_path = "../models/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# 打印模型结构
model.summary()
#训练模型
history = model.fit(train_x,train_y,batch_size=batchsize,epochs=70,validation_data=(test_x, test_y),verbose=1,callbacks = [cp_callback])

#画训练和测试的损失值曲线图
print("history[loss]:",type(history.history['loss']))
fw = open("../output_data/record_lstm.json","w")
save_dict = {"train_loss":history.history['loss']
             ,"test_loss":history.history['val_loss']}

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# pyplot.show()


yhat = model.predict(train_x)
pyplot.figure()
# ll = list(range(len(yhat[:,53])))
pyplot.plot(yhat[:,53], label='predict')
pyplot.plot(train_y[:,53], label='true')
pyplot.legend()
pyplot.title("train set validation itself")


yhat = model.predict(test_x)

save_dict["test_pres"] = yhat.tolist()
save_dict["test_true"] = test_y.tolist()

# print(yhat.shape)
# ll = list(range(len(yhat[:,53])))
# test_r = test_x.reshape([-1,23*23])
# test_t = test_r[list(range(9,len(test_r),10)),53]
pyplot.figure()
pyplot.plot(yhat[:,53], label='predict')
pyplot.plot(test_y[:,53], label='true',marker="*",markersize = 5)
# pyplot.plot(test_t, label='check',marker="*",markersize = 5)
pyplot.legend()
pyplot.title("test set validation")


yhat = model.predict(test_x[-200:])
# print(yhat.shape)
pyplot.figure()

pyplot.plot(yhat[:,53], label='predict',marker="*",markersize = 5 )
pyplot.plot(test_y[-200:,53], label='true',marker="o",markersize=5)
pyplot.legend()
pyplot.title("test set validation lstm")

json.dump(save_dict,fw)
fw.close()
pyplot.show()


# scores = model.evaluate(test_x, test_y)
# yhat = model.predict(test_x)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#加载模型的函数
# model.load_weights(checkpoint_path)