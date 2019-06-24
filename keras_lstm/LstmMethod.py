import tensorflow as tf

import time,os
import utils
from dataLoader import dataloader
import numpy as np
import datetime

class Model(tf.keras.Model):
    def __init__(self, out_size, units):
        super(Model, self).__init__()
        self.units = units

        self.bn = tf.keras.layers.BatchNormalization()
        # self.bn2 = tf.keras.layers.BatchNormalization()
        # self.bn3 = tf.keras.layers.BatchNormalization()
        # if tf.test.is_gpu_available():
        #     self.gru = tf.keras.layers.CuDNNGRU(self.units,
        #                                         return_sequences=True,
        #                                         recurrent_initializer='glorot_uniform',
        #                                         stateful=True)
        # else:
        self.lstm = tf.keras.layers.LSTM(self.units,return_sequences=True,dropout=0.5,stateful=False)
            # self.gru = tf.keras.layers.GRU(self.units,
            #                                return_sequences=True,
            #                                recurrent_activation='sigmoid',
            #                                recurrent_initializer='glorot_uniform',
            #                                stateful=True)

        self.fc = tf.keras.layers.Dense(2500,activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.fc2 = tf.keras.layers.Dense(600,activation="relu")
        self.fc3 = tf.keras.layers.Dense(600,activation="relu")
        self.fc4 = tf.keras.layers.Dense(out_size)

    def call(self, x):
        # output at every time step
        # output shape == (batch_size, seq_length, hidden_size)
        x_bn = self.bn(x)
        output = self.lstm(x_bn)
        # output = self.gru(x)

        # The dense layer will output predictions for every time_steps(seq_length)
        # output shape after the dense layer == (seq_length * batch_size, vocab_size)
        output = tf.reshape(output,[output.shape[0],-1])

        output = self.fc(output)
        # output = self.bn2(output)
        output = self.fc2(output)
        output = self.dropout(output)
        # output = self.bn3(output)
        output = self.fc3(output)
        prediction = self.fc4(output)

        # states will be used to pass at every step to the model while training
        return prediction

def loss_function(real, preds):
    # y_error = tf.subtract(real,preds)
    # return tf.nn.l2_loss(y_error)
    return tf.losses.absolute_difference(real,preds)
    # return tf.losses.mean_squared_error(real,preds)

print("version of tensorflow:",tf.__version__)

tf.enable_eager_execution()

tms = utils.buildAllTMToATensor(xmlPath = "../traffic-matrices/")
data_loader = dataloader(tms,EMA=True,alpha=0.9)

model = Model(out_size=23*23, units=23*23)
# Using adam optimizer with default arguments
lr = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
BATCH_SIZE = 32
seq_length = 20

ds_list=[0,1,1,1,1,1,1,1,1,1]
model.build(tf.TensorShape([BATCH_SIZE, len(ds_list),23*23]))
model.summary()
# Directory where the checkpoints will be saved
checkpoint_path = './training_checkpoints.ckpt'


loss_record = []
fw = open("../output_data/loss.txt","a+")
nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
fw.write("\n\n******"+nowTime+" new train record begin, lr = {}*******\n".format(lr))
fw.flush()
for epoch in range(200):
    start = time.time()
    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    if epoch== 0:
        loss_sum,l_n = 0,0
        generator_test = data_loader.iter_for_test(ds_list,batchsize=100,target_pos=1)
        for i, (batch_x_t, batch_y_t) in enumerate(generator_test):
            print("\r{}".format(i),end="")
            # batch_x = np.transpose(batch_x,axes=[1,0,2,3])
            b_t, l_t, m_t, n_t = batch_x_t.shape
            # print("b l m n:",b_t,l_t,m_t,n_t)
            by, _, _ = batch_y_t.shape
            batch_x_t = np.reshape(batch_x_t, [b_t, l_t, m_t * n_t])
            batch_y_t = np.reshape(batch_y_t, [by, -1])
            batch_x_t = tf.convert_to_tensor(batch_x_t, dtype=tf.float32)
            pred = model(batch_x_t)
            loss_sum = loss_sum+loss_function(batch_y_t, pred)
            # print("loss ",loss_sum)
            l_n = l_n+1
        print("\n")
        print("l_n is:",l_n)
        # print("loss_sum is:",loss_sum)
        print("loss of test set is:{:.4f}".format(loss_sum / l_n ))
        print("*"*70)
        loss_record.append("{:.4f}".format(loss_sum / l_n))
        fw.write("epoch {}: -1 {:.4f}\n".format(0,loss_sum / l_n))
        fw.flush()

    # hidden = model.reset_states()
    generator_train = data_loader.iter_for_train(ds_list=ds_list,batchsize=BATCH_SIZE,target_pos=1)
    print("*"*30+"[epoch {}]".format(epoch+1)+"*"*30)
    l_sum_train,n_cnt = 0,0
    for i,(batch_x,batch_y) in enumerate(generator_train):

        # batch_x = np.transpose(batch_x,axes=[1,0,2,3])
        b, l, m, n = batch_x.shape
        by, _, _ = batch_y.shape
        batch_x = np.reshape(batch_x,[b,l,m*n])
        batch_y = np.reshape(batch_y,[by,-1])
        batch_x = tf.convert_to_tensor(batch_x,dtype=tf.float32)
        with tf.GradientTape() as tape:

            predictions = model(batch_x)

            # loss = tf.reduce_sum(tf.abs(batch_y - predictions)) / b
            loss = loss_function(batch_y, predictions)
            l_sum_train = l_sum_train + loss
            n_cnt +=1
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))
        # optimizer.minimize(loss)
        if i % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, i,loss))
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_path)
    if (epoch+1) % 1 == 0:
        loss_sum,l_n = 0,0
        generator_test = data_loader.iter_for_test(ds_list,batchsize=100,target_pos=1)
        for i, (batch_x_t, batch_y_t) in enumerate(generator_test):
            print("\r{}".format(i),end="")
            # batch_x = np.transpose(batch_x,axes=[1,0,2,3])
            b_t, l_t, m_t, n_t = batch_x_t.shape
            # print("b l m n:",b_t,l_t,m_t,n_t)
            by, _, _ = batch_y_t.shape
            batch_x_t = np.reshape(batch_x_t, [b_t, l_t, m_t * n_t])
            batch_y_t = np.reshape(batch_y_t, [by, -1])
            batch_x_t = tf.convert_to_tensor(batch_x_t, dtype=tf.float32)
            pred = model(batch_x_t)
            loss_sum = loss_sum+loss_function(batch_y_t, pred)
            # print("loss ",loss_sum)
            l_n = l_n+1
        print("\n")
        print("l_n is:",l_n)
        # print("loss_sum is:",loss_sum)
        print("loss of test set is:{:.4f}".format(loss_sum / l_n ))
        print("*"*70)
        loss_record.append("{:.4f}".format(loss_sum / l_n))
        fw.write("epoch {}: {:4f} {:.4f}\n".format(epoch+1,l_sum_train/n_cnt,loss_sum / l_n))
        fw.flush()
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
fw.close()
print("ok！")
