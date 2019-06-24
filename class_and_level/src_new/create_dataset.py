import numpy as np
import pandas as pd
import random

class Data_utils():

    def __init__(self, filename, num_step, time_scale=10, train_scale=0.6):
        self.num_step = num_step
        print('loading data set...')
        self.rawdata, self.label = self._load(filename, time_scale)
        print('loading complete!')
        self.n, self.m = self.rawdata.shape
        print("time points is %d, sensor nums is %d"%(self.n, self.m))
        self.out_dim0, self.out_dim1 = self.label.shape[1], 4
        self._split(int(self.n*train_scale))
        print("train point num is %d"%int(self.n*train_scale))

    def correctConcentrations(self, conc_ideal):
        '''
        Returns the actual concentrations given ideal concentrations by accounting
        for delays and time flows.
        矫正浓度延时
        '''
        conc_real = np.zeros(conc_ideal.shape)
        # Delay in flow
        flow = 100  # cm3/min
        diameter = (3. / 16) * 2.54  # cm
        length = 2.5 * 100  # cm
        volume = length * np.pi * (diameter ** 2 / 4.)
        delay = volume / flow * 60  # seconds
        print("Delay: {0}sec".format(delay))
        freq = 100  # Hz
        t_step_offset = int(np.round(delay * freq))
        # CO/Methane have a higher flow rate 200 cm3/min
        conc_real[int(np.round(t_step_offset / 1.5)):, 0] = conc_ideal[:-int(np.round(t_step_offset / 1.5)), 0]
        # Ethylene has a lower flow rate 100 cm3/min
        conc_real[t_step_offset:, 1] = conc_ideal[:-t_step_offset, 1]  # Ethylene
        # conc_real = conc_ideal
        return conc_real


    def _load(self, file, time_scale):
        X = pd.read_table(file, sep='\s+')

        targets = X.values[:, 1:3]
        datas = self._normalized(2, X.values[:, 3:])
        targets = self.correctConcentrations(targets)
        targets = targets[10000::time_scale]
        datas = datas[10000::time_scale] # down sample 10 Hz
        return datas, targets

    def _normalized(self, normalized, rawdata):

        if normalized == 0:
            dat = rawdata

        if normalized == 1:
            mean = np.mean(rawdata, axis=0)
            std = np.std(rawdata, axis=0)
            dat = (rawdata - mean) / std

        if normalized == 2:
            self.scale = np.max(rawdata, axis=0)
            dat = rawdata / np.max(rawdata, axis=0)
        return dat

    def _split(self, train):
        train_set = range(self.num_step-1, train)
        test_set = range(train, self.n)
        self.train= self._batchify(train_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):
        """
        :param idx_set:输入的数据集
        :return: 划分时间步长的数据和对应步长最后的标签
        """
        n = len(idx_set)
        X = np.zeros((n, self.num_step, self.m))
        Y = np.zeros((n, self.out_dim0))

        for i in range(n):
            end = idx_set[i] + 1
            start = end - self.num_step
            X[i, :, :] = self.rawdata[start:end, :]
            Y[i, :] = self.label[idx_set[i], :]
        Y_label = self._define_lable(Y, n)
        Y = np.concatenate([Y_label, Y], axis=1)
        return [X,Y]

    def _define_lable(self, idx, n):

        Y = np.zeros((n, self.out_dim1))
        for i in range(n):
            if idx[i][0] == 0.0 and idx[i][1] == 0.0:
                Y[i, :] = np.array([1.0,0.0,0.0,0.0])
            elif idx[i][0] != 0.0 and idx[i][1] == 0.0:
                Y[i, :] = np.array([0.0,1.0,0.0,0.0])
            elif idx[i][0] == 0.0 and idx[i][1] != 0.0:
                Y[i, :] = np.array([0.0,0.0,1.0,0.0])
            else:
                Y[i, :] = np.array([0.0, 0.0, 0.0, 1.0])
        return Y

    def get_batch_train(self,batch_size, shuffle=True):
        """
        输入整体的划分好数据集和标签集
        训练时产生的batch以随机的方式产生
        测试的时候以时间顺序的方式产生
        """
        length = len(self.train[0])
        index = np.arange(length)
        if shuffle:
            random.shuffle(index)
        start_idx = 0
        while(start_idx < length):
            end_idx = min(length, start_idx+batch_size)
            batch_ep = index[start_idx:end_idx]
            X = self.train[0][batch_ep]
            y = self.train[1][batch_ep]
            y = y[np.newaxis, :, :]
            yield (X, y)
            start_idx+=batch_size
    def get_batch_test(self,batch_size=1, shuffle=False):
        """
        输入整体的划分好数据集和标签集
        训练时产生的batch以随机的方式产生
        测试的时候以时间顺序的方式产生
        """
        length = len(self.test[0])
        index = np.arange(length)
        if shuffle:
            random.shuffle(index)
        start_idx = 0
        while(start_idx < length):
            end_idx = min(length, start_idx+batch_size)
            batch_ep = index[start_idx:end_idx]
            X = self.test[0][batch_ep]
            y = self.test[1][batch_ep]
            y = y[np.newaxis,:,:]
            yield (X, y)
            start_idx+=batch_size


if __name__ == '__main__':
    filename = '../data/ethylene_CO-1.txt'
    DU = Data_utils(filename, 64, 50)
    for i, (X, y) in enumerate(DU.get_batch_train(True)):
        pass
    print(X.shape, y)
