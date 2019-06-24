import numpy as np
import pandas as pd
import random
import os

class Data_utils():

    def __init__(self, filename0, filename1, num_step, time_scale=10, train_scale=0.6):
        # self.batch_size = batch_size
        self.num_step = num_step

        self.rawdata_CO, self.label_CO, self.scale_CO = self._load(filename0, time_scale)
        self.n_CO, self.m_CO = self.rawdata_CO.shape
        self.out_dim0, self.out_dim1 = self.label_CO.shape[1], 3
        train_CO_X, train_CO_y, test_CO_X, test_CO_y = self._split(int(self.n_CO*train_scale), self.n_CO,
                                                                   self.rawdata_CO, self.label_CO, self.m_CO)

        print("%s time points is %d, sensor nums is %d"% (self.file_, self.n_CO, self.m_CO))

        self.rawdata_Me, self.label_Me, self.scale_Me = self._load(filename1, time_scale)
        self.n_Me, self.m_Me = self.rawdata_Me.shape
        print("%s time points is %d, sensor nums is %d" % (self.file_, self.n_Me, self.m_Me))

        train_Me_X, train_Me_y,test_Me_X, test_Me_y = self._split(int(self.n_Me*train_scale), self.n_Me,
                                                                  self.rawdata_Me, self.label_Me, self.m_Me)
        self.train_X, self.train_y = np.concatenate([train_CO_X, train_Me_X], axis=0), \
                                     np.concatenate([train_CO_y, train_Me_y], axis=0)
        self.test_X, self.test_y = np.concatenate([test_CO_X, test_Me_X], axis=0), \
                                     np.concatenate([test_CO_y, test_Me_y], axis=0)

        print("train point num is %d"%(int(self.n_CO*train_scale) + int(self.n_Me*train_scale)))



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
        self.file_ = os.path.basename(file.split('/')[-1]).split('.')[0]
        if self.file_ == "ethylene_CO-1":
            self.IsCO = True
        else:
            self.IsCO = False
        X = pd.read_table(file, sep='\s+')
        print('%s loading data set...'% self.file_)
        targets = X.values[:, 1:3]
        datas, _ = self._normalized(2, X.values[:, 3:])
        targets = self.correctConcentrations(targets)
        targets, scale = self._normalized(2, targets)
        print("%s \'s max value is %s"%(self.file_, scale))
        targets = targets[10000::time_scale]
        datas = datas[10000::time_scale] # down sample 10 Hz
        print('%s loading complete!'%self.file_)
        return datas, targets, scale

    def _normalized(self, normalized, rawdata):

        if normalized == 0:
            dat = rawdata

        if normalized == 1:
            mean = np.mean(rawdata, axis=0)
            std = np.std(rawdata, axis=0)
            dat = (rawdata - mean) / std

        if normalized == 2:
            scale = np.max(rawdata, axis=0)
            dat = rawdata / np.max(rawdata, axis=0)
        return dat, scale

    def _split(self, train, n, rawdata, label, m):
        train_set = range(self.num_step-1, train)
        test_set = range(train, n)
        trs_X, trs_y = self._batchify(train_set, rawdata, label, m)
        tes_X, tes_y = self._batchify(test_set, rawdata, label, m)
        return trs_X, trs_y, tes_X, tes_y



    def _batchify(self, idx_set, rawdata, label, m):
        """
        :param idx_set:输入的数据集
        :return: 划分时间步长的数据和对应步长最后的标签
        """
        n = len(idx_set)
        X = np.zeros((n, self.num_step, m))
        Y = np.zeros((n, self.out_dim0))

        for i in range(n):
            end = idx_set[i] + 1
            start = end - self.num_step
            X[i, :, :] = rawdata[start:end, :]
            Y[i, :] = label[idx_set[i], :]
        if self.IsCO:
            # print("*" * 50, "isco", "*" * 50)
            Y_label = self._define_lable_CO(Y, n)
        else:
            Y_label = self._define_lable_Met(Y, n)
        Y = np.concatenate([Y_label, Y], axis=1)
        # print("*" * 50, "batch", "*" * 50)
        return [X,Y]

    def _define_lable_CO(self, idx, n):
        """

        :return:标签[Eth, CO, Met]
        """
        Y = np.zeros((n, self.out_dim1))
        for i in range(n):
            if idx[i][0] == 0.0 and idx[i][1] == 0.0:
                # print("*" * 50, "here 0", "*" * 50)
                Y[i, :] = np.array([0.0,0.0,0.0])
            elif idx[i][0] != 0.0 and idx[i][1] == 0.0:
                # print("*"*50,"here 1","*"*50)
                Y[i, :] = np.array([0.0,1.0,0.0])
            elif idx[i][0] == 0.0 and idx[i][1] != 0.0:
                # print("*" * 50, "here 2", "*" * 50)
                Y[i, :] = np.array([1.0,0.0,0.0])
            else:
                # print("*" * 50, "here 3", "*" * 50)
                Y[i, :] = np.array([1.0, 1.0, 0.0])
        return Y

    def _define_lable_Met(self, idx, n):
        """

        :return:标签[Eth, CO, Met]
        """
        Y = np.zeros((n, self.out_dim1))
        for i in range(n):
            if idx[i][0] == 0.0 and idx[i][1] == 0.0:
                # print("*" * 50, "here 3", "*" * 50)
                Y[i, :] = np.array([0.0,0.0,0.0])
            elif idx[i][0] != 0.0 and idx[i][1] == 0.0:
                Y[i, :] = np.array([0.0,0.0,1.0])
            elif idx[i][0] == 0.0 and idx[i][1] != 0.0:
                Y[i, :] = np.array([1.0,0.0,0.0])
            else:
                Y[i, :] = np.array([1.0, 0.0, 1.0])
        return Y


    def get_batch_train(self, batch_size,shuffle=True):
        """
        输入整体的划分好数据集和标签集
        训练时产生的batch以随机的方式产生
        测试的时候以时间顺序的方式产生
        """
        length = len(self.train_X)
        index = np.arange(length)
        if shuffle:
            random.shuffle(index)
        start_idx = 0
        while(start_idx < length):
            end_idx = min(length, start_idx+batch_size)
            batch_ep = index[start_idx:end_idx]
            X = self.train_X[batch_ep]
            y = self.train_y[batch_ep]
            yield (X, y)
            start_idx+=batch_size

    def get_batch_test(self, batch_size ,shuffle=False):
        """
        输入整体的划分好数据集和标签集
        训练时产生的batch以随机的方式产生
        测试的时候以时间顺序的方式产生
        """
        length = len(self.test_X)
        index = np.arange(length)
        if shuffle:
            random.shuffle(index)
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            batch_ep = index[start_idx:end_idx]
            X = self.test_X[batch_ep]
            y = self.test_y[batch_ep]
            yield (X, y)
            start_idx += batch_size


if __name__ == '__main__':
    filename0 = '../data/ethylene_CO-1.txt'
    filename1 = '../data/ethylene_methane-1.txt'
    DU = Data_utils(filename0, filename1, 64, 50, train_scale=0.6)
    for i, (X, y) in enumerate(DU.get_batch_train(True)):
        pass
    print(i, X.shape, y)
    # print(DU.scale_CO)
    # print(DU.scale_Me)