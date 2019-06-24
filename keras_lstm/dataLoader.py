
import numpy as np
import random
import myema as ema
import utils
random.seed(2019)

class dataloader():
    def __init__(self,inputData,ratio_for_train = 0.6,EMA=False,alpha=0.5,BOX=True,normalization=1):
        self.totalData = inputData
        self.train_ratio = ratio_for_train
        self.ema = EMA
        self.alpha = alpha
        self.box = BOX
        self.normalization=normalization
        self._build_train_test_tensor()
    def _build_train_test_tensor(self):
        l,m,n = self.totalData.shape
        train_end = int(l*self.train_ratio)
        test_start = train_end

        if self.box:
            try:
                self.totalData = np.fromfile("../noOutliers.bin",dtype=np.float32).reshape([l,m,n])
            except:
                self.tmp_tensor = np.zeros_like(self.totalData,np.float32)
                L, M, N = self.totalData.shape
                print("remove outliers", L, M, N)
                for i in range(M):
                    for j in range(N):
                        print("\r {}/{}  {}/{}".format(i+1,M,j+1,N))
                        single_ch = self.totalData[:, i, j].tolist()
                        single_ch = utils.remove_outliers(single_ch)
                        # single_ch = utils.remove_outliers(single_ch)
                        self.tmp_tensor[:, i, j] = np.array(single_ch)
                self.totalData = self.tmp_tensor
                self.totalData.tofile("../noOutliers.bin")
        if self.ema:
            self.tmp_tensor = np.zeros_like(self.totalData)
            L,M,N = self.totalData.shape
            print("do ema ",L,M,N)
            for i in range(M):
                for j in range(N):
                    single_ch = self.totalData[:,i,j].tolist()
                    single_ch = ema.compute_single(alpha=self.alpha,s=single_ch)
                    self.tmp_tensor[:,i,j] = np.array(single_ch)
            self.totalData = self.tmp_tensor


        self.train_tensor = self.totalData[:train_end, :, :]
        self.test_tensor = self.totalData[test_start:, :, :]
        self.mean_train_x = np.mean(self.totalData,axis=0)
        self.std_train_x = np.std(self.totalData,axis=0)
        self.max_train_x = np.max(self.totalData,axis=0)

        print("mean tensor of train x is:",self.mean_train_x)
        print("std tensor of train x is:",self.std_train_x)


    def iter_for_train(self,ds_list, batchsize = 6, target_pos = 1):
        l, _, _ = self.train_tensor.shape
        block_size = max(ds_list) + target_pos
        indexes_list = list(range(l-block_size+1))
        # random.shuffle(indexes_list)

        for i in range(0,len(indexes_list),batchsize):
            batch_indexes = indexes_list[i:i+batchsize]
            batch_x = np.zeros(shape=[batchsize,len(ds_list),23,23])
            batch_y = np.zeros(shape=[batchsize,23,23])
            for cnt, j in enumerate(batch_indexes):
                start_pos = j
                absolute_pos_list = [ii+start_pos for ii in ds_list]
                single_x = self.train_tensor[absolute_pos_list,:,:]    # 取单个输入x
                single_y = self.train_tensor[start_pos+max(ds_list)+target_pos-1,:,:]  # 取单个标签y
                # print("*" * 70)
                #
                if self.normalization == 0:
                    single_x = (single_x - self.mean_train_x) / (self.std_train_x+1e-8)
                    single_y = (single_y - self.mean_train_x) / (self.std_train_x+1e-8)
                if self.normalization == 1:
                    single_x = single_x  / (self.max_train_x + 1e-8)
                    single_y = single_y  / (self.max_train_x + 1e-8)

                batch_x[cnt] = single_x     # 存入到batch_x 的对应位置
                batch_y[cnt] = single_y     # 存入到batch_y 的对应位置


            yield batch_x,batch_y

    def iter_for_test(self,ds_list, batchsize = 1, target_pos = 1):
        l, _, _ = self.test_tensor.shape
        block_size = max(ds_list) + target_pos
        indexes_list = list(range(l - block_size + 1))

        for i in range(0, len(indexes_list), batchsize):
            batch_indexes = indexes_list[i:i + batchsize]
            batch_x = np.zeros(shape=[batchsize, len(ds_list), 23, 23])
            batch_y = np.zeros(shape=[batchsize, 23, 23])
            for cnt, j in enumerate(batch_indexes):
                start_pos = j
                absolute_pos_list = [ii + start_pos for ii in ds_list]
                single_x = self.test_tensor[absolute_pos_list, :, :]
                single_y = self.test_tensor[start_pos + max(ds_list) + target_pos - 1, :, :]  # 取单个标签y

                if self.normalization == 0:
                    single_x = (single_x - self.mean_train_x) / (self.std_train_x+1e-8)
                    single_y = (single_y - self.mean_train_x) / (self.std_train_x+1e-8)
                if self.normalization == 1:
                    single_x = single_x  / (self.max_train_x + 1e-8)
                    single_y = single_y  / (self.max_train_x + 1e-8)


                batch_x[cnt] = single_x
                batch_y[cnt] = single_y

            yield batch_x,batch_y

if __name__=="__main__":
    tms = utils.buildAllTMToATensor(xmlPath="../traffic-matrices/")
    aa = dataloader(tms,0.75,EMA=False)
    # print(aa.max_train_x)
    save_list = aa.max_train_x.tolist()
    save_list2 = aa.mean_train_x.tolist()

    # print(save_list)
    fw = open("../output_data/max_value.json","w")
    import json
    json.dump({"max_v":save_list,"mean_v":save_list2},fw)

    fw.close()
