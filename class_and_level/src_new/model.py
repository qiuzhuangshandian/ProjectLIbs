from mxnet import nd,init,gluon,autograd
from mxnet.gluon import nn,loss as gloss,rnn
import csv
param_enc = "enc.params"
param_dec = "dec.params"


# def grad_clipping(params, theta):
#     """Clip the gradient."""
#     if theta is not None:
#         norm = nd.array([0.0])
#         for param in params:
#             norm += (param.grad ** 2).sum()
#         norm = norm.sqrt().asscalar()
#         if norm > theta:
#             for param in params:
#                 param.grad[:] *= theta / norm

class Encoder(nn.Block):
    def __init__(self,  num_hiddens, num_layers,
                 drop_prob=0,enOutsize=2, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        # self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=drop_prob)
        self.dense = gluon.nn.Sequential()
        self.dense.add(
            nn.Dense(enOutsize,use_bias=True, flatten=False)
        )
        # self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        # self.out = nn.Dense(2)

    def forward(self, inputs, state):
        # 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
        # embedding = self.embedding(inputs).swapaxes(0, 1)
        # self.rnn(inputs, state)
        rnn_out,rnn_state = self.rnn(inputs, state)
        
        # self.dense(rnn_out)
        return self.dense(rnn_out),rnn_state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
def attention_model(attention_size):
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation='tanh', use_bias=False,
                       flatten=False),
              nn.Dense(1, use_bias=False, flatten=False))
    return model
def attention_forward(model, enc_states):
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
    # dec_states = nd.broadcast_axis(
    #     dec_state.expand_dims(0), axis=0, size=enc_states.shape[0])
    # enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)
    enc_and_dec_states = nd.array(enc_states)
    # print("shape of enc_states:",enc_states.shape)
    e = model(enc_and_dec_states)  # 形状为(时间步数, 批量大小, 1)
    alpha = nd.softmax(e, axis=0)  # 在时间步维度做softmax运算
    # print("shape of alpha:", alpha.shape)
    out = (alpha * enc_states).sum(axis=0)
    # print("shape of out:", out.shape)
    return  out # 返回背景变量

def fully_connect(num_hiddens_fully,outSize):
    model = nn.Sequential()
    model.add(nn.Dense(num_hiddens_fully, activation='tanh', use_bias=False,
                       flatten=False),
              nn.Dense(outSize, use_bias=False, flatten=False))
    return model
# def fully_forward(model,enc_states):
#
#     e = model()
#
#     return e
class Decoder(nn.Block): 
    def __init__(self,outSize, num_hiddens, num_layers,
                 attention_size, drop_prob=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(attention_size)
        # self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        # self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=drop_prob)
        self.fully_connect = fully_connect(50,outSize)
        # self.out = nn.Dense(outSize, flatten=False)

    def forward(self, state, enc_states):
        # 使用注意力机制计算背景向量
        c = attention_forward(self.attention, enc_states)
        # print(type(c),type(state),type(enc_states))
        # print(c.shape,state[0].shape,enc_states.shape)
        # 将嵌入后的输入和背景向量在特征维连结
        # input_and_c = nd.concat(self.embedding(cur_input), c, dim=1)
        input_and_c = nd.concat(c, state[0][0], dim=1)
        # print("shape of input_and_c:",input_and_c.shape)
        # 为输入和背景向量的连结增加时间步维，时间步个数为1
        # input_and_c = cur_input
        # output, state = self.rnn(input_and_c.expand_dims(0), state)
        # output, state = self.rnn(input_and_c, state)
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        # output = self.out(output).squeeze(axis=0)
        output = self.fully_connect(input_and_c)

        return output

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state

def class_and_score_forward(x):
    class_part = nd.slice_axis(x,begin=0,end=3,axis=-1)
    concentration_part = nd.slice_axis(x,begin=3,end=5,axis=-1)

    class_part = nd.sigmoid(class_part)
    concentration_part = nd.sigmoid(concentration_part)
    return class_part,concentration_part

def batch_loss(encoder, decoder, X, Y, loss_l2,loss_ce,cl_w,score_w):
    batch_size = X.shape[1]
    # print(X[0].shape)

    enc_state = encoder.begin_state(batch_size=batch_size)
    enc_outputs, enc_state = encoder(X, enc_state)

    # 初始化解码器的隐藏状态
    dec_state = decoder.begin_state(enc_state)

    dec_output = decoder(dec_state, enc_outputs)

    cl_res,score_res = class_and_score_forward(dec_output)

    # print("shape of cl_res:",cl_res.shape,"shape of Y[0][:,:3]:",Y.shape)
    # print("shape of score_res:",score_res.shape,"shape of Y[0][:,3:]:",Y.shape)
    cl_weight,score_weight = nd.ones_like(cl_res)*cl_w,nd.ones_like(score_res)*score_w
    l_ce = loss_ce(cl_res, Y[:,:3],cl_weight).sum()
    l_l2 = loss_l2(score_res, Y[:,3:],score_weight).sum()


    n = Y.shape[0]

    return l_ce,l_l2, n,cl_res,score_res
def get_accuracy(pre_l,true_l):
    one_zero_pre = nd.where(pre_l>0.5,nd.ones_like(pre_l),nd.zeros_like(pre_l))
    compare = nd.equal(one_zero_pre,true_l).sum(axis=1)
    samples_right = nd.where(compare==3,nd.ones_like(compare),nd.zeros_like(compare)).sum()
    all_num = pre_l.shape[0]
    return samples_right / all_num

def get_rmse(class_pre_l,class_true_l,con_pre_l,con_true_l,data_utils):
    # find right predictions
    one_zero_pre = nd.where(class_pre_l > 0.5, nd.ones_like(class_pre_l), nd.zeros_like(class_pre_l))
    compare = nd.equal(one_zero_pre, class_true_l).sum(axis=1)
    weight_right = nd.repeat(nd.expand_dims(nd.where(compare == 3, nd.ones_like(compare), nd.zeros_like(compare)),\
                                            axis=0),repeats=2,axis=0).transpose()

    # calculate rmse based on right prediction
    eth_co_me_limit = nd.array([[data_utils.scale_CO[1],data_utils.scale_CO[0],data_utils.scale_Me[0]]])
    concentration_mat = nd.where(class_pre_l > 0.5, nd.repeat(eth_co_me_limit,repeats=class_pre_l.shape[0],axis=0), \
                            nd.zeros_like(class_pre_l))
    eth_pre_con,eth_pre_con_true = concentration_mat[:,0]*con_pre_l[:,1],concentration_mat[:,0]*con_true_l[:,1]
    co_pre_con,co_pre_con_true = concentration_mat[:,1]*con_pre_l[:,0],concentration_mat[:,1]*con_true_l[:,0]
    me_pre_con,me_pre_con_true = concentration_mat[:,2]*con_pre_l[:,0],concentration_mat[:,2]*con_true_l[:,0]
    co_or_me_con,co_or_me_con_true = co_pre_con + me_pre_con, co_pre_con_true+me_pre_con_true

    co_or_me_eth_con = nd.concat(nd.expand_dims(co_or_me_con,axis=0),nd.expand_dims(eth_pre_con,axis=0),dim=0).transpose()
    co_or_me_eth_con_true = nd.concat(nd.expand_dims(co_or_me_con_true,axis=0),nd.expand_dims(eth_pre_con_true,axis=0),dim=0).transpose()

    # rmse = (((co_or_me_eth_con-co_or_me_eth_con_true)**2*weight_right).sum()/(weight_right[:,0].sum()))**(0.5)
    rmse = (((co_or_me_eth_con-co_or_me_eth_con_true)**2).mean(axis=0))

    return rmse


def concentration_transfer(class_pre_l,class_true_l,con_pre_l,con_true_l,data_utils):
    eth_co_me_limit = nd.array([[data_utils.scale_CO[1], data_utils.scale_CO[0], data_utils.scale_Me[0]]])
    concentration_mat_pre = nd.where(class_pre_l > 0.5,
                                     nd.repeat(eth_co_me_limit, repeats=class_pre_l.shape[0], axis=0), \
                                     nd.zeros_like(class_pre_l))
    concentration_mat_true = nd.where(class_true_l == 1,
                                      nd.repeat(eth_co_me_limit, repeats=class_true_l.shape[0], axis=0), \
                                      nd.zeros_like(class_true_l))

    eth_con_pre, eth_con_true = concentration_mat_pre[:, 0] * con_pre_l[:, 1], concentration_mat_true[:,
                                                                               0] * con_true_l[:, 1]
    co_con_pre, co_con_true = concentration_mat_pre[:, 1] * con_pre_l[:, 0], concentration_mat_true[:, 1] * con_true_l[
                                                                                                            :, 0]
    me_con_pre, me_con_true = concentration_mat_pre[:, 2] * con_pre_l[:, 0], concentration_mat_true[:, 2] * con_true_l[
                                                                                                            :, 0]

    eth_co_me_con_pre = nd.concat(nd.expand_dims(eth_con_pre, axis=0), nd.expand_dims(co_con_pre, axis=0), \
                                  nd.expand_dims(me_con_pre, axis=0), dim=0).transpose()
    eth_co_me_con_true = nd.concat(nd.expand_dims(eth_con_true, axis=0), nd.expand_dims(co_con_true, axis=0), \
                                   nd.expand_dims(me_con_true, axis=0), dim=0).transpose()
    return eth_co_me_con_pre,eth_co_me_con_true


def get_rmse_v2(eth_co_me_con_pre, eth_co_me_con_true):
    # eth_co_me_con_pre, eth_co_me_con_true = concentration_transfer(class_pre_l,class_true_l,con_pre_l,con_true_l,data_utils)
    rmse = (((eth_co_me_con_pre - eth_co_me_con_true) ** 2).mean(axis=0))
    return rmse

def results_writer(fw,class_pres,class_trues,concentration_pres,concentration_trues):

    all_class_pres_list = class_pres.asnumpy().tolist()
    all_class_trues_list = class_trues.asnumpy().tolist()
    all_con_pres_list = concentration_pres.asnumpy().tolist()
    all_con_trues_list = concentration_trues.asnumpy().tolist()

    csv_writer = csv.writer(fw, dialect='excel')
    for i in range(class_pres.shape[0]):
        content = all_class_pres_list[i] + ["   "] + \
                  all_class_trues_list[i] + ["  | "] + \
                  all_con_pres_list[i] + ["  "] + \
                  all_con_trues_list[i]
        csv_writer.writerow(content)


def train(encoder, decoder, data_utils, param_dict, batch_size, num_epochs,cl_w,score_w):
    
    encoder.initialize(init.Xavier(), force_reinit=True)
    decoder.initialize(init.Xavier(), force_reinit=True)
    params_enc = encoder.collect_params()
    params_dec = decoder.collect_params()
   

    enc_trainer = gluon.Trainer(params_enc, 'sgd', 
                    {'learning_rate': param_dict["lr"], "momentum":param_dict["momentum"],
                     'wd': param_dict["wd"],'clip_gradient': param_dict["clip_gradient"]})
    dec_trainer = gluon.Trainer(params_dec, 'sgd',
                                {'learning_rate': param_dict["lr"], "momentum": param_dict["momentum"],
                                 'wd': param_dict["wd"], 'clip_gradient': param_dict["clip_gradient"]})
    loss1 = gloss.L2Loss()
    loss2 = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    pre_rmse = 100000
    print("train begin......")

    for epoch in range(num_epochs):
        print("*" * 70)
        l_sum,l_class_sum,l_score_sum,n,acc_sum = 0.0,0.0,0.0,0,0.0
        rmse_sum = nd.array([0,0,0])
        data_iter = data_utils.get_batch_train(batch_size)
        if (epoch +1) > 10 and enc_trainer.learning_rate > 0.001:
            lr_enc_trainer = enc_trainer.learning_rate* 0.99
            lr_dec_trainer = dec_trainer.learning_rate* 0.98
            enc_trainer.set_learning_rate(lr_enc_trainer)
            dec_trainer.set_learning_rate(lr_dec_trainer)

        for i, (X, Y) in enumerate(data_iter):

            X ,Y= nd.array(X).transpose([1,0,2]),nd.array(Y)
            # print(X.shape,Y.shape)
            with autograd.record():
                # print("here:",X.shape,Y.shape)
                l_class,l_score,b ,cl_res_mini,score_res_mini= batch_loss(encoder, decoder, X, Y, loss1, loss2,cl_w,score_w)
                l = (l_class/b)+(l_score/b)
            l.backward()

            enc_trainer.step(1)
            dec_trainer.step(1)

            l_sum += l.asscalar()
            l_class_sum+=(l_class/b).asscalar()
            l_score_sum+=(l_score/b).asscalar()
            tmp_mini_acc = get_accuracy(cl_res_mini, Y[:, :3]).asscalar()
            # tmp_mini_rmse = get_rmse(cl_res_mini,Y[:, :3],score_res_mini,Y[:,3:],data_utils)
            pre_and_true_concentration = concentration_transfer(cl_res_mini,Y[:, :3],score_res_mini,Y[:,3:],data_utils)
            tmp_mini_rmse = get_rmse_v2(pre_and_true_concentration[0],pre_and_true_concentration[1])
            acc_sum += tmp_mini_acc
            rmse_sum = rmse_sum+ tmp_mini_rmse
            n += 1

            if (i+1)%1000 ==0 or (i==0 and epoch==0):
                c1,c2,c3 = ((rmse_sum/n)**0.5)[0].asscalar(),((rmse_sum/n)**0.5)[1].asscalar(),((rmse_sum/n)**0.5)[2].asscalar()
                print("total loss: {:.6}".format(l_sum/n),",loss class:{:.6}".format(l_class_sum/n),\
                      ",loss concentration:{:.6}".format(l_score_sum/n),",[train acc:{:.6}]".format(acc_sum/n),\
                      "[Eth Co Me rmse:[{:.6},{:.6},{:.6}]]".format(c1,c2,c3),
                      "lr is:{:.6}".format(enc_trainer.learning_rate))

        #for test loss
        n_test = 0
        l_test = nd.array([0])
        test_data = data_utils.get_batch_test(batch_size=100)

        cl_pres,cl_trues = [],[]
        con_pres,con_trues = [],[]
        for x,y in test_data:
            x,y = nd.array(x).transpose([1,0,2]),nd.array(y)

            batchsize_t = x.shape[1]

            enc_state = encoder.begin_state(batch_size=batchsize_t)
            enc_outputs, enc_state = encoder(x, enc_state)

            dec_state = decoder.begin_state(enc_state)

            dec_output = decoder(dec_state, enc_outputs)

            cl_res, score_res = class_and_score_forward(dec_output)
            cl_pres.append(cl_res)
            cl_trues.append(y[:,:3])
            con_pres.append(score_res)
            con_trues.append(y[:,3:])

            l_ce = loss2(cl_res, y[:, :3]).sum()
            l_l2 = loss1(score_res, y[:, 3:]).sum()

            l_total = l_ce + l_l2

            l_test +=l_total
            n_test += y.shape[0]



        all_class_pres = nd.concat(*cl_pres,dim=0)
        all_class_trues = nd.concat(*cl_trues,dim=0)
        acc = get_accuracy(all_class_pres,all_class_trues).asscalar()

        all_con_pres = nd.concat(*con_pres,dim=0)
        all_con_trues = nd.concat(*con_trues,dim=0)
        # test_rmse = get_rmse(all_class_pres, all_class_trues, all_con_pres, all_con_trues, data_utils)
        test_pre_and_true_concentration = concentration_transfer(all_class_pres, all_class_trues, all_con_pres, all_con_trues, data_utils)
        test_rmse = get_rmse_v2(test_pre_and_true_concentration[0],test_pre_and_true_concentration[1])

        f = open("record.csv", "w", newline='')
        results_writer(f,all_class_pres,all_class_trues,\
                       test_pre_and_true_concentration[0],test_pre_and_true_concentration[1])


        if (epoch + 1) % 1 == 0:
            c1,c2,c3 = (test_rmse**0.5)[0].asscalar(),(test_rmse**0.5)[1].asscalar(),(test_rmse**0.5)[2].asscalar()
            print("epoch %d, train_loss %.3f,test_loss: %.3f, " % (epoch + 1, l_sum/n,l_test.asscalar()/n_test)+\
            "[test Eth Co Me rmse:[{:.6},{:.6},{:.6}]]".format(c1,c2,c3),"[train acc:{:.6}]".format(acc_sum/n),\
                  "[test acc:{:.6}]".format(acc))
        # if test_rmse < pre_rmse:
        encoder.save_parameters(param_enc)
        decoder.save_parameters(param_dec)

        # pre_rmse = test_rmse
        print("params updated!")
        print("*"*70)
        f.close()
            
if __name__ == "__main__":
    #def get_rmse(class_pre_l, class_true_l, con_pre_l, con_true_l, data_utils):
    # a = nd.array([[0.2,0.6,0.9],[0.9,0.1,0.9]])
    # b = nd.array([[0,0,1],[1,0,1]])
    # c = nd.array()
    pass