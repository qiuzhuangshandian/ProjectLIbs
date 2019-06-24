
# from src_new.create_dataset import Data_utils
from src_new.reader_new import Data_utils

from src_new.model import Encoder,Decoder,train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datafile0",default="/home/hai/PythonProject/gas/level_and_class/MixtureGas326/data/ethylene_CO-1.txt")
parser.add_argument("--datafile1",default="/home/hai/PythonProject/gas/level_and_class/MixtureGas326/data/ethylene_methane-1.txt")
parser.add_argument("--num_hiddens_en",default=50,type=int)
parser.add_argument("--num_hiddens_de",default=50,type=int)
parser.add_argument("--num_layers_en",default=1,type=int)
parser.add_argument("--num_layers_de",default=1,type=int)
parser.add_argument("--enOutsize",default=20,type=int)
parser.add_argument("--outSize",default=5,type=int)

parser.add_argument("--attention_size",default=10,type=int)
parser.add_argument("--drop_prob",default=0.5,type=float)
parser.add_argument("--num_step",default=100,type=int)
parser.add_argument("--batch_size",default=100,type=int)
parser.add_argument("--lr",default=0.01,type=float)
parser.add_argument("--momentum",default=0.9,type=float)
parser.add_argument("--wd",default=1e-5,type=float)
parser.add_argument("--clip_gradient",default=0.01,type=float)

parser.add_argument("--cl_w",default=1.0,type=float)
parser.add_argument("--score_w",default=10.0,type=float)

parser.add_argument("--num_epochs",default=200,type=int)
parser.add_argument("--time_scale",default=10,type=int)
parser.add_argument("--train_scale",default=0.6,type=int)

args = parser.parse_args()


if __name__=="__main__":
    encoder = Encoder(num_hiddens= args.num_hiddens_en,
                      num_layers=args.num_layers_en ,
                      drop_prob=args.drop_prob,enOutsize=args.enOutsize)
    decoder = Decoder(outSize=args.outSize , num_hiddens=args.num_hiddens_de ,
                      num_layers=args.num_layers_de ,attention_size=args.attention_size ,drop_prob=0)
    Gas_data = Data_utils(filename0=args.datafile0,filename1=args.datafile1,num_step=args.num_step ,
                          time_scale=args.time_scale,train_scale=args.train_scale )

    param_dict = {"lr":args.lr,
                  "momentum":args.momentum,
                  "wd":args.wd,
                  "clip_gradient":args.clip_gradient
                  }
    train(encoder,decoder,data_utils=Gas_data,param_dict = param_dict,
          batch_size=args.batch_size,num_epochs=args.num_epochs,
          cl_w=args.cl_w,score_w=args.score_w)


