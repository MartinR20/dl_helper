from dl_helper import *

obj = dataObj('..\\data\\prices-split-adjusted.csv', 10, 10, 10)

print(f"ValidateLSTM:{ValidateLSTM(4, 200, 2, 4, cutoff=9)(obj.x_train[obj.companys['AMZN']]).size()}")
print(f"ValidateLSTMQuantile:{ValidateLSTMQuantile([0.5,0.9,1], 4, 200, 2, 4, cutoff=9)(obj.x_train[obj.companys['AMZN']]).size()}")

obj.reshape()

print(f"LSTMAE:{LSTMAE(2004, 512, 1, False, cutoff=8)(obj.x_train).size()}")

## not enough ram to test these on my laptop with original data (they expect a sequence length of 100)
import torch
print(f"ConvLSTMAE_v0:{ConvLSTMAE_v0(cutoff=99)(torch.randn((5,100,2004))).size()}")
print(f"ConvLSTMAE_v1:{ConvLSTMAE_v1(cutoff=99)(torch.randn((5,100,2004))).size()}")