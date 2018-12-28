from dl_helper import *

obj = dataObj('..\\data\\prices-split-adjusted.csv', 10, 10, 10)

print(f"ValidateLSTM:{ValidateLSTM(4, 200, 2, 4, cutoff=9)(obj.x_train[obj.companys['AMZN']]).size()}")
print(f"ValidateLSTMQuantile:{ValidateLSTMQuantile([0.5,0.9,1], 4, 200, 2, 4, cutoff=9)(obj.x_train[obj.companys['AMZN']]).size()}")

obj.reshape()

print(f"LSTMAE:{LSTMAE(2004, 512, 1, False, cutoff=9)(obj.x_train).size()}")

## not enough ram to test these on my laptop (they expect a sequence length of 100)
#print(ConvLSTMAE_v0(cutoff=9)(obj.x_train).size())
#print(ConvLSTMAE_v1(cutoff=9)(obj.x_train).size())