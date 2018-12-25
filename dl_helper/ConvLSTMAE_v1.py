import torch
import torch.nn as nn
import numpy as np

class ConvLSTMAE_v1(nn.Module):
  #epoch:99 train_loss:0.0003731108154170215 valid_loss:0.019016167148947716

  def __init__(self, cutoff=0):
    super(ConvLSTMAE_v1, self).__init__()
    self.cutoff = cutoff
    
    #encode
    self.conv0 = nn.Conv1d(100,200, 4, stride=2, padding=1)
    self.tanconv0 = nn.Tanh()
    self.conv1 = nn.Conv1d(200,400, 4, stride=2, padding=0)
    self.tanconv1 = nn.Tanh()
    self.maxpool0 = nn.MaxPool1d(3, return_indices=True)
    self.tanmaxpool0 = nn.Tanh()
    self.conv2 = nn.Conv1d(400, 800, 4, stride=2, padding=2)
    self.tanconv2 = nn.Tanh()
    self.conv3 = nn.Conv1d(800, 1600, 4, stride=2, padding=0)
    self.tanconv3 = nn.Tanh()
    self.maxpool1 = nn.MaxPool1d(2, return_indices=True)
    self.relumaxpool1 = nn.ReLU()
    
    #lstm
    self.lstm0 = nn.LSTM(20,20,5,batch_first=True)
    self.tanlstm0 = nn.Tanh()

    nn.init.xavier_uniform(self.lstm0.weight_ih_l0, gain=np.sqrt(2))
    nn.init.xavier_uniform(self.lstm0.weight_hh_l0, gain=np.sqrt(2))
    
    #decode
    self.maxunpool0 = nn.MaxUnpool1d(2)
    self.tanmaxunpool0 = nn.Tanh()
    self.convt0 = nn.ConvTranspose1d(1600, 800, 4, stride=2, padding=0)
    self.tanconvt0 = nn.Tanh()
    self.convt1 = nn.ConvTranspose1d(800, 400, 4, stride=2, padding=2)
    self.tanconvt1 = nn.Tanh()
    self.maxunpool1 = nn.MaxUnpool1d(3)
    self.tanmaxunpool1 = nn.Tanh()
    self.convt2 = nn.ConvTranspose1d(400, 200, 4, stride=2, padding=0)
    self.tanconvt2 = nn.Tanh()
    self.convt3 = nn.ConvTranspose1d(200, 100, 4, stride=2, padding=1)
    self.tanconvt3 = nn.Tanh()
    
    #output
    self.lin0 = nn.Linear(2004, 2004)
    self.relulin0 = nn.Sigmoid()
  
  def forward(self, input):
    #encode
    out = self.tanconv0(self.conv0(input))
    out = self.tanconv1(self.conv1(out))
    
    size0 = list(out.size())
    out, indices0 = self.maxpool0(out)
    out = self.tanmaxpool0(out)
    
    out = self.tanconv2(self.conv2(out))   
    out = self.tanconv3(self.conv3(out))
    
    size1 = list(out.size())
    out, indices1 = self.maxpool1(out)
    out = self.relumaxpool1(out)
    
    #lstm
    out = self.tanlstm0(self.lstm0(out)[0])
    
    #decode
    out = self.tanmaxunpool0(self.maxunpool0(out, indices1, output_size=size1))

    out = self.tanconvt0(self.convt0(out))
    out = self.tanconvt1(self.convt1(out))
    
    out = self.tanmaxunpool1(self.maxunpool1(out, indices0, output_size=size0))
    
    out = self.tanconvt2(self.convt2(out))
    out = self.tanconvt3(self.convt3(out))
    
    #out
    out = self.relulin0(self.lin0(out[:,self.cutoff-1,:]))
    
    return out