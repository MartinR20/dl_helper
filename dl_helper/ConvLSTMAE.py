import torch
import torch.nn as nn
import numpy as np

class ConvLSTMEncoder(nn.Module):
  def __init__(self, cutoff=0):
      super(ConvLSTMEncoder, self).__init__()    
      self.cutoff = cutoff
      self.conv1 = nn.Conv1d(100, 400, 64, stride=8)
      self.r1 = nn.ReLU()
      self.lstm1 = nn.LSTM(243, 128, 1, batch_first=True)
      self.r2 = nn.ReLU()
      self.conv2 = nn.Conv1d(400, 800, 64, stride=4)
      self.r3 = nn.ReLU() 
      self.lstm2 = nn.LSTM(17, 8, 1, batch_first=True)
      self.r4 = nn.ReLU()

      #initialize weights
      nn.init.xavier_uniform(self.lstm1.weight_ih_l0, gain=np.sqrt(2))
      nn.init.xavier_uniform(self.lstm1.weight_hh_l0, gain=np.sqrt(2))
      nn.init.xavier_uniform(self.lstm2.weight_ih_l0, gain=np.sqrt(2))
      nn.init.xavier_uniform(self.lstm2.weight_hh_l0, gain=np.sqrt(2))

  def forward(self, input):
    conv1_out = self.conv1(input)
    conv1_out = self.r1(conv1_out)
    lstm1_out, _ = self.lstm1(conv1_out)
    lstm1_out = self.r2(lstm1_out)
    conv2_out = self.conv2(lstm1_out)
    conv2_out = self.r1(conv2_out)
    lstm2_out, _ = self.lstm2(conv2_out)
    lstm2_out = self.r2(lstm2_out)
    return lstm2_out

class ConvLSTMDecoder(nn.Module):
  def __init__(self):
      super(ConvLSTMDecoder, self).__init__()    
      self.lstm1 = nn.LSTM(8, 17, 1, batch_first=True)
      self.r1 = nn.ReLU()
      self.conv1 = nn.ConvTranspose1d(800, 400, 64, stride=4)
      self.r2 = nn.ReLU() 
      self.lstm2 = nn.LSTM(128, 243, 1, batch_first=True)
      self.r3 = nn.ReLU()
      self.conv2 = nn.ConvTranspose1d(400, 100, 68, stride=8)
      self.r4 = nn.ReLU()
  
      #initialize weights
      nn.init.xavier_uniform(self.lstm1.weight_ih_l0, gain=np.sqrt(2))
      nn.init.xavier_uniform(self.lstm1.weight_hh_l0, gain=np.sqrt(2))
      nn.init.xavier_uniform(self.lstm2.weight_ih_l0, gain=np.sqrt(2))
      nn.init.xavier_uniform(self.lstm2.weight_hh_l0, gain=np.sqrt(2))

  def forward(self, input):
    lstm1_out, _ = self.lstm1(input)
    lstm1_out = self.r2(lstm1_out)
    conv1_out = self.conv1(lstm1_out)
    conv1_out = self.r1(conv1_out)
    lstm2_out, _ = self.lstm2(conv1_out)
    lstm2_out = self.r2(lstm2_out)
    conv2_out = self.conv2(lstm2_out)
    conv2_out = self.r1(conv2_out)
    return conv2_out
  
class ConvLSTMAE(nn.Module):
  def __init__(self):
      super(ConvLSTMAE, self).__init__()
      self.encoder = ConvLSTMEncoder().cuda(0)
      self.decoder = ConvLSTMDecoder().cuda(0)

  def forward(self, input):
      encoded_input = self.encoder(input)
      decoded_output = self.decoder(encoded_input)
      outputs = decoded_output[:,self.cutoff-1,:] # keep only last output of sequence

      return outputs
