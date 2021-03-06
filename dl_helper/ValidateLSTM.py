import torch
import torch.nn as nn

class ValidateLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size, cutoff=0):
    super(ValidateLSTM, self).__init__()
    self.cutoff = cutoff
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.output_size = output_size
    
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.linear = nn.Linear(hidden_size, output_size)

    
  def forward(self,input):
    rnn_outputs, hidden = self.lstm(input)
    stacked_rnn_outputs = rnn_outputs.contiguous().view(-1, self.hidden_size)
    stacked_outputs = self.linear(stacked_rnn_outputs)
    outputs = stacked_outputs.view(-1, self.cutoff+1, self.output_size)
    outputs = stacked_outputs[:,self.cutoff-1,:] # keep only last output of sequence
    return outputs
