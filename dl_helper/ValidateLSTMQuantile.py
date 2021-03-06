import torch
import torch.nn as nn

class ValidateLSTMQuantile(nn.Module):
  def __init__(self, quantiles, input_size, hidden_size, num_layers, output_size, cutoff=0):
    super(ValidateLSTMQuantile, self).__init__()
    self.cutoff = cutoff
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.output_size = output_size
    self.quantiles = torch.tensor(quantiles)
    
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.quantile_layers = []
    
    for idx in range(len(quantiles)):
      self.quantile_layers.append(nn.Linear(hidden_size, output_size))
      
  def forward(self,input):
    rnn_outputs, hidden = self.lstm(input)
    stacked_rnn_outputs = rnn_outputs.contiguous().view(-1, self.hidden_size)
    
    stacked_outputs = torch.empty([0])
    for layer in self.quantile_layers:
      layer_output = layer(stacked_rnn_outputs)
      stacked_outputs = torch.cat((stacked_outputs, layer_output))
        
    outputs = stacked_outputs.view(len(self.quantile_layers), -1, self.cutoff, self.output_size)
    #outputs = outputs[:, :,self.cutoff:,:] # keep only last output of sequence
    return outputs
