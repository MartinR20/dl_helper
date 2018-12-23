class ValidateLSTMQuantile(nn.Module):
  def __init__(self, quantiles, input_size, hidden_size, num_layers, output_size):
    super(ValidateLSTMQuantile, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.output_size = output_size
    self.quantiles = torch.tensor(quantiles,device=0)
    
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.quantile_layers = []
    
    for idx in range(len(quantiles)):
      self.quantile_layers.append(nn.Linear(hidden_size, output_size).to(0))
      
  def forward(self,input):
    rnn_outputs, hidden = self.lstm(input)
    stacked_rnn_outputs = rnn_outputs.contiguous().view(-1, self.hidden_size)
    
    stacked_outputs = torch.empty([0], device=0)
    for layer in self.quantile_layers:
      layer_output = layer(stacked_rnn_outputs)
      stacked_outputs = torch.cat((stacked_outputs, layer_output))
        
    outputs = stacked_outputs.view(len(self.quantile_layers), -1, n_steps, self.output_size)
    outputs = outputs[:, :,n_steps-1,:] # keep only last output of sequence
    return outputs
