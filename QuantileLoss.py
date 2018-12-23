class QuantileLoss(nn.modules.loss._Loss):
  """
    QauntileLoss is basically MSE but with the adittion that
    it lets you choose if you value over or underestimation more
    
    y  = target
    yp = input 
    
    Ly(y,yp) = sum( (gamma - 1) * abs(y - yp) ) + sum( gamma * abs(y - yp) )
              y < yp                             y > yp
  """
  
  
  def __init__(self, quantiles, size_average=None, reduce=None, reduction='elementwise_mean'):
    super(QuantileLoss, self).__init__(size_average, reduce, reduction)
    self.quantiles = torch.tensor(quantiles, device=0)
    
  def forward(self, output, y):
    losses = torch.empty([0], device=0)
    
    for i, quantile in enumerate(quantiles):
      error = y - output[i]
      loss = torch.mean(torch.max(quantile*error, (quantile-1)*error), dim=-1)
      losses = torch.cat((losses, loss))
      
    combined_loss = torch.sum(losses)
      
    return combined_loss
