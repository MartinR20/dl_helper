import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing
import psutil
from tqdm import tqdm
import torch

class dataObj():
  def __init__(self, path, valid_percentage, test_percentage, seq_len):
    print('...loading data')
    # import all stock prices 
    df = pd.read_csv(path, index_col = 0, encoding='utf-8-sig')
    print(df.keys(), df.index.name)
    
    print('...processing data')
    data, self.companys = self.__get_matrix__(df, (1762,4), seq_len)
    
    print(f'Raw data shape: {data.shape}')

    array_list = self.__seperate_data__(valid_percentage, test_percentage, data)  

    self.x_train = torch.from_numpy(array_list[0]).float()
    self.y_train = torch.from_numpy(array_list[1]).float()
    self.x_valid = torch.from_numpy(array_list[2]).float()
    self.y_valid = torch.from_numpy(array_list[3]).float()
    self.x_test  = torch.from_numpy(array_list[4]).float()
    self.y_test  = torch.from_numpy(array_list[5]).float()

    print(f'Reshaped size: \n  train:{self.x_train.shape} \n  validtion:{self.x_valid.shape} \n  test:{self.x_test.shape}')

    print('...finished')
    print(f'number of different stocks: {len(list(set(df.symbol)))}')

  def reshape(self):
    self.x_train, self.y_train =  self.__reshape__(self.x_train, self.y_train)
    self.x_valid, self.y_valid =  self.__reshape__(self.x_valid, self.y_valid)
    self.x_test, self.y_test =  self.__reshape__(self.x_test, self.y_test)

  @staticmethod
  def __reshape__(X, Y):
    x_sz = X.size()
    X = X.permute(1,2,0,3).contiguous().view(x_sz[1],x_sz[2],x_sz[0]*x_sz[3])
    
    y_sz = Y.size()
    Y = Y.permute(1,0,2).contiguous().view(y_sz[1],y_sz[0]*y_sz[2])

    return X, Y

  @staticmethod
  def __get_matrix__(df, data_shape, seq_len):
    data = []

    df.drop(['volume'],1,inplace=True) 
    df.sort_values([u'symbol',u'date'],inplace=True)

    ## initalize object for normalization
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()

    ndf = df.values
    symbol_column = ndf[:,0]
    
    keys = sorted(set(df.symbol))
    values = set(range(len(df.symbol)))
    companys = dict(zip(keys, values))
    
    startindex = 0

    tbar = tqdm(range(len(companys)), ascii=True)

    for idx in tbar:  
      if not idx % 50:
        mem = psutil.virtual_memory()
        tbar.set_description(f"memory used:{int(mem.used/10**8)/10}/{int(mem.total/10**8)/10}GB")

      for idy in range(startindex, symbol_column.shape[0]):
        if keys[idx] == symbol_column[idy]:
          first = idy 
          startindex = idy
          break
      
      for idy in range(startindex, symbol_column.shape[0]):
        if keys[idx] != symbol_column[idy]:
          last = idy
          startindex = idy
          break

      if first >= last: 
        last = symbol_column.shape[0]

      df_stock = ndf[first:last]
      df_stock = df_stock[:,1:].astype(np.float64)
      
      ## noramlize input
      df_stock = min_max_scaler.fit_transform(df_stock)
      
      df_zeros = np.zeros((data_shape[0]-seq_len, seq_len, data_shape[1]))
      
      # create all possible sequences of length seq_len
      start_offset = data_shape[0] - df_stock.shape[0] 
      
      for row in range(df_stock.shape[0] - seq_len): 
          df_zeros[start_offset + row] = df_stock[row : row + seq_len]
    
      data.append(df_zeros)
      
    return np.array(data), companys

  @staticmethod
  def __seperate_data__(valid_set_size_percentage, test_set_size_percentage, data):     
    valid_set_size = int(np.round(valid_set_size_percentage/100*data[0].shape[0]))
    test_set_size = int(np.round(test_set_size_percentage/100*data[0].shape[0]))
    train_set_size = data[0].shape[0] - (valid_set_size + test_set_size)

    x_train = data[:,:train_set_size,:-1,:]
    y_train = data[:,:train_set_size,-1,:]

    x_valid = data[:,train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[:,train_set_size:train_set_size+valid_set_size,-1,:]

    x_test = data[:,train_set_size+valid_set_size:,:-1,:]
    y_test = data[:,train_set_size+valid_set_size:,-1,:]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]
