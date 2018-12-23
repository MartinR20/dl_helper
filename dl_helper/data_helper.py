import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing
import psutil

def get_matrix(input_df, data_shape, seq_len):
  company_names = set(input_df.symbol)

  data = []
  name_array_map = dict()
  df = input_df.drop(['volume'],1) 
  ## initalize object for normalization
  min_max_scaler = sklearn.preprocessing.MinMaxScaler()
  
  for idx, company in enumerate(company_names):
    name_array_map[company] = idx
    df_stock = df[df.symbol == company]

    df_stock = df_stock.drop(['symbol'],1)
    ## noramlize input
    df_stock = min_max_scaler.fit_transform(df_stock)
    
    df_zeros = np.zeros((data_shape[0]-seq_len, seq_len, data_shape[1]))
    
    # create all possible sequences of length seq_len
    
    start_offset = data_shape[0] - df_stock.shape[0] 
    
    for row in range(df_stock.shape[0] - seq_len): 
        df_zeros[start_offset + row] = df_stock[row : row + seq_len]
    
    if not idx % 50:
      mem = psutil.virtual_memory()
      print(f"idx {idx} memory used:{int(mem.used/10**8)/10}/{int(mem.total/10**8)/10}GB")
  
    data.append(df_zeros)
    
  return np.array(data), name_array_map

def seperate_data(valid_set_size_percentage, test_set_size_percentage, data):     
  valid_set_size = int(np.round(valid_set_size_percentage/100*data[0].shape[0]));  
  test_set_size = int(np.round(test_set_size_percentage/100*data[0].shape[0]));
  train_set_size = data[0].shape[0] - (valid_set_size + test_set_size);

  x_train = data[:,:train_set_size,:-1,:]
  y_train = data[:,:train_set_size,-1,:]

  x_valid = data[:,train_set_size:train_set_size+valid_set_size,:-1,:]
  y_valid = data[:,train_set_size:train_set_size+valid_set_size,-1,:]

  x_test = data[:,train_set_size+valid_set_size:,:-1,:]
  y_test = data[:,train_set_size+valid_set_size:,-1,:]

  return [x_train, y_train, x_valid, y_valid, x_test, y_test]

def prepare_data(path, valid_set_size_percentage, test_set_size_percentage, seq_len):
  # import all stock prices 
  df = pd.read_csv(path, index_col = 0)

  # number of different stocks
  print('\nnumber of different stocks: ', len(list(set(df.symbol))))
  
  data, name_array_map = get_matrix(df, (1762,4), seq_len)
  print(data.shape)
  
  return seperate_data(valid_set_size_percentage, test_set_size_percentage, data)
