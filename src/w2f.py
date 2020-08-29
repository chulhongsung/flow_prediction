import numpy as np
import tensorflow as tf
from tensorflow import keras

def w2f_preprocess(data, y, binary, t, k):
    n = data.shape[0] - t - k
    
    new_data1 = np.zeros((n, t-1, 6), dtype=np.float32)
    new_data2 = np.zeros((n, 1, 22), dtype=np.float32)
    
    target = np.zeros((n, 1, k), dtype=np.float32)
    binary_target = np.zeros((n, 1, k), dtype=np.float32)
    initial = np.zeros((n, 1, 1), dtype=np.float32)
    
    for i in range(n):
        new_data1[i, :, 0:4] = data[i:(t+i-1), 0:4]
        new_data1[i, :, 4] = binary[i:(t+i-1)]
        new_data1[i, :, 5] = y[i:(t+i-1)]
        
        new_data2[i, :, 0:20] = data[(t+i-1),:]
        new_data2[i, :, 20] = binary[(t+i-1)]
        new_data2[i, :, 21] = y[(t+i-1)]
        
        target[i, :, :] = y[(t+i):(t+i+k)]
        binary_target[i, :, :] = binary[(t+i):(t+i+k)]
        initial[i, :, :] = y[(t+i-1)]
        
    return new_data1, new_data2, target, binary_target, initial 

class Encoder(keras.layers.Layer):
    def __init__(self, d_model, l2_regularizer=0, dropout=0): 
        super(Encoder, self).__init__()        

        self.lstm1 = keras.layers.LSTM(d_model,
                                       return_sequences=True,
                                       return_state=True,
                                       activation='tanh',
                                       dropout=dropout,
                                       recurrent_initializer=keras.initializers.GlorotNormal(1),
                                       kernel_regularizer=keras.regularizers.l2(l2_regularizer),
                                       recurrent_regularizer=keras.regularizers.l2(l2_regularizer))
        

        self.lstm2 = keras.layers.LSTM(d_model,
                                       return_sequences=True,
                                       return_state=True,
                                       activation='tanh',
                                       dropout=dropout,
                                       recurrent_initializer=keras.initializers.GlorotNormal(1),
                                       kernel_regularizer=keras.regularizers.l2(l2_regularizer),
                                       recurrent_regularizer=keras.regularizers.l2(l2_regularizer))
        
    def call(self, data1, data2):
        _, lst_hidden, lst_cell_state  = self.lstm1(data1)
        h_, lst_hidden2, lst_cell_state2 = self.lstm2(data2, initial_state=[lst_hidden, lst_cell_state]) 
        
        return lst_hidden2, lst_cell_state2

class Decoder(keras.layers.Layer):
    def __init__(self, d_model, k, l2_regularizer=0, dropout=0): 
        super(Decoder, self).__init__()
        self.k = k
        
        self.decoder_lstm = keras.layers.LSTM(d_model,
                                       return_sequences=True,
                                       return_state=True,
                                       activation='tanh',
                                       dropout=dropout,
                                       recurrent_initializer=keras.initializers.GlorotNormal(1),
                                       kernel_regularizer=keras.regularizers.l2(l2_regularizer),
                                       recurrent_regularizer=keras.regularizers.l2(l2_regularizer))
        
        self.dense1 = keras.layers.Dense(1, activation='sigmoid')
        self.dense2 = keras.layers.Dense(1, activation='elu')
        
    def call(self, initial, encoder_hidden, encoder_cell): 
       
        hidden_x_, _, __  = self.decoder_lstm(initial, initial_state=[encoder_hidden, encoder_cell])
        hidden_x1 = self.dense1(hidden_x_)
        hidden_x2 = self.dense2(hidden_x_)
 
        output1 = [hidden_x1] 
        output2 = [hidden_x2]
        
       
        for i in range(self.k-1):
            hidden_x_, _, __ = self.decoder_lstm(hidden_x1, initial_state = [ _, __ ])
            hidden_x1 = self.dense1(hidden_x_)
            hidden_x2 = self.dense2(hidden_x_)
           
            output1.append(hidden_x1)
            output2.append(hidden_x2)     
           
        return tf.squeeze(tf.transpose(tf.convert_to_tensor(output1), perm=[1, 0, 2, 3])), tf.squeeze(tf.transpose(tf.convert_to_tensor(output2), perm=[1, 0, 2, 3])) 

class W2F(keras.models.Model):
    def __init__(self, d_model, k, l2_regularizer=0, dropout=0):
        super(W2F, self).__init__()
        self.encoder = Encoder(d_model, l2_regularizer, dropout)
        self.decoder = Decoder(d_model, k, l2_regularizer, dropout)
        
    def call(self, input):
        data1, data2, y = input
        encoder_hidden, encoder_cell = self.encoder(data1, data2)
        output1, output2 = self.decoder(y, encoder_hidden, encoder_cell)
        
        return output1, output1 * output2
