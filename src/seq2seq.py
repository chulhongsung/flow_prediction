#%% library
import numpy as np
import tensorflow as tf
from tensorflow import keras
#%% Class Encoder
class Encoder(tf.keras.models.Model):
    def __init__(self, d_model, l2_regularizer=0, dropout=0): # d_model: LSTM cell의 ouput dimension
        super(Encoder, self).__init__()        
        # lstm1: (t-k+1)시점부터 (t-1)시점까지의 기후자료만을 input으로 하는 LSTM layer
        self.lstm1 = keras.layers.LSTM(d_model,
                                       return_sequences=True,
                                       return_state=True,
                                       activation='tanh',
                                       dropout=dropout,
                                       recurrent_initializer=keras.initializers.GlorotNormal(1),
                                       kernel_regularizer=keras.regularizers.l2(l2_regularizer),
                                       recurrent_regularizer=keras.regularizers.l2(l2_regularizer))
        
        # lstm2: 마지막 t시점에서 기후자료와 예보자료를 input으로 하는 LSTM cell
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
        h_, lst_hidden2, lst_cell_state2 = self.lstm2(data2, initial_state=[lst_hidden, lst_cell_state]) # lstm1의 hidden state와 cell state를 initial state로 받음.
        
        return lst_hidden2, lst_cell_state2
#%% Class Decoder
class Decoder(tf.keras.models.Model):
    def __init__(self, d_model, k, l2_regularizer=0, dropout=0): # d_model: LSTM cell의 ouput dimension
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
        
        # Encoder에서 lstm의 output을 받아서 0~1 값으로 변환하기 위한 dense layer
        # self.dense = [keras.layers.Dense(1, activation='sigmoid') for _ in range(self.k)] # 1: output dimension
        self.dense = keras.layers.Dense(1)
        
    def call(self, initial, encoder_hidden, encoder_cell): 
        # initial: input 마지막 시점 개방 여부, encoder_hidden: 인코더에서 출력된 hidden state, encoder_cell: 인코더에서 출력된 cell state
        hidden_x_, _, __  = self.decoder_lstm(initial, initial_state=[encoder_hidden, encoder_cell])

        hidden_x = self.dense(hidden_x_)
        
        output = [hidden_x] # 출력을 위한 output들을 저장하기 위한 list
        
        #반복문을 통해 Decoder를 실행
        for i in range(self.k-1):
           hidden_x_, _, __ = self.decoder_lstm(hidden_x, initial_state = [ _, __ ]) # hidden_state, hidden_state, cell_state

           hidden_x = self.dense(hidden_x_)
           
           output.append(hidden_x)
                
        return tf.squeeze(tf.transpose(tf.convert_to_tensor(output), perm=[1, 0, 2, 3]))
#%% Weather2Flow
class Seq2Seq(tf.keras.models.Model):
    def __init__(self, d_model, k, l2_regularizer=0, dropout=0):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(d_model, l2_regularizer, dropout)
        self.decoder = Decoder(d_model, k, l2_regularizer, dropout)
        
    def call(self, input):
        data1, data2, y = input
        encoder_hidden, encoder_cell = self.encoder(data1, data2)
        output = self.decoder(y, encoder_hidden, encoder_cell)
        
        return output
