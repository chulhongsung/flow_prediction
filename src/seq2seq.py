import numpy as np
import tensorflow as tf
from tensorflow import keras

class Encoder(tf.keras.models.Model):
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

class Decoder(tf.keras.models.Model):
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
        
        self.dense = keras.layers.Dense(1)
        
    def call(self, initial, encoder_hidden, encoder_cell): 
        hidden_x_, _, __  = self.decoder_lstm(initial, initial_state=[encoder_hidden, encoder_cell])

        hidden_x = self.dense(hidden_x_)
        
        output = [hidden_x] 
       
        for i in range(self.k-1):
           hidden_x_, _, __ = self.decoder_lstm(hidden_x, initial_state = [ _, __ ]) 

           hidden_x = self.dense(hidden_x_)
           
           output.append(hidden_x)
                
        return tf.squeeze(tf.transpose(tf.convert_to_tensor(output), perm=[1, 0, 2, 3]))

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
