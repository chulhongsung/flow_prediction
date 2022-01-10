import os

import tensorflow as tf
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from w2f import W2F, w2f_preprocess
from seq2seq import Seq2Seq

t = 10
k = 7

encoder_input1, encoder_input2, y, binary, initial = w2f_preprocess(df_pre, musu_y, musu_binary, t, k)

y_= np.sum(y, axis=2) >= 1
split = StratifiedShuffleSplit(test_size=0.1, random_state=0)

for train_idx, test_idx in split.split(encoder_input1, y_):
    encoder1_train = encoder_input1[train_idx]
    encoder2_train = encoder_input2[train_idx]
    y_train = y[train_idx]
    binary_train = binary[train_idx]
    initial_train = initial[train_idx]
    
    encoder1_test = encoder_input1[test_idx]
    encoder2_test = encoder_input2[test_idx]
    y_test = y[test_idx]
    binary_test = binary[test_idx]
    initial_test = initial[test_idx]

bce = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

w2f = W2F(d_model=100, k=k, l2_regularizer=0.001, dropout=0.1)

w2f.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=[bce, mse])
w2f.fit([encoder1_train, encoder2_train, initial_train], [tf.squeeze(binary_train), tf.squeeze(y_train)], batch_size=100, epochs=100, verbose=2)

_, __, w2f_test_mse = w2f.evaluate([encoder1_test, encoder2_test, initial_test], [tf.squeeze(binary_test), tf.squeeze(y_test)])

print("W2F Test MSE: {:.03f}".format(w2f_test_mse))

s2s = Seq2Seq(d_model=100, k=k, l2_regularizer=0.001, dropout=0.1)

s2s.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss= mse)
s2s.fit([encoder1_train, encoder2_train, initial_train], tf.squeeze(y_train), batch_size=100, epochs=100, verbose=2)

s2s_test_mse = s2s.evaluate([encoder1_test, encoder2_test, initial_test], tf.squeeze(y_test))
print("Test MSE: {:.03f}".format(s2s_test_mse))
