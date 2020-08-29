#%%
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from w2f import W2F, w2f_preprocess
from seq2seq import Seq2Seq

from multiregression import preprocess2
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
#%%
os.chdir('/Users/chulhongsung/Desktop/lab/flow prediction')
#%%
df_pre = np.load('./data/df_pre.npy')
musu_y = np.load('./data/musu_y.npy')
musu_binary = np.load('./data/musu_binary.npy')
#%%
t = 10
k = 7
#%%
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
#%% Loss 
bce = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()
#%% W2F
w2f = W2F(d_model=100, k=k, l2_regularizer=0.001, dropout=0.1)
#%% Weather2Flow model Train
w2f.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=[bce, mse])
w2f.fit([encoder1_train, encoder2_train, initial_train], [tf.squeeze(binary_train), tf.squeeze(y_train)], batch_size=100, epochs=100, verbose=2)
#%% TEST 
_, __, w2f_test_mse = w2f.evaluate([encoder1_test, encoder2_test, initial_test], [tf.squeeze(binary_test), tf.squeeze(y_test)])

print("W2F Test MSE: {:.03f}".format(w2f_test_mse))
#%% Reference model
s2s = Seq2Seq(d_model=100, k=k, l2_regularizer=0.001, dropout=0.1)
#%%
s2s.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss= mse)
s2s.fit([encoder1_train, encoder2_train, initial_train], tf.squeeze(y_train), batch_size=100, epochs=100, verbose=2)
#%%
s2s_test_mse = s2s.evaluate([encoder1_test, encoder2_test, initial_test], tf.squeeze(y_test))
print("Test MSE: {:.03f}".format(s2s_test_mse))
#%%
#%% RandomForest 
# rf_input, rf_y = preprocess2(df_pre, musu_binary, t, k)

# rf_input_train = rf_input[train_idx]
# rf_y_train = rf_y[train_idx]

# rf_input_test = rf_input[test_idx]
# rf_y_test = rf_y[test_idx]

# _samples, n_features = rf_input_train.shape # 10,100
# n_classes = 2
# rforest = RandomForestClassifier(random_state=1)
# multi_target_rforest = MultiOutputClassifier(rforest)
# rf_model = multi_target_rforest.fit(rf_input_train, rf_y_train)
#%% ROC CURVE 
# fpr, tpr, thr = roc_curve(np.reshape(np.squeeze(binary_test), [-1, 1]), np.reshape(w2f_predict1, [-1, 1]))
# auc(fpr, tpr)
# #%%
# rf_test_prob = rf_model.predict_proba(rf_input_test)
# rf_result = np.array(rf_test_prob).reshape([-1, 2])[:, 1]

# rf_fpr, rf_tpr, rf_thr = roc_curve(np.reshape(np.squeeze(binary_test), [-1, 1]), rf_result)

# auc(rf_fpr, rf_tpr)
# #%%
# plt.plot(fpr, tpr, 'b--',
#          label='Proposed model({:.03f})'.format(auc(fpr,tpr)), lw=2)
# plt.plot(rf_fpr, rf_tpr, 'g-.',
#          label='RandomForest({:.03f})'.format(auc(rf_fpr, rf_tpr)), lw=2)
# plt.plot([0,0,1],
#          [0,1,1],
#          linestyle=':',
#          color='black',
#          label='Perfect performance(1)')
# plt.xlabel('false positive rate')
# plt.ylabel('true positive rate')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.show()
# #%%
# def AR_preprocess(y, lag):
#     n = y.shape[0] - lag - 7
        
#     y_lag = np.zeros((n, lag))
#     target = np.zeros((n, 7))
    
#     for i in range(n):
#         y_lag[i, :] = y[i:(i+lag)]
#         target[i, :] = y[(i+lag):(i+lag+7)]    
#     return y_lag, target
# #%%
# y_ar_lag, y_ar_target = AR_preprocess(musu_y, 10)

# ar_train = y_ar_lag[train_idx]
# target_ar_train = y_ar_target[train_idx]

# ar_test = y_ar_lag[test_idx]
# target_ar_test = y_ar_target[test_idx]
# #%%
# plot_acf(pd.Series(musu_y[0:560]), lags=40)
# #%%
# ar_train = musu_y[0:560]
# ar_test = musu_y[560:700]
# # %%
# mor = MultiOutputRegressor(LinearRegression()).fit(ar_train, target_ar_train)
# mse(mor.predict(ar_test).reshape([-1]), tf.reshape(tf.squeeze(target_ar_test), [-1]))
# #<tf.Tensor: shape=(), dtype=float64, numpy=44.29827117919922>
# #%%
# plt.plot(tf.reshape(tf.squeeze(target_ar_test), [-1]).numpy(), color='red')
# plt.plot(mor.predict(ar_test).reshape([-1]))
# plt.show()
# # %%
# plt.plot(tf.reshape(tf.squeeze(y_test), [-1]).numpy(), color='red', alpha=0.5)
# plt.plot(tf.reshape(w2f_predict1 * w2f_predict2, [-1]), alpha=0.6)
# plt.show()
# # %%
# plt.plot(tf.reshape(tf.squeeze(y_test), [-1]).numpy(), color='red', alpha=0.5)
# plt.plot(tf.reshape(s2s_predict, [-1]), alpha=0.6)
# plt.show()