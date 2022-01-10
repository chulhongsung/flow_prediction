#%%
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras as K
#%%
class GLULN(K.layers.Layer):
    def __init__(self, d_model):
        super(GLULN, self).__init__()    
        self.dense1 = K.layers.Dense(d_model, activation='sigmoid')
        self.dense2 = K.layers.Dense(d_model)
        self.layer_norm = K.layers.LayerNormalization()
        
    def call(self, x, y):
        return self.layer_norm(tf.keras.layers.Multiply()([self.dense3(x),
                                        self.dense4(x)]) + y)
#%% Gating mechanism
class GatedResidualNetwork(K.layers.Layer):
    def __init__(self, d_model, dr): 
        super(GatedResidualNetwork, self).__init__()        
        # lstm1: (t-k+1)시점부터 (t-1)시점까지의 기후자료만을 input으로 하는 LSTM layer
        self.dense1 = K.layers.Dense(d_model, activation='elu')        
        self.dense2 = K.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dr)
        self.glu_and_layer_norm = GLULN(d_model)
        
        
    def call(self, a):
        eta_2 = self.dense1(a)
        eta_1 = self.dropout(self.dense2(eta_2))
        grn_output = self.glu_and_layer_norm(eta_1, a)
        
        return grn_output
#%% Variable selection network
class VariableSelectionNetwork(K.layers.Layer):
    def __init__(self, d_model, d_input, dr):
        super(VariableSelectionNetwork, self).__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.dr = dr
        self.v_grn = GatedResidualNetwork(d_input, dr)
        self.softmax = K.layers.Softmax()
        
        
    def call(self, xi):
        
        Xi = tf.keras.layers.Flatten()(xi) # tf.reshape(xi, [tf.shape(xi)[0], 1, -1])
        
        weights = tf.expand_dims(self.softmax(self.v_grn(Xi)), axis=-1)
            
        tmp_xi_list = []                    
        
        for i in range(self.d_input):
            tmp_xi = GatedResidualNetwork(self.d_model, self.dr)(xi[:, i:i+1, :])            
            tmp_xi_list.append(tmp_xi)
        
        xi_list = tf.concat(tmp_xi_list, axis=1)
        combined = tf.keras.layers.Multiply()([weights, xi_list]) # attention

        vsn_output = tf.reduce_sum(combined, axis=1) 
    
        return vsn_output
    
#%%
def scaled_dot_product_attention(q, k, v, d_model, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(d_model, tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    # mask_ = mask[:, tf.newaxis, ...] - 1 # (batch_size, num_heads, seq_len_q, seq_len_q)

    # scaled_attention_logits += (mask_ * 1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = K.layers.Softmax()(scaled_attention_logits)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class InterpretableMultiHeadAttention(K.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, self.d_model, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        scaled_attention_mean = tf.reduce_mean(scaled_attention, axis=2)
        
        concat_attention = tf.reshape(scaled_attention_mean,
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
#%%
imha = InterpretableMultiHeadAttention(4, 2)

tmp_x = tf.random.normal([2, 3, 4])

imha(tmp_x, tmp_x, tmp_x)
#%%

#%%
class TemporalFusionDecoder(K.layers.Layer):
    def __init__(self, d_model, dr, num_heads):
        super(TemporalFusionDecoder, self).__init__()
        self.d_model = d_model
        self.dr = dr
        self.lstm_obs = K.layers.LSTM(d_model2,
                                      return_sequence=True,
                                      return_state=True,
                                      activation='tahn',
                                      recurrent_activation='sigmoid',
                                      recurrent_dropout=0,
                                      unroll=False,
                                      use_bias=True)
        
        self.lstm_future = K.layers.LSTM(d_model,
                                return_sequence=True,
                                activation='tahn',
                                recurrent_activation='sigmoid',
                                recurrent_dropout=0,
                                unroll=False,
                                use_bias=True)        

        self.glu_and_layer_norm1 = GLULN(d_model)
        
        self.imha = InterpretableMultiHeadAttention(d_model, num_heads=num_heads)
        
        self.glu_and_layer_norm2 = GLULN(d_model)
        
        
    def call(self, obs_input, future_input):
        obs_lstm, obs_h, obs_c = self.lstm_obs(obs_input)
        future_lstm = self.lstm_future(future_input, initial_state=[obs_h, obs_c])
        
        lstm_layer = tf.concat([obs_lstm, future_lstm], axis=1)    
        input_embeddings = tf.concat([obs_input, future_input], axis=1)
        
        glu_phi = self.glu_and_layer_norm1(lstm_layer, input_embeddings)
        B = self.imha(glu_phi, glu_phi, glu_phi)
        delta = self.glu_and_layer_norm2(B, glu_phi)
    
        return delta, glu_phi
#%% 
class PointWiseFeedForward(K.layers.Layer):
    def __init__(self, d_model, dr):
        self.grn = GatedResidualNetwork(d_model, dr)
        self.glu_and_layer_norm = GLULN(d_model)
        
        
    def call(self, delta, phi):
        varphi_ = self.grn(delta)
        varphi = self.glu_and_layer_norm(varphi_, phi)
        
        return varphi
#%%
class TFT(K.models.Model):
    def __init__(self, d_model, d_input, dr, cat_dim, cat_len, num_heads):
        super(TFT, self).__init__()
        self.d_model = d_model
        self.dr = dr
        self.embedding = K.layers.Embedding(input_dim=cat_len, output_dim=d_model, input_length=cat_dim)
        self.vsn1 = K.layers.TimeDistributed(VariableSelectionNetwork(d_model, d_input, dr)) ### past input
        self.vsn2 = K.layers.TimeDistributed(VariableSelectionNetwork(d_model, d_input, dr)) ### future input
        self.tfd = TemporalFusionDecoder(d_model, dr, num_heads)
        self.pwff = PointWiseFeedForward(d_model, dr)
        
        
    def call(self, obs_inputs, future_inputs):
        ### embdding 해서 xi 만들기
        x1 = self.vsn1(obs_inputs)
        x2 = self.vsn2(future_inputs)
        
        delta, glu_phi = self.tfd(x1, x2)
        varphi = self.pwff(delta, glu_phi)
        
        return varphi