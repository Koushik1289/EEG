import tensorflow as tf

def build_decoder(seq_len, feat_dim, states):
    inputs = tf.keras.Input(shape=(seq_len, feat_dim))
    lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
    out, _, _ = lstm(inputs, initial_state=states)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(feat_dim))(out)
    return inputs, out
