import tensorflow as tf

def build_encoder(seq_len, feat_dim):
    inputs = tf.keras.Input(shape=(seq_len, feat_dim))
    x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    x2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x1)
    x = tf.keras.layers.Add()([x1, x2])
    out, fh, fc, bh, bc = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_state=True)
    )(x)
    h = tf.keras.layers.Concatenate()([fh, bh])
    c = tf.keras.layers.Concatenate()([fc, bc])
    return inputs, out, [h, c]
