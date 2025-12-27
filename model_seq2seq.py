import tensorflow as tf
from model_encoder import build_encoder
from model_decoder import build_decoder
from config import *

def build_seq2seq(feat_dim):
    enc_in, enc_out, states = build_encoder(SEQ_LEN, feat_dim)
    dec_in, dec_out = build_decoder(SEQ_LEN, feat_dim, states)
    model = tf.keras.Model([enc_in, dec_in], dec_out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="mse"
    )
    return model
