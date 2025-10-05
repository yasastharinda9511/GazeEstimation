import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np

from vitsModel.customLayers.MultiHeadAttentionLayer import MultiHeadAttention

class TransformerBlock(layers.Layer):
    """Enhanced transformer block with improved regularization"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__( **kwargs)
        self.att = MultiHeadAttention(embed_dim, num_heads, rate)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),  # GELU often works better than ReLU
            layers.Dropout(rate),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        # Pre-norm architecture (often more stable)
        norm1 = self.layernorm1(inputs)
        attn_output = self.att(norm1, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        
        norm2 = self.layernorm2(out1)
        ffn_output = self.ffn(norm2, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output