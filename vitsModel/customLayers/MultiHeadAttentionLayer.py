import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np

class MultiHeadAttention(layers.Layer):
    """Improved multi-head attention with better initialization"""
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__( **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim, kernel_initializer="glorot_uniform")
        self.key_dense = layers.Dense(embed_dim, kernel_initializer="glorot_uniform")
        self.value_dense = layers.Dense(embed_dim, kernel_initializer="glorot_uniform")
        self.combine_heads = layers.Dense(embed_dim, kernel_initializer="glorot_uniform")
        self.dropout = layers.Dropout(dropout_rate)

    def attention(self, query, key, value, training=None):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.dropout(weights, training=training)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        
        attention, weights = self.attention(query, key, value, training=training)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output