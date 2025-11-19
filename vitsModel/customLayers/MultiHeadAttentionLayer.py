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
        # 1. Calculate standard score
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # === STEP 2 FIX: Auto-Detect Zeros ===
        # Calculate the magnitude (sum of absolute values) of every patch in the Key
        key_magnitude = tf.reduce_sum(tf.abs(key), axis=-1)  # Shape: (Batch, Heads, Patches)
        
        # Create a mask: 1.0 if data exists, 0.0 if it is a zero-vector
        # We add a tiny epsilon (1e-6) to handle floating point errors
        auto_mask = tf.cast(key_magnitude > 1e-6, tf.float32)
        
        # Reshape for broadcasting: (Batch, Heads, 1, Patches)
        # This ensures the mask applies to all query positions
        auto_mask = tf.expand_dims(auto_mask, axis=-2)

        # Apply the penalty: If mask is 0, subtract 1 billion from the score.
        # This forces Softmax to output 0.0 for these patches.
        scaled_score += (1.0 - auto_mask) * -1e9
        # =====================================

        # 3. Standard Softmax
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