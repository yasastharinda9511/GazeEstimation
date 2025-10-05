import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np

class PatchEmbedding(layers.Layer):
    """Enhanced patch embedding with learnable class token"""
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEmbedding, self).__init__( **kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        # Add learnable class token for better global representation
        self.class_token = self.add_weight(
            shape=(1, 1, projection_dim),
            initializer="random_normal",
            trainable=True,
            name="class_token"
        )

    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        
        # Add class token to each batch
        class_tokens = tf.broadcast_to(self.class_token, [batch_size, 1, self.class_token.shape[-1]])
        
        # Project patches and add positional encoding
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        # Concatenate class token
        encoded = tf.concat([class_tokens, encoded], axis=1)
        return encoded