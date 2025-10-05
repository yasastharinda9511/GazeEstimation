import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np

class UpsampleBlock(layers.Layer):
    """Enhanced upsampling block with skip connections support"""
    def __init__(self, filters, kernel_size=3, use_attention=False,  **kwargs):
        super(UpsampleBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_attention = use_attention
        
        # Main upsampling path
        self.upsample = layers.Conv2DTranspose(
            filters, 2, strides=2, padding='same',
            kernel_initializer='he_normal'
        )
        self.conv1 = layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_initializer='he_normal'
        )
        self.conv2 = layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_initializer='he_normal'
        )
        
        # Normalization and activation
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.activation = layers.Activation('gelu')
        
        # Optional spatial attention
        if use_attention:
            self.spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        
    def call(self, x, training=None):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        
        # Apply spatial attention if enabled
        if self.use_attention:
            attention_weights = self.spatial_attention(x)
            x = x * attention_weights
            
        return x