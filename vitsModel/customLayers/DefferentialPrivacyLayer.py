import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np

class DifferentialPrivacyLayer(layers.Layer):
    """
    Differential Privacy Layer (Third component in peer review architecture)
    Adds calibrated noise for formal privacy guarantees
    """
    def __init__(self, noise_scale=0.1, clip_norm=1.0, always_apply=True, name="differential_privacy_layer", **kwargs):
        super(DifferentialPrivacyLayer, self).__init__(**kwargs)
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.always_apply = always_apply  # Always apply DP noise during training and inference
        
    def call(self, x, training=None):
        if training or self.always_apply:
            if self.clip_norm > 0:
                x = tf.clip_by_norm(x, self.clip_norm, axes=-1)
            
            noise = tf.random.normal(
                tf.shape(x), mean=0.0, stddev=self.noise_scale, dtype=x.dtype
            )
            return x + noise
        return x