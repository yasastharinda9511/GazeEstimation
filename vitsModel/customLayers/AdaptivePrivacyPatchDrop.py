import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np

class AdaptivePrivacyPatchDrop(layers.Layer):
    """
    Novel Privacy-Aware Patch Dropout Layer
    - Drops or attenuates patches based on learned sensitivity
    - Can improve privacy by removing identity-revealing patches
    """
    def __init__(self, base_drop_rate=0.1, sensitivity_strength=0.5, **kwargs):
        super().__init__(**kwargs)
        self.base_drop_rate = base_drop_rate          # Minimum drop rate
        self.sensitivity_strength = sensitivity_strength  # How strongly to drop sensitive patches
        self.last_mask = None

    def build(self, input_shape):
        # Learnable sensitivity per embedding dimension
        # shape: (embedding_dim,)
        self.sensitivity_weights = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
            name="sensitivity_weights"
        )
        super().build(input_shape)

    def call(self, patches, training=None):
        if not training or self.base_drop_rate <= 0.0:
            return patches

        # Compute patch sensitivity: L2 norm weighted by sensitivity_weights
        # shape: (batch_size, num_patches)
        patch_sensitivity = tf.norm(patches * self.sensitivity_weights, axis=-1)

        # Normalize to [0, 1] to get probability
        patch_sensitivity = patch_sensitivity / (tf.reduce_max(patch_sensitivity, axis=-1, keepdims=True) + 1e-6)

        # Compute adaptive drop probability per patch
        drop_prob = self.base_drop_rate + self.sensitivity_strength * patch_sensitivity
        drop_prob = tf.clip_by_value(drop_prob, 0.0, 0.95)  # avoid dropping all patches

        # Create random mask
        random_tensor = tf.random.uniform(tf.shape(drop_prob))
        mask = tf.cast(random_tensor >= drop_prob, patches.dtype)
        mask = tf.expand_dims(mask, axis=-1)  # broadcast over embedding dim
        self.last_mask = mask  # Store last mask for inspection if needed

        # Apply mask
        return patches * mask