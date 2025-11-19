import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np

class PrivacyAwareFeatureExtractor(layers.Layer):
    """
    Privacy-Aware Feature Extractor (First component in peer review architecture)
    Extracts features while removing identity-specific information
    """
    def __init__(self, projection_dim, anonymization_strength=0.3, name="privacy_aware_feature_extractor",  **kwargs):
        super(PrivacyAwareFeatureExtractor, self).__init__( **kwargs)
        self.projection_dim = projection_dim
        self.anonymization_strength = anonymization_strength
        
        # Privacy-aware feature projection
        self.privacy_projection = layers.Dense(
            projection_dim, 
            activation='gelu',
            kernel_initializer='he_normal',
            name='privacy_projection'
        )
        
        # Identity suppression layer
        self.identity_suppression = layers.Dense(
            projection_dim,
            use_bias=False,
            kernel_initializer='orthogonal',
            name='identity_suppression'
        )
        
        # Feature normalization
        self.feature_norm = layers.LayerNormalization(epsilon=1e-6, name='privacy_norm')
        
    def call(self, x, training=None):
        features = self.privacy_projection(x)
        
        # ✅ ALWAYS compute suppressed (ensures weights are always used)
        suppressed = self.identity_suppression(features)
        
        # Control effect via anonymization_strength
        features = (1.0 - self.anonymization_strength) * features + \
                self.anonymization_strength * suppressed
        
        return self.feature_norm(features)