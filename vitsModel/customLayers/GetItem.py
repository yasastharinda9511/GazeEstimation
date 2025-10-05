import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np

@tf.keras.utils.register_keras_serializable()
class GetItem(tf.keras.layers.Layer):
    """Serializable slicing layer: GetItem(slice(1, None))(x) == x[:, 1:, :]"""

    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        if isinstance(idx, slice):
            self.is_slice = True
            self.start = idx.start
            self.stop = idx.stop
            self.step = idx.step
            self.idx = None
        else:
            self.is_slice = False
            self.idx = idx
            self.start = self.stop = self.step = None

    def call(self, inputs):
        if self.is_slice:
            return inputs[:, self.start:self.stop:self.step, :]
        else:
            return inputs[:, self.idx, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "is_slice": self.is_slice,
            "idx": self.idx,
            "start": self.start,
            "stop": self.stop,
            "step": self.step
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Clean out any None placeholders before passing to __init__
        is_slice = config.pop("is_slice", False)
        idx = config.pop("idx", None)
        start = config.pop("start", None)
        stop = config.pop("stop", None)
        step = config.pop("step", None)

        if is_slice:
            return cls(slice(start, stop, step), **config)
        else:
            return cls(idx, **config)