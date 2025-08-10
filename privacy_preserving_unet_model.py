import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

class LightInvarianceLayer(Layer):
    """
    Adds photometric augmentations for light invariance:
      - Random brightness and contrast during training
      - Optional CLAHE (uses OpenCV) during training
      - Per-image standardization at inference & training end

    Config:
      - brightness_delta: float
      - contrast_lower, contrast_upper: float
      - use_clahe: bool
      - clahe_clip_limit, clahe_tile_grid_size
      - training_only_augment: bool (apply chosen augmentations only during training)
      - preserve_range: whether to assume inputs in [0,1] (True) or [0,255]
    """

    def __init__(self,
                 brightness_delta=0.1,
                 contrast_lower=0.9,
                 contrast_upper=1.1,
                 use_clahe=False,
                 clahe_clip_limit=2.0,
                 clahe_tile_grid_size=(8, 8),
                 training_only_augment=True,
                 preserve_range=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.brightness_delta = float(brightness_delta)
        self.contrast_lower = float(contrast_lower)
        self.contrast_upper = float(contrast_upper)
        self.use_clahe = bool(use_clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = tuple(clahe_tile_grid_size)
        self.training_only_augment = bool(training_only_augment)
        self.preserve_range = bool(preserve_range)

    def _apply_clahe_numpy(self, img):
        # img expected HWC float32 in [0,1] or [0,255] depending on preserve_range
        import cv2
        import numpy as np
        img_np = img.numpy().astype(np.float32)
        if self.preserve_range:
            # map to [0,255] for CLAHE
            arr = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(img_np, 0, 255).astype(np.uint8)

        # If single-channel, apply CLAHE directly; else convert to LAB and apply to L
        if arr.ndim == 2 or arr.shape[2] == 1:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)
            out = clahe.apply(arr.squeeze())
            out = out[..., np.newaxis]
        else:
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if self.preserve_range:
            out = out.astype(np.float32) / 255.0
        else:
            out = out.astype(np.float32)

        return out

    def call(self, inputs, training=None):
        # Handle case where training parameter is not passed
        if training is None:
            training = tf.constant(True)  # Default to training mode
        
        def augment(x):
            # x is a float32 tensor with shape [B, H, W, C]
            x_aug = x
            # Random brightness
            x_aug = tf.image.random_brightness(x_aug, max_delta=self.brightness_delta)
            # Random contrast
            x_aug = tf.image.random_contrast(x_aug, lower=self.contrast_lower, upper=self.contrast_upper)
            # option: other photometric transforms could be added (saturation/hue) for RGB
            if self.use_clahe:
                # apply CLAHE per-image using tf.py_function (maps to numpy+cv2)
                def _clahe_fn(img):
                    return tf.py_function(func=self._apply_clahe_numpy, inp=[img], Tout=tf.float32)
                # map_fn over batch
                x_aug = tf.map_fn(_clahe_fn, x_aug, dtype=tf.float32)
            return x_aug

        # If training-only augmentations requested, condition on training
        if self.training_only_augment:
            x = tf.cond(tf.cast(training, tf.bool),
                        lambda: augment(inputs),
                        lambda: inputs)
        else:
            x = augment(inputs)

        # Always do per-image standardization (zero mean, unit variance)
        # But per_image_standardization expects float32
        x = tf.map_fn(lambda im: tf.image.per_image_standardization(im), x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'brightness_delta': self.brightness_delta,
            'contrast_lower': self.contrast_lower,
            'contrast_upper': self.contrast_upper,
            'use_clahe': self.use_clahe,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_grid_size': self.clahe_tile_grid_size,
            'training_only_augment': self.training_only_augment,
            'preserve_range': self.preserve_range
        })
        return cfg

class FeatureEncryptionLayer(Layer):
    def __init__(self, encryption_strength=0.5, use_random_projection=True, **kwargs):
        super().__init__(**kwargs)
        self.encryption_strength = encryption_strength
        self.use_random_projection = use_random_projection

    def build(self, input_shape):
        # Create learnable encryption matrix
        self.encryption_matrix = self.add_weight(
            name='encryption_matrix',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='orthogonal',
            trainable=False
        )
        
        # Create bias for additional obfuscation
        self.encryption_bias = self.add_weight(
            name='encryption_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=False
        )
        
        self.random_projection = self.add_weight(
            name='random_projection',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='random_normal',
            trainable=False
        )
    def call(self, inputs, training=None):
        # Apply learnable transformation
        encrypted = tf.matmul(inputs, self.encryption_matrix) + self.encryption_bias
        
        encrypted = tf.matmul(encrypted, self.random_projection)

        encrypted = tf.nn.tanh(encrypted)
        
        # Mix original and encrypted features based on strength
        mixed = (1 - self.encryption_strength) * inputs + self.encryption_strength * encrypted
        
        return mixed
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'encryption_strength': self.encryption_strength,
            'use_random_projection': self.use_random_projection
        })
        return config

class GaussianNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, stddev, training_only=True, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
        self.training_only = training_only
    
    def call(self, inputs, training=None):
        noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=self.stddev)
        return inputs + noise
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'stddev': self.stddev,
            'training_only': self.training_only
        })
        return config
    
class GaussianBlur(Layer):
    def __init__(self, kernel_size=5, sigma=1.0, **kwargs):
        super(GaussianBlur, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def build(self, input_shape):
        ax = tf.range(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, dtype=tf.float32)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2. * self.sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.expand_dims(kernel, axis=-1)
        kernel = tf.expand_dims(kernel, axis=-1)
        kernel = tf.repeat(kernel, input_shape[-1], axis=-1)
        self.kernel = tf.Variable(kernel, trainable=False)

    def call(self, x):
        return tf.nn.depthwise_conv2d(x, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

class WeightNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs * noise
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({'stddev': self.stddev})
        return config

class PrivacyPreservingUNet:
    def __init__(self, 
                 input_size=(None, None, 1), 
                 target_size=(128, 128), 
                 num_classes=4,
                 privacy_config=None):
        """
        Privacy-preserving U-Net using only native TensorFlow/Keras
        
        Args:
            privacy_config: dict with privacy settings
                - 'gradient_clipping': float, gradient clipping value (default: 1.0)
                - 'input_noise': float, input perturbation level (default: 0.01)
                - 'weight_noise': float, weight noise injection (default: 0.001)
                - 'privacy_dropout': float, enhanced dropout for privacy (default: 0.2)
                - 'feature_noise': bool, add noise to intermediate features
        """
        self.input_size = input_size
        self.target_size = target_size
        self.num_classes = num_classes
        
        # Default privacy configuration
        self.privacy_config = {
            'gradient_clipping': 1.0,
            'input_noise': 0.01,
            'weight_noise': 0.001,
            'privacy_dropout': 0.2,
            'feature_noise':0.005,
            'feature_encryption': True,
            'encryption_strength': 0.5,
        }
        
        if privacy_config:
            self.privacy_config.update(privacy_config)

        print("Privacy configuration:")
        for key, value in self.privacy_config.items():
            print(f"  {key}: {value}")
    
    def add_gaussian_noise_layer(self, x, noise_stddev=0.01, training_only=True, name="noise"):
        """Add Gaussian noise layer for privacy protection"""
        return GaussianNoiseLayer(stddev=noise_stddev, training_only=training_only, name=name)(x)
    
    def add_encryption_layer(self, x, layer_name):
        """Add appropriate encryption layer based on configuration"""
        if not self.privacy_config.get('feature_encryption', False):
            return x
        
        return FeatureEncryptionLayer(
            encryption_strength=self.privacy_config.get('encryption_strength', 0.5),
            use_random_projection=True,
            name=f"{layer_name}_encryption"
        )(x)
    
    def privacy_aware_conv_block(self, x, feature_noise, filters, block_name, dropout_rate=0.1):
        """Convolutional block with privacy-enhancing modifications"""
        # First convolution
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same', 
                     kernel_initializer='he_normal', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.001),
                     name=f"{block_name}_conv1")(x)
        
        # Add feature noise for privacy
        if feature_noise > 0:
            conv = self.add_gaussian_noise_layer(
                conv, noise_stddev=feature_noise, name=f"{block_name}_feature_noise1"
            )
        
        conv = BatchNormalization(name=f"{block_name}_bn1")(conv)
        
        # Second convolution
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same', 
                     kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001),
                     name=f"{block_name}_conv2")(conv)
        
        if self.privacy_config.get('feature_encryption', False):
            conv = self.add_encryption_layer(conv, f"{block_name}_conv2")
       
        
        if feature_noise > 0:
            conv = self.add_gaussian_noise_layer(
                conv, noise_stddev= feature_noise, name=f"{block_name}_feature_noise2"
            )
        
        conv = BatchNormalization(name=f"{block_name}_bn2")(conv)
        
        # Enhanced dropout for privacy
        privacy_dropout = max(dropout_rate, self.privacy_config.get('privacy_dropout'))
        if privacy_dropout > 0:
            conv = Dropout(privacy_dropout, name=f"{block_name}_dropout")(conv)
        
        return conv
    
    def add_weight_noise_layer(self, x, noise_stddev=0.001, name="weight_noise"):
        """Add noise to simulate weight perturbation"""
        return WeightNoiseLayer(stddev=noise_stddev, name=name)(x)
    
    def build_model(self):
        inputs = Input(self.input_size, name="input_layer")
        
        # Optional input noise for privacy
        gaussuian_blur = self.privacy_config.get('gaussian_blur', True)
        input_noise_stddev = self.privacy_config.get('input_noise', 0.01)
        drop_out_rate = self.privacy_config.get('privacy_dropout')
        add_feature_noise = self.privacy_config.get('feature_noise', False)
        feature_noise = self.privacy_config.get('feature_noise', 0.005)
        
        print(f"Adding input noise with stddev: {input_noise_stddev}, dropout rate: {drop_out_rate}, feature noise: {add_feature_noise}")

        
        x = tf.keras.layers.Resizing(self.target_size[0], self.target_size[1], name="resize")(inputs)
        x = tf.keras.layers.Rescaling(1./255, name="rescale")(x)

        use_light = self.privacy_config.get('use_light_augment', True)
        if use_light:
            print("Adding light invariance layer")
            light_layer = LightInvarianceLayer(
                brightness_delta=self.privacy_config.get('brightness_delta', 0.08),
                contrast_lower=self.privacy_config.get('contrast_lower', 0.9),
                contrast_upper=self.privacy_config.get('contrast_upper', 1.1),
                use_clahe=self.privacy_config.get('use_clahe', False),
                clahe_clip_limit=self.privacy_config.get('clahe_clip_limit', 2.0),
                clahe_tile_grid_size=self.privacy_config.get('clahe_tile_grid_size', (8, 8)),
                training_only_augment=self.privacy_config.get('light_training_only', True),
                preserve_range=True,
                name="light_invariance"
            )
            x = light_layer(x)


        if gaussuian_blur:
            x = GaussianBlur(name="gaussian_blur")(x)

        if self.privacy_config.get('input_noise', 0) > 0:
            x = self.add_gaussian_noise_layer(x, noise_stddev=input_noise_stddev , name="input_noise")
        
        # Block 1

        c1 = self.privacy_aware_conv_block(x, feature_noise= feature_noise,  filters=32, block_name="encoder_block1", 
                                          dropout_rate= drop_out_rate)
        p1 = MaxPooling2D((2, 2), name="pool1")(c1)

        # Block 2
        c2 = self.privacy_aware_conv_block(p1, feature_noise= feature_noise, filters=64, block_name="encoder_block2", 
                                          dropout_rate=drop_out_rate)
        p2 = MaxPooling2D((2, 2), name="pool2")(c2)

        # Block 3
        c3 = self.privacy_aware_conv_block(p2, feature_noise= feature_noise, filters=128, block_name="encoder_block3", 
                                          dropout_rate=drop_out_rate)
        p3 = MaxPooling2D((2, 2), name="pool3")(c3)

        # Bottleneck
        c4 = self.privacy_aware_conv_block(p3, feature_noise= feature_noise, filters=256, block_name="bottleneck", 
                                          dropout_rate=drop_out_rate)
        
        # Add weight noise to bottleneck if configured
        weight_noise = self.privacy_config.get('weight_noise', 0.001)
        
        if self.privacy_config.get('weight_noise', 0) > 0:
            c4 = self.add_weight_noise_layer(c4, noise_stddev=weight_noise, 
                                           name="bottleneck_weight_noise")
        # Block 5
        u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name="upsample5")(c4)
        u5 = concatenate([u5, c3], name="concat5")
        c5 = self.privacy_aware_conv_block(u5, feature_noise= feature_noise, filters=128, block_name="decoder_block5", 
                                          dropout_rate=drop_out_rate)

        # Block 6
        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name="upsample6")(c5)
        u6 = concatenate([u6, c2], name="concat6")
        c6 = self.privacy_aware_conv_block(u6, feature_noise= feature_noise, filters=64, block_name="decoder_block6", 
                                          dropout_rate=drop_out_rate)

        # Block 7
        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name="upsample7")(c6)
        u7 = concatenate([u7, c1], name="concat7")
        c7 = self.privacy_aware_conv_block(u7, feature_noise= feature_noise, filters=32, block_name="decoder_block7", 
                                          dropout_rate=drop_out_rate)
        

        # Output layer
        outputs = Conv2D(self.num_classes, (1, 1), activation='softmax', name="output_layer")(c7)

        model = Model(inputs, outputs, name="privacy_preserving_unet")
        return model
    @staticmethod
    def load_model(model_path):
        """
        Load a pre-trained model with custom objects
        """
        custom_objects = {
            'GaussianBlur': GaussianBlur,
            'GaussianNoiseLayer': GaussianNoiseLayer,
            'WeightNoiseLayer': WeightNoiseLayer,
            'FeatureEncryptionLayer': FeatureEncryptionLayer,
            'LightInvarianceLayer': LightInvarianceLayer,
            'dice_coefficient_multiclass': dice_coefficient_multiclass,
            'iou_metric_multiclass': iou_metric_multiclass
        }
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
       

# Custom optimizer with gradient noise and clipping
class PrivacyAwareOptimizer:
    def __init__(self, base_optimizer, gradient_noise_stddev=0.1, gradient_clip_value=1.0):
        self.base_optimizer = base_optimizer
        self.gradient_noise_stddev = gradient_noise_stddev
        self.gradient_clip_value = gradient_clip_value
    
    def get_optimizer(self):
        """Get optimizer with privacy-preserving modifications"""
        # Note: This is a conceptual implementation
        # In practice, you would need to implement custom training loop
        return self.base_optimizer

# Custom training function with privacy techniques
def train_with_privacy_techniques(model, train_data, val_data, privacy_config, 
                                 epochs=100, batch_size=32):
    """
    Custom training function with privacy-preserving techniques
    """

    img_train, label_train = train_data
    
    # Compile model with privacy-aware settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        clipnorm=privacy_config.get('gradient_clipping', 1.0)  # Gradient clipping
    )


    model.compile(
        optimizer=optimizer,
        loss= 'sparse_categorical_crossentropy',
        metrics=['accuracy', dice_coefficient_multiclass, iou_metric_multiclass]
    )
    
    # Custom callbacks for privacy
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    ]
    
    
    print("Privacy-preserving training configuration:")
    for key, value in privacy_config.items():
        print(f"  {key}: {value}")
    
    history = model.fit(
        img_train, label_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    
    return history

def dice_coefficient_multiclass(y_true, y_pred, smooth=1e-6):
        """Dice coefficient for multi-class segmentation with sparse labels"""
        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) != len(y_pred.shape):
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
            y_true = tf.cast(y_true, tf.float32)
        
        # Flatten tensors
        y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return tf.reduce_mean(dice)

def dice_loss_multiclass(y_true, y_pred, smooth=1e-6):
    """Dice loss for multi-class segmentation"""
    return 1 - dice_coefficient_multiclass(y_true, y_pred, smooth)

# IoU Metric for multi-class
def iou_metric_multiclass(y_true, y_pred, smooth=1e-6):
    # Convert predictions to class labels
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(tf.cast(y_true_f == y_pred_f, tf.float32))
    union = tf.cast(tf.size(y_true_f, out_type=tf.int32), tf.float32)

    return (intersection + smooth) / (union + smooth)


# Improved Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    else:
        return lr * 0.1

# Factory function for different privacy levels
