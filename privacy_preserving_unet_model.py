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
            trainable=True
        )
        
        # Create bias for additional obfuscation
        self.encryption_bias = self.add_weight(
            name='encryption_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        
        if self.use_random_projection:
            # Fixed random projection matrix (not trainable)
            self.random_projection = self.add_weight(
                name='random_projection',
                shape=(input_shape[-1], input_shape[-1]),
                initializer='random_normal',
                trainable=False
            )
    def call(self, inputs, training=None):
        # Apply learnable transformation
        encrypted = tf.matmul(inputs, self.encryption_matrix) + self.encryption_bias
        
        # Apply random projection if enabled
        if self.use_random_projection:
            encrypted = tf.matmul(encrypted, self.random_projection)
        
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
        if self.training_only and not training:
            return inputs
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
    
    def call(self, inputs):
        noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=self.stddev)
        return inputs + noise
    
    def get_config(self):
        config = super().get_config()
        config.update({'stddev': self.stddev})
        return config

class MixupLayer(tf.keras.layers.Layer):
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        batch_size = tf.shape(inputs)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_inputs = tf.gather(inputs, indices)
        
        # Generate lambda from uniform distribution (approximating Beta)
        lam = tf.random.uniform([], 0, self.alpha)
        mixed_inputs = lam * inputs + (1 - lam) * shuffled_inputs
        
        return mixed_inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha})
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
                - 'gradient_noise': float, noise level for gradient perturbation (default: 0.1)
                - 'gradient_clipping': float, gradient clipping value (default: 1.0)
                - 'input_noise': float, input perturbation level (default: 0.01)
                - 'weight_noise': float, weight noise injection (default: 0.001)
                - 'privacy_dropout': float, enhanced dropout for privacy (default: 0.2)
                - 'label_smoothing': float, label smoothing for privacy (default: 0.1)
                - 'mixup_alpha': float, mixup augmentation parameter (default: 0.2)
                - 'feature_noise': bool, add noise to intermediate features
        """
        self.input_size = input_size
        self.target_size = target_size
        self.num_classes = num_classes
        
        # Default privacy configuration
        self.privacy_config = {
            'gradient_noise': 0.1,
            'gradient_clipping': 1.0,
            'input_noise': 0.01,
            'weight_noise': 0.001,
            'privacy_dropout': 0.2,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2,
            'feature_noise': True
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
    
    def privacy_aware_conv_block(self, x, noise_stddev, filters, block_name, dropout_rate=0.1, add_feature_noise=False):
        """Convolutional block with privacy-enhancing modifications"""
        # First convolution
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same', 
                     kernel_initializer='he_normal', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.001),
                     name=f"{block_name}_conv1")(x)
        
        # Add feature noise for privacy
        if add_feature_noise:
            conv = self.add_gaussian_noise_layer(
                conv, noise_stddev=noise_stddev, name=f"{block_name}_feature_noise1"
            )
        
        conv = BatchNormalization(name=f"{block_name}_bn1")(conv)
        
        # Second convolution
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same', 
                     kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001),
                     name=f"{block_name}_conv2")(conv)
        
        conv = self.add_encryption_layer(conv, f"{block_name}_conv2")
        
        if add_feature_noise and self.privacy_config.get('feature_noise', False):
            conv = self.add_gaussian_noise_layer(
                conv, noise_stddev=0.005, name=f"{block_name}_feature_noise2"
            )
        
        conv = BatchNormalization(name=f"{block_name}_bn2")(conv)
        
        # Enhanced dropout for privacy
        privacy_dropout = max(dropout_rate, self.privacy_config.get('privacy_dropout', 0.2))
        conv = Dropout(privacy_dropout, name=f"{block_name}_dropout")(conv)
        
        return conv
    
    def add_weight_noise_layer(self, x, noise_stddev=0.001, name="weight_noise"):
        """Add noise to simulate weight perturbation"""
        return WeightNoiseLayer(stddev=noise_stddev, name=name)(x)
    
    def mixup_layer(self, x, y=None, alpha=0.2, name="mixup"):
        """Mixup augmentation for privacy protection"""
        return MixupLayer(alpha=alpha, name=name)(x)

    
    def build_model(self):
        inputs = Input(self.input_size, name="input_layer")
    
        # Preprocessing
        x = tf.keras.layers.Resizing(self.target_size[0], self.target_size[1], name="resize")(inputs)
        x = tf.keras.layers.Rescaling(1./255, name="rescale")(x)
        
        # Optional input noise for privacy
        noise_stddev = self.privacy_config.get('input_noise', 0.01)
        drop_out_rate = max(self.privacy_config.get('privacy_dropout', 0.2), 0.1)
        add_feature_noise = self.privacy_config.get('feature_noise', False)
        
        print(f"Adding input noise with stddev: {noise_stddev}, dropout rate: {drop_out_rate}, feature noise: {add_feature_noise}")

        x = GaussianBlur(name="gaussian_blur")(x)

        if self.privacy_config.get('input_noise', 0) > 0:
            x = self.add_gaussian_noise_layer(x, noise_stddev=noise_stddev , name="input_noise")
        
        # Block 1

        c1 = self.privacy_aware_conv_block(x, noise_stddev= noise_stddev,  filters=32, block_name="encoder_block1", 
                                          dropout_rate= drop_out_rate, add_feature_noise=add_feature_noise)
        p1 = MaxPooling2D((2, 2), name="pool1")(c1)

        # Block 2
        c2 = self.privacy_aware_conv_block(p1, noise_stddev= noise_stddev, filters=64, block_name="encoder_block2", 
                                          dropout_rate=drop_out_rate, add_feature_noise=add_feature_noise)
        p2 = MaxPooling2D((2, 2), name="pool2")(c2)

        # Block 3
        c3 = self.privacy_aware_conv_block(p2, noise_stddev= noise_stddev, filters=128, block_name="encoder_block3", 
                                          dropout_rate=drop_out_rate, add_feature_noise=add_feature_noise)
        p3 = MaxPooling2D((2, 2), name="pool3")(c3)

        # Bottleneck
        c4 = self.privacy_aware_conv_block(p3, noise_stddev= noise_stddev, filters=256, block_name="bottleneck", 
                                          dropout_rate=drop_out_rate, add_feature_noise=add_feature_noise)
        
        # Add weight noise to bottleneck if configured
        weight_noise = self.privacy_config.get('weight_noise', 0.001)

        if self.privacy_config.get('weight_noise', 0) > 0:
            c4 = self.add_weight_noise_layer(c4, noise_stddev=weight_noise, 
                                           name="bottleneck_weight_noise")

        # Decoder Path using privacy-aware blocks
        
        # Block 5
        u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name="upsample5")(c4)
        u5 = concatenate([u5, c3], name="concat5")
        c5 = self.privacy_aware_conv_block(u5, noise_stddev= noise_stddev, filters=128, block_name="decoder_block5", 
                                          dropout_rate=drop_out_rate, add_feature_noise=add_feature_noise)

        # Block 6
        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name="upsample6")(c5)
        u6 = concatenate([u6, c2], name="concat6")
        c6 = self.privacy_aware_conv_block(u6, noise_stddev= noise_stddev, filters=64, block_name="decoder_block6", 
                                          dropout_rate=drop_out_rate, add_feature_noise=add_feature_noise)

        # Block 7
        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name="upsample7")(c6)
        u7 = concatenate([u7, c1], name="concat7")
        c7 = self.privacy_aware_conv_block(u7, noise_stddev= noise_stddev, filters=32, block_name="decoder_block7", 
                                          dropout_rate=drop_out_rate, add_feature_noise=add_feature_noise)

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
            'MixupLayer': MixupLayer,
            'FeatureEncryptionLayer': FeatureEncryptionLayer,
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
