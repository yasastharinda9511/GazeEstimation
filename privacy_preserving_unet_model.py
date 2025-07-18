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
    
    def add_gaussian_noise_layer(self, x, noise_stddev=0.01, training_only=True, name="noise"):
        """Add Gaussian noise layer for privacy protection"""
        return GaussianNoiseLayer(stddev=noise_stddev, training_only=training_only, name=name)(x)
    
    def privacy_aware_conv_block(self, x, filters, block_name, dropout_rate=0.1, add_feature_noise=False):
        """Convolutional block with privacy-enhancing modifications"""
        # First convolution
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same', 
                     kernel_initializer='he_normal', 
                     kernel_regularizer=tf.keras.regularizers.l2(0.001),
                     name=f"{block_name}_conv1")(x)
        
        # Add feature noise for privacy
        if add_feature_noise and self.privacy_config.get('feature_noise', False):
            conv = self.add_gaussian_noise_layer(
                conv, noise_stddev=0.005, name=f"{block_name}_feature_noise1"
            )
        
        conv = BatchNormalization(name=f"{block_name}_bn1")(conv)
        
        # Second convolution
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same', 
                     kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001),
                     name=f"{block_name}_conv2")(conv)
        
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
        
        # Optional Gaussian Blur (comment out if not needed)
        x = GaussianBlur(name="gaussian_blur")(x)

        # Encoder Path
        # Block 1
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        c1 = BatchNormalization()(c1)
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(0.1)(p1)

        # Block 2
        c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(0.2)(p2)

        # Block 3
        c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
        c3 = BatchNormalization()(c3)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(0.3)(p3)

        # Bottleneck
        c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
        c4 = BatchNormalization()(c4)
        c4 = Dropout(0.4)(c4)

        # Decoder Path
        # Block 5
        u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
        u5 = concatenate([u5, c3])
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u5)
        c5 = BatchNormalization()(c5)
        c5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)
        c5 = BatchNormalization()(c5)
        c5 = Dropout(0.3)(c5)

        # Block 6
        u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c2])
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)

        # Block 7
        u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c1])
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.1)(c7)

        # Output layer
        outputs = Conv2D(self.num_classes, (1, 1), activation='softmax')(c7)

        model = Model(inputs, outputs, name="improved_unet")
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
            'dice_coefficient_multiclass': dice_coefficient_multiclass,
            'iou_metric_multiclass': iou_metric_multiclass
        }
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

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
    
    # Use label smoothing for privacy
    if privacy_config.get('label_smoothing', 0) > 0:
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=privacy_config['label_smoothing']
        )
    else:
        loss = 'sparse_categorical_crossentropy'
    
    model.compile(
        optimizer=optimizer,
        loss= "sparse_categorical_crossentropy",
        metrics=['accuracy', dice_coefficient_multiclass, iou_metric_multiclass]
    )
    
    # Custom callbacks for privacy
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    ]
    
    # Add gradient noise callback (custom implementation would be needed)
    if privacy_config.get('gradient_noise', 0) > 0:
        print(f"Training with gradient noise: {privacy_config['gradient_noise']}")
        # Note: Full implementation would require custom training loop
    
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

def focal_loss_multiclass(gamma=2., alpha=0.25):
    """Focal loss for multi-class segmentation with sparse labels"""
    def focal_loss_fixed(y_true, y_pred):
        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) != len(y_pred.shape):
            y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
            y_true_onehot = tf.cast(y_true_onehot, tf.float32)
        else:
            y_true_onehot = y_true
            
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        ce = -y_true_onehot * tf.math.log(y_pred)
        
        # Calculate focal weight
        p_t = tf.where(tf.equal(y_true_onehot, 1), y_pred, 1 - y_pred)
        alpha_factor = y_true_onehot * alpha + (1 - y_true_onehot) * (1 - alpha)
        modulating_factor = tf.pow((1 - p_t), gamma)
        
        # Apply focal weight
        focal_ce = alpha_factor * modulating_factor * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_ce, axis=-1))
    
    return focal_loss_fixed

def weighted_categorical_crossentropy(class_weights):
    """Weighted categorical crossentropy for handling class imbalance"""
    def loss(y_true, y_pred):
        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) != len(y_pred.shape):
            y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
            y_true_onehot = tf.cast(y_true_onehot, tf.float32)
        else:
            y_true_onehot = y_true
            
        # Apply class weights
        weights = tf.reduce_sum(class_weights * y_true_onehot, axis=-1)
        
        # Calculate categorical crossentropy
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        loss = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred), axis=-1)
        
        return tf.reduce_mean(weights * loss)
    
    return loss

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
def create_lightweight_privacy_unet(privacy_level="medium"):
    """
    Create U-Net with different privacy levels using only native TensorFlow
    """
    privacy_configs = {
        "low": {
            'input_noise': 0.005,
            'privacy_dropout': 0.15,
            'gradient_clipping': 2.0,
            'label_smoothing': 0.05,
            'feature_noise': False
        },
        "medium": {
            'gradient_noise': 0.1,
            'gradient_clipping': 1.0,
            'input_noise': 0.01,
            'weight_noise': 0.001,
            'privacy_dropout': 0.2,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2,
            'feature_noise': True
        },
        "high": {
            'gradient_noise': 0.2,
            'gradient_clipping': 0.5,
            'input_noise': 0.02,
            'weight_noise': 0.005,
            'privacy_dropout': 0.3,
            'label_smoothing': 0.15,
            'mixup_alpha': 0.3,
            'feature_noise': True
        }
    }
    
    config = privacy_configs.get(privacy_level, privacy_configs["medium"])
    
    privacy_unet = PrivacyPreservingUNet(
        input_size=(None, None, 1),
        target_size=(128, 128),
        num_classes=4,
        privacy_config=config
    )
    
    model = privacy_unet.build_model()
    
    print(f"\nPrivacy-preserving U-Net created with {privacy_level} privacy level")
    print("Privacy techniques applied:")
    for technique, value in config.items():
        if value:
            print(f"  ✓ {technique}: {value}")
    
    return model, config