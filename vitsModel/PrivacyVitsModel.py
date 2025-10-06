import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from vitsModel.customLayers.PatchExtract import PatchExtract
from vitsModel.customLayers.PatchEmbeddingLayer import PatchEmbedding
from vitsModel.customLayers.TransformerBlock import TransformerBlock
from vitsModel.customLayers.UpSampleBlock import UpsampleBlock
from vitsModel.customLayers.GetItem import GetItem
from vitsModel.customLayers.PrivacyFeatureExtractor import PrivacyAwareFeatureExtractor
from vitsModel.customLayers.DefferentialPrivacyLayer import DifferentialPrivacyLayer
from vitsModel.customLayers.AdaptivePrivacyPatchDrop import AdaptivePrivacyPatchDrop
from vitsModel.customLayers.MultiHeadAttentionLayer import MultiHeadAttention

class PrivacyVitsModel:

    def create_enhanced_vit_eye_segmentation_model(self,
        image_size=224,
        patch_size=16,
        num_patches=196,
        projection_dim=768,
        num_heads=12,
        transformer_units=[3072, 768],
        transformer_layers=8,
        num_classes=4,
        dropout_rate=0.1,
        input_channels=1,
        use_attention_in_decoder=True,
        # Privacy parameters
        base_drop_rate=0.1,
        sensitivity_strength=0.5,
        dp_clip_norm=1.0,
        anonymization_strength=0.3,
        differential_privacy_noise=0.05
    ):
        
        inputs = layers.Input(shape=(image_size, image_size, input_channels))
        
        # Create patches
        patches = PatchExtract(patch_size)(inputs)
        
        # Encode patches (now includes class token)
        encoded_patches = PatchEmbedding(num_patches, projection_dim)(patches)

        encoded_patches = AdaptivePrivacyPatchDrop(
            base_drop_rate=base_drop_rate, 
            sensitivity_strength= sensitivity_strength)(encoded_patches, training=True)
        

        if(anonymization_strength > 0.0):
            encoded_patches = PrivacyAwareFeatureExtractor(
            projection_dim=projection_dim,
            anonymization_strength=anonymization_strength,
            name="privacy_aware_feature_extractor"
            )(encoded_patches)
        
        

        for _ in range(transformer_layers):
            encoded_patches = TransformerBlock(
                projection_dim, num_heads, transformer_units[0], dropout_rate
            )(encoded_patches)
        
        # Layer normalization
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        privacy_protected_features = DifferentialPrivacyLayer(
            noise_scale=differential_privacy_noise,
            clip_norm=dp_clip_norm,
            name="differential_privacy_layer"
        )(representation)

        representation = GetItem(slice(1, None))(privacy_protected_features)
        
        # Reshape for decoder
        h = w = int(np.sqrt(num_patches))
        representation = layers.Reshape((h, w, projection_dim))(representation)
        
        # Enhanced decoder with optional attention
        x = UpsampleBlock(512, use_attention=use_attention_in_decoder)(representation)  # 14x14 -> 28x28
        x = UpsampleBlock(256, use_attention=use_attention_in_decoder)(x)               # 28x28 -> 56x56
        x = UpsampleBlock(128, use_attention=use_attention_in_decoder)(x)               # 56x56 -> 112x112
        x = UpsampleBlock(64, use_attention=use_attention_in_decoder)(x)                # 112x112 -> 224x224
        
        # Final segmentation layer with improved initialization
        segmentation_output = layers.Conv2D(
            num_classes, 1, 
            activation='softmax', 
            name='segmentation',
            kernel_initializer='he_normal'
        )(x)
        
        model = keras.Model(inputs, segmentation_output, name="Enhanced_ViT_Eye_Segmentation")
        return model


    # Enhanced loss functions
    def dice_coefficient_enhanced(self, y_true, y_pred, smooth=1e-6, class_weights=None):
        """Enhanced dice coefficient with class weighting"""
        # Convert sparse labels to one-hot
        if len(y_true.shape) != len(y_pred.shape):
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
            y_true = tf.cast(y_true, tf.float32)
        
        # Flatten tensors
        y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        # Calculate intersection and union for each class
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
        union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
        dice_per_class = (2. * intersection + smooth) / (union + smooth)
        
        # Apply class weights if provided
        if class_weights is not None:
            class_weights = tf.constant(class_weights, dtype=tf.float32)
            dice_per_class = dice_per_class * class_weights
        
        return tf.reduce_mean(dice_per_class)

    # Enhanced data loading with better augmentation
    def enhanced_data_augmentation(self):
        """Create enhanced data augmentation pipeline"""
        def augment_data(image, label):
            # Random horizontal flip
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                label = tf.image.flip_left_right(tf.expand_dims(label, -1))
                label = tf.squeeze(label, -1)
            
            # Random rotation (small angles)
            if tf.random.uniform(()) > 0.7:
                angle = tf.random.uniform((), -0.1, 0.1)  # ±5.7 degrees
                image = tf.image.rot90(image, k=tf.cast(angle * 4, tf.int32))
                label = tf.image.rot90(tf.expand_dims(label, -1), k=tf.cast(angle * 4, tf.int32))
                label = tf.squeeze(label, -1)
            
            # Random brightness and contrast
            if tf.random.uniform(()) > 0.5:
                image = tf.image.random_brightness(image, 0.15)
            if tf.random.uniform(()) > 0.5:
                image = tf.image.random_contrast(image, 0.85, 1.15)
            
            # Random noise
            if tf.random.uniform(()) > 0.8:
                noise = tf.random.normal(tf.shape(image), mean=0, stddev=0.02)
                image = tf.clip_by_value(image + noise, 0.0, 1.0)
            
            # Ensure image is in [0,1] range
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            return image, label
        
        return augment_data

    # Training configuration
    def get_optimized_training_config(self):
        """Get optimized training configuration"""
        
        # Learning rate schedule
        def lr_schedule(epoch):
            """Custom learning rate schedule"""
            if epoch < 5:
                return 1e-4
            elif epoch < 15:
                return 5e-5
            elif epoch < 25:
                return 1e-5
            else:
                return 5e-6
        
        # Optimizer with weight decay
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=0.01,
            clipnorm=1.0
        )
        
        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                'best_enhanced_vit_eye_model.h5',
                monitor='val_dice_coefficient_enhanced',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_dice_coefficient_enhanced',
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return optimizer, callbacks

    # Example usage WITHOUT privacy layers (your original model)
    def create_and_compile_enhanced_model(self):
        """Create and compile the enhanced model - original functionality"""
        
        model = self.create_enhanced_vit_eye_segmentation_model(
            image_size=224,
            patch_size=16,
            num_patches=196,
            projection_dim=768,
            num_heads=12,
            transformer_layers=8,
            num_classes=4,
            input_channels=1,
            dropout_rate=0.15,
            use_attention_in_decoder=True,
            # Disable privacy properties
            base_drop_rate = 0.0 ,
            sensitivity_strength= 0.0 ,
            dp_clip_norm= 0.0,
            anonymization_strength= 0.0,
            differential_privacy_noise= 0.0
        )
        
        optimizer, callbacks = self.get_optimized_training_config()
        
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy', self.dice_coefficient_enhanced]
        )
        
        return model, callbacks

    # Example usage WITH privacy layers
    def create_and_compile_model_with_privacy(self , base_drop_rate=0.1,
        sensitivity_strength=0.5,
        dp_clip_norm=1.0,
        anonymization_strength=0.3,
        differential_privacy_noise=0.05):
        """Create and compile the model with privacy-preserving layers"""
        
        model = self.create_enhanced_vit_eye_segmentation_model(
            image_size=224,
            patch_size=16,
            num_patches=196,
            projection_dim=768,
            num_heads=12,
            transformer_layers=8,
            num_classes=4,
            input_channels=1,
            dropout_rate=0.15,
            use_attention_in_decoder=True,
            # Enable privacy Properties
            base_drop_rate = base_drop_rate ,
            sensitivity_strength= sensitivity_strength ,
            dp_clip_norm= dp_clip_norm,
            anonymization_strength= anonymization_strength,
            differential_privacy_noise= differential_privacy_noise
        )
        
        optimizer, callbacks = self.get_optimized_training_config()
        
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy', self.dice_coefficient_enhanced]
        )
        
        return model, callbacks

    # Function to extract privacy features after training
    def extract_privacy_features(self, model, images):
        """Extract privacy-preserving features from the trained model"""
        
        # Create a new model that outputs the privacy features
        privacy_feature_layer = None
        for layer in model.layers:
            if 'privacy_features_output' in layer.name:
                privacy_feature_layer = layer
                break
        
        if privacy_feature_layer is not None:
            # Create extractor model
            feature_extractor = keras.Model(
                inputs=model.input,
                outputs=privacy_feature_layer.output
            )
            privacy_features = feature_extractor.predict(images)
            return privacy_features
        else:
            print("Privacy layers not found in the model. Set add_privacy_layers=True when creating the model.")
            return None

    # Enhanced visualization
    def visualize_predictions_enhanced(self, model, images, labels, num_samples=4):
        """Enhanced visualization of model predictions"""
        
        # Make predictions
        predictions = model.predict(images[:num_samples])
        pred_masks = np.argmax(predictions, axis=-1)
        
        # Color mapping for classes
        colors = np.array([
            [0, 0, 0],        # Background - Black
            [255, 255, 255],  # Sclera - White  
            [0, 255, 0],      # Iris - Green
            [255, 0, 0]       # Pupil - Red
        ])
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        
        for i in range(num_samples):
            # Original image
            img = images[i].squeeze() if len(images[i].shape) == 3 else images[i]
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth
            gt_colored = colors[labels[i]]
            axes[i, 1].imshow(gt_colored)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            pred_colored = colors[pred_masks[i]]
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Overlay
            img_rgb = np.stack([img, img, img], axis=-1) if len(img.shape) == 2 else img
            if img_rgb.max() <= 1.0:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            
            overlay = cv2.addWeighted(
                img_rgb.astype(np.uint8), 0.7, 
                pred_colored.astype(np.uint8), 0.3, 0
            )
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('enhanced_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def complete_training_pipeline(self, model, images, labels, epochs=15, batch_size=4, 
                             test_size=0.2, enhanced_augmentation=True, 
                             verbose_evaluation=True):
    
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, 
            test_size=test_size, 
            random_state=42
        )

        
        def enhanced_augment_data(image, label):
            """Enhanced data augmentation"""
            # Random horizontal flip
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                label = tf.image.flip_left_right(tf.expand_dims(label, -1))
                label = tf.squeeze(label, -1)
            
            if enhanced_augmentation:
                # Random brightness and contrast
                if tf.random.uniform(()) > 0.6:
                    image = tf.image.random_brightness(image, 0.15)
                if tf.random.uniform(()) > 0.6:
                    image = tf.image.random_contrast(image, 0.85, 1.15)
                
                # Random noise
                if tf.random.uniform(()) > 0.8:
                    noise = tf.random.normal(tf.shape(image), mean=0, stddev=0.02)
                    image = tf.clip_by_value(image + noise, 0.0, 1.0)
                
                # Random gamma correction
                if tf.random.uniform(()) > 0.7:
                    gamma = tf.random.uniform((), 0.8, 1.2)
                    image = tf.pow(image, gamma)
            
            # Ensure image is in [0,1] range
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            return image, label
        
        # Create training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=min(1000, len(X_train)), seed=42)
        # train_dataset = train_dataset.map(enhanced_augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset (no augmentation)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        _, callbacks = self.create_and_compile_enhanced_model()  # Just to get callbacks
        
        # Update the checkpoint path to be unique for this batch
        for callback in callbacks:
            if isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                callback.filepath = f'best_enhanced_vit_eye_model_batch_{step if "step" in globals() else "current"}.h5'
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        def evaluate_model_enhanced(model, X_val, y_val, batch_size=4):
            """Enhanced model evaluation with detailed metrics"""
            print("🔍 Making predictions on validation set...")
            
            # Make predictions
            predictions = model.predict(X_val, batch_size=batch_size, verbose=1)
            pred_masks = np.argmax(predictions, axis=-1)
            
            # Calculate metrics per class
            num_classes = 4
            class_names = ['Background', 'Sclera', 'Iris', 'Pupil']
            
            print("\n📊 Per-class Dice Coefficients:")
            class_dice_scores = []
            
            for i in range(num_classes):
                true_class = (y_val == i).astype(np.float32)
                pred_class = (pred_masks == i).astype(np.float32)
                
                intersection = np.sum(true_class * pred_class)
                union = np.sum(true_class) + np.sum(pred_class)
                
                if union == 0:
                    dice = 1.0 if intersection == 0 else 0.0
                else:
                    dice = (2. * intersection) / union
                
                class_dice_scores.append(dice)
                print(f"  {class_names[i]:<12}: {dice:.4f}")
            
            # Overall metrics
            mean_dice = np.mean(class_dice_scores)
            accuracy = np.mean(pred_masks == y_val)
            
            print(f"\n📊 Overall Metrics:")
            print(f"  Mean Dice:     {mean_dice:.4f}")
            print(f"  Pixel Accuracy: {accuracy:.4f}")
            
            # Class distribution in predictions vs ground truth
            print(f"\n📊 Class Distribution Comparison:")
            total_pixels = y_val.size
            
            for i, class_name in enumerate(class_names):
                true_pct = (np.sum(y_val == i) / total_pixels) * 100
                pred_pct = (np.sum(pred_masks == i) / total_pixels) * 100
                print(f"  {class_name:<12}: True {true_pct:5.1f}% | Pred {pred_pct:5.1f}%")
            
            return pred_masks, {
                'class_dice_scores': class_dice_scores,
                'mean_dice': mean_dice,
                'accuracy': accuracy
            }
        
        if verbose_evaluation:
            pred_masks, eval_metrics = evaluate_model_enhanced(model, X_val, y_val, batch_size)
            eval_results = (X_val, y_val, pred_masks, eval_metrics)
        else:
            # Quick evaluation
            val_loss, val_acc, val_dice = model.evaluate(val_dataset, verbose=0)
            print(f"✅ Quick evaluation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Dice: {val_dice:.4f}")
            eval_results = {'val_loss': val_loss, 'val_accuracy': val_acc, 'val_dice': val_dice}
        
        # ============================================================================
        # 5. VISUALIZATION
        # ============================================================================
        
        if verbose_evaluation and len(X_val) >= 4:
            print("\n🎨 Creating visualizations...")
            
            try:
                self.visualize_predictions_enhanced(model, X_val, y_val, pred_masks, num_samples=min(4, len(X_val)))
                print("✅ Visualizations saved!")
            except Exception as e:
                print(f"⚠️  Visualization failed: {str(e)}")
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return model, history, eval_results
    
    @staticmethod
    def load_model(model_path):
        """Load a saved model from disk"""

        def dice_coefficient_enhanced(y_true, y_pred, smooth=1e-6, class_weights=None):
            # Convert sparse labels to one-hot
            if len(y_true.shape) != len(y_pred.shape):
                y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
                y_true = tf.cast(y_true, tf.float32)
            
            # Flatten tensors
            y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
            y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
            
            # Calculate intersection and union for each class
            intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
            union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
            dice_per_class = (2. * intersection + smooth) / (union + smooth)
            
            # Apply class weights if provided
            if class_weights is not None:
                class_weights = tf.constant(class_weights, dtype=tf.float32)
                dice_per_class = dice_per_class * class_weights
            
            return tf.reduce_mean(dice_per_class)

        if os.path.exists(model_path):
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'dice_coefficient_enhanced': dice_coefficient_enhanced,
                    'AdaptivePrivacyPatchDrop': AdaptivePrivacyPatchDrop,
                    'PrivacyAwareFeatureExtractor': PrivacyAwareFeatureExtractor,
                    'DifferentialPrivacyLayer': DifferentialPrivacyLayer,
                    'PatchExtract': PatchExtract,
                    'PatchEmbedding': PatchEmbedding,
                    'TransformerBlock': TransformerBlock,
                    'UpsampleBlock': UpsampleBlock,
                    'GetItem': GetItem,
                    'MultiHeadAttention': MultiHeadAttention
                }
            )
            print(f"✅ Model loaded from {model_path}")
            return model
        else:
            print(f"❌ Model path {model_path} does not exist.")
            return None