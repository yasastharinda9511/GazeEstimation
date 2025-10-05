import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
import warnings

from vitsModel.PrivacyVitsModel import PrivacyVitsModel

class Pipeline:
    @staticmethod
    def complete_training_pipeline(model, callbacks, images, labels, epochs=15, batch_size=4, 
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
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Create validation dataset (no augmentation)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
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
                visualize_predictions_enhanced(model, X_val, y_val, pred_masks, num_samples=min(4, len(X_val)))
                print("✅ Visualizations saved!")
            except Exception as e:
                print(f"⚠️  Visualization failed: {str(e)}")
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return model, history, eval_results