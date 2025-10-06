import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
try:
    from skimage.metrics import structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️  scikit-image not found. Install with: pip install scikit-image")
import cv2

class ModelInversionAttack:
    """
    Model Inversion Attack to test privacy preservation of eye segmentation models.
    
    This attack attempts to reconstruct original training images from:
    1. Model predictions (segmentation masks)
    2. Model gradients
    3. Intermediate feature representations
    """
    
    def __init__(self, model, image_size=224, num_classes=4):
        """
        Initialize the attack.
        
        Args:
            model: Trained Keras model to attack
            image_size: Size of input images
            num_classes: Number of segmentation classes
        """
        self.model = model
        self.image_size = image_size
        self.num_classes = num_classes
        
    def gradient_based_reconstruction(self, target_mask, num_iterations=500, 
                                     learning_rate=0.1, l2_reg=0.01):
        """
        Attempt to reconstruct an image from a target segmentation mask using gradients.
        
        This simulates an attacker who has access to:
        - The model
        - A target segmentation mask (output)
        - Gradient information
        
        Args:
            target_mask: Target segmentation mask (H, W) with class labels
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for reconstruction
            l2_reg: L2 regularization strength
            
        Returns:
            reconstructed_image: Reconstructed image
            loss_history: Loss values during optimization
        """
        print(f"\n🔓 Starting Gradient-Based Reconstruction Attack...")
        print(f"   Iterations: {num_iterations}, LR: {learning_rate}")
        
        # Initialize random image
        reconstructed_img = tf.Variable(
            tf.random.normal([1, self.image_size, self.image_size, 1], 
                           mean=0.5, stddev=0.1),
            trainable=True
        )
        
        # Convert target mask to one-hot
        target_mask_onehot = tf.one_hot(target_mask, depth=self.num_classes)
        target_mask_onehot = tf.expand_dims(target_mask_onehot, 0)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        loss_history = []
        
        for iteration in range(num_iterations):
            with tf.GradientTape() as tape:
                # Forward pass
                pred_mask = self.model(reconstructed_img, training=False)
                
                # Segmentation loss (cross-entropy)
                seg_loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(
                        target_mask_onehot, pred_mask
                    )
                )
                
                # L2 regularization (natural image prior)
                l2_loss = l2_reg * tf.reduce_mean(tf.square(reconstructed_img))
                
                # Total variation loss (smoothness prior)
                tv_loss = 0.001 * (
                    tf.reduce_mean(tf.abs(reconstructed_img[:, :-1, :, :] - reconstructed_img[:, 1:, :, :])) +
                    tf.reduce_mean(tf.abs(reconstructed_img[:, :, :-1, :] - reconstructed_img[:, :, 1:, :]))
                )
                
                total_loss = seg_loss + l2_loss + tv_loss
            
            # Compute gradients and update
            gradients = tape.gradient(total_loss, [reconstructed_img])
            optimizer.apply_gradients(zip(gradients, [reconstructed_img]))
            
            # Clip to valid range
            reconstructed_img.assign(tf.clip_by_value(reconstructed_img, 0.0, 1.0))
            
            loss_history.append(total_loss.numpy())
            
            if iteration % 100 == 0:
                print(f"   Iteration {iteration:3d}/{num_iterations}: Loss = {total_loss.numpy():.4f}")
        
        return reconstructed_img.numpy()[0], loss_history
    
    def feature_based_reconstruction(self, target_image, layer_name=None, 
                                    num_iterations=300, learning_rate=0.1):
        """
        Attempt to reconstruct an image by matching intermediate features.
        
        This attack tries to invert the learned representations.
        
        Args:
            target_image: Original image to extract features from
            layer_name: Name of layer to match features (if None, uses last conv layer)
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate
            
        Returns:
            reconstructed_image: Reconstructed image
            loss_history: Loss values during optimization
        """
        print(f"\n🔓 Starting Feature-Based Reconstruction Attack...")
        
        # Find target layer
        if layer_name is None:
            # Find last convolutional or dense layer before output
            for layer in reversed(self.model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                    layer_name = layer.name
                    break
        
        print(f"   Target layer: {layer_name}")
        
        # Create feature extractor
        try:
            feature_layer = self.model.get_layer(layer_name)
            feature_extractor = tf.keras.Model(
                inputs=self.model.input,
                outputs=feature_layer.output
            )
        except:
            print(f"   ⚠️  Layer {layer_name} not found. Using model output.")
            feature_extractor = self.model
        
        # Extract target features
        target_features = feature_extractor(target_image, training=False)
        
        # Initialize random image
        reconstructed_img = tf.Variable(
            tf.random.normal([1, self.image_size, self.image_size, 1], 
                           mean=0.5, stddev=0.1),
            trainable=True
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_history = []
        
        for iteration in range(num_iterations):
            with tf.GradientTape() as tape:
                # Extract features from reconstructed image
                recon_features = feature_extractor(reconstructed_img, training=False)
                
                # Feature matching loss
                feature_loss = tf.reduce_mean(tf.square(recon_features - target_features))
                
                # Natural image priors
                tv_loss = 0.001 * (
                    tf.reduce_mean(tf.abs(reconstructed_img[:, :-1, :, :] - reconstructed_img[:, 1:, :, :])) +
                    tf.reduce_mean(tf.abs(reconstructed_img[:, :, :-1, :] - reconstructed_img[:, :, 1:, :]))
                )
                
                total_loss = feature_loss + tv_loss
            
            gradients = tape.gradient(total_loss, [reconstructed_img])
            optimizer.apply_gradients(zip(gradients, [reconstructed_img]))
            reconstructed_img.assign(tf.clip_by_value(reconstructed_img, 0.0, 1.0))
            
            loss_history.append(total_loss.numpy())
            
            if iteration % 50 == 0:
                print(f"   Iteration {iteration:3d}/{num_iterations}: Loss = {total_loss.numpy():.4f}")
        
        return reconstructed_img.numpy()[0], loss_history
    
    def membership_inference_attack(self, model, train_images, train_labels, 
                                   test_images, test_labels, threshold=None):
        """
        Membership Inference Attack: Determine if an image was in training set.
        
        This attack exploits the fact that models tend to be more confident
        on training data than test data.
        
        Args:
            model: Trained model
            train_images: Images from training set
            train_labels: Labels from training set
            test_images: Images NOT from training set
            test_labels: Labels from test set
            threshold: Confidence threshold (auto-computed if None)
            
        Returns:
            Dictionary with attack results
        """
        print(f"\n🔓 Starting Membership Inference Attack...")
        
        # Get predictions and confidence scores
        train_preds = model.predict(train_images, verbose=0)
        test_preds = model.predict(test_images, verbose=0)
        
        # Calculate confidence: average probability of correct class
        def get_confidence(predictions, labels):
            confidences = []
            for pred, label in zip(predictions, labels):
                # Get probability map for true labels
                correct_probs = []
                for i in range(pred.shape[0]):
                    for j in range(pred.shape[1]):
                        true_class = label[i, j]
                        correct_probs.append(pred[i, j, true_class])
                confidences.append(np.mean(correct_probs))
            return np.array(confidences)
        
        train_confidences = get_confidence(train_preds, train_labels)
        test_confidences = get_confidence(test_preds, test_labels)
        
        # Determine threshold
        if threshold is None:
            threshold = (np.mean(train_confidences) + np.mean(test_confidences)) / 2
        
        # Classify as member or non-member
        train_classified = (train_confidences > threshold).astype(int)
        test_classified = (test_confidences > threshold).astype(int)
        
        # Calculate accuracy
        train_accuracy = np.mean(train_classified == 1)  # Should be classified as members
        test_accuracy = np.mean(test_classified == 0)    # Should be classified as non-members
        overall_accuracy = (train_accuracy + test_accuracy) / 2
        
        print(f"   Threshold: {threshold:.4f}")
        print(f"   Train confidence: {np.mean(train_confidences):.4f} ± {np.std(train_confidences):.4f}")
        print(f"   Test confidence:  {np.mean(test_confidences):.4f} ± {np.std(test_confidences):.4f}")
        print(f"   Attack Accuracy: {overall_accuracy*100:.2f}%")
        
        return {
            'train_confidences': train_confidences,
            'test_confidences': test_confidences,
            'threshold': threshold,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overall_accuracy': overall_accuracy
        }
    
    def _calculate_ssim_manual(self, img1, img2):
        """Manual SSIM calculation if skimage is not available"""
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        
        return ssim
    
    def evaluate_reconstruction_quality(self, original_image, reconstructed_image):
        """
        Evaluate the quality of reconstruction using multiple metrics.
        
        Args:
            original_image: Original ground truth image
            reconstructed_image: Reconstructed image from attack
            
        Returns:
            Dictionary with quality metrics
        """
        # Ensure same shape
        orig = original_image.squeeze()
        recon = reconstructed_image.squeeze()
        
        # Mean Squared Error (lower is better)
        mse = mean_squared_error(orig.flatten(), recon.flatten())
        
        # Peak Signal-to-Noise Ratio (higher is better)
        psnr = cv2.PSNR(
            (orig * 255).astype(np.uint8),
            (recon * 255).astype(np.uint8)
        )
        
        # Structural Similarity Index (higher is better, max=1.0)
        if SKIMAGE_AVAILABLE:
            ssim = structural_similarity(orig, recon, data_range=1.0)
        else:
            ssim = self._calculate_ssim_manual(orig, recon)
        
        # Normalized Cross-Correlation (higher is better, max=1.0)
        orig_norm = (orig - np.mean(orig)) / (np.std(orig) + 1e-8)
        recon_norm = (recon - np.mean(recon)) / (np.std(recon) + 1e-8)
        ncc = np.mean(orig_norm * recon_norm)
        
        metrics = {
            'MSE': mse,
            'PSNR': psnr,
            'SSIM': ssim,
            'NCC': ncc
        }
        
        return metrics
    
    def visualize_attack_results(self, original_image, reconstructed_images, 
                                attack_names, save_path='attack_results.png'):
        """
        Visualize results of multiple attacks.
        
        Args:
            original_image: Original image
            reconstructed_images: List of reconstructed images from different attacks
            attack_names: List of attack names
            save_path: Path to save visualization
        """
        num_attacks = len(reconstructed_images)
        fig, axes = plt.subplots(1, num_attacks + 1, figsize=(4 * (num_attacks + 1), 4))
        
        # Original image
        axes[0].imshow(original_image.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Reconstructed images
        for i, (recon_img, name) in enumerate(zip(reconstructed_images, attack_names)):
            axes[i + 1].imshow(recon_img.squeeze(), cmap='gray', vmin=0, vmax=1)
            
            # Calculate metrics
            metrics = self.evaluate_reconstruction_quality(original_image, recon_img)
            title = f'{name}\nSSIM: {metrics["SSIM"]:.3f} | PSNR: {metrics["PSNR"]:.1f}dB'
            axes[i + 1].set_title(title, fontsize=10)
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to {save_path}")
        plt.show()
    
    def comprehensive_privacy_test(self, test_images, test_labels, 
                                  train_images=None, train_labels=None,
                                  num_samples=5):
        """
        Run comprehensive privacy attacks and generate report.
        
        Args:
            test_images: Test images to attack
            test_labels: Test labels
            train_images: Training images (for membership inference)
            train_labels: Training labels (for membership inference)
            num_samples: Number of samples to test
            
        Returns:
            Dictionary with all attack results
        """
        print("\n" + "="*80)
        print("🔒 COMPREHENSIVE PRIVACY ATTACK TESTING")
        print("="*80)
        
        results = {
            'gradient_reconstructions': [],
            'feature_reconstructions': [],
            'metrics': []
        }
        
        # Test reconstruction attacks on multiple samples
        for i in range(min(num_samples, len(test_images))):
            print(f"\n--- Sample {i+1}/{num_samples} ---")
            
            sample_img = test_images[i:i+1]
            sample_label = test_labels[i]
            
            # Gradient-based attack
            grad_recon, grad_loss = self.gradient_based_reconstruction(
                sample_label, num_iterations=500
            )
            results['gradient_reconstructions'].append(grad_recon)
            
            # Feature-based attack
            feat_recon, feat_loss = self.feature_based_reconstruction(
                sample_img, num_iterations=300
            )
            results['feature_reconstructions'].append(feat_recon)
            
            # Evaluate reconstruction quality
            grad_metrics = self.evaluate_reconstruction_quality(sample_img[0], grad_recon)
            feat_metrics = self.evaluate_reconstruction_quality(sample_img[0], feat_recon)
            
            results['metrics'].append({
                'gradient_metrics': grad_metrics,
                'feature_metrics': feat_metrics
            })
            
            print(f"\n   📊 Reconstruction Quality:")
            print(f"   Gradient-based - SSIM: {grad_metrics['SSIM']:.4f}, PSNR: {grad_metrics['PSNR']:.2f}dB")
            print(f"   Feature-based  - SSIM: {feat_metrics['SSIM']:.4f}, PSNR: {feat_metrics['PSNR']:.2f}dB")
        
        # Membership inference attack
        if train_images is not None and train_labels is not None:
            print(f"\n{'='*80}")
            membership_results = self.membership_inference_attack(
                self.model, train_images[:100], train_labels[:100],
                test_images[:100], test_labels[:100]
            )
            results['membership_inference'] = membership_results
        
        # Overall privacy score
        avg_ssim_grad = np.mean([m['gradient_metrics']['SSIM'] for m in results['metrics']])
        avg_ssim_feat = np.mean([m['feature_metrics']['SSIM'] for m in results['metrics']])
        
        print(f"\n{'='*80}")
        print("🎯 PRIVACY PROTECTION SUMMARY")
        print("="*80)
        print(f"Average Reconstruction Quality (SSIM):")
        print(f"  Gradient-based Attack: {avg_ssim_grad:.4f}")
        print(f"  Feature-based Attack:  {avg_ssim_feat:.4f}")
        
        # Privacy score (lower reconstruction quality = better privacy)
        privacy_score = 1.0 - (avg_ssim_grad + avg_ssim_feat) / 2
        print(f"\n🛡️  Privacy Protection Score: {privacy_score*100:.2f}% ")
        print(f"   (100% = perfect privacy, 0% = no privacy)")
        
        if privacy_score > 0.7:
            print("   ✅ STRONG privacy protection!")
        elif privacy_score > 0.5:
            print("   ⚠️  MODERATE privacy protection")
        else:
            print("   ❌ WEAK privacy protection - model is vulnerable!")
        
        print("="*80)
        
        return results


# Example usage function
def run_privacy_attack_comparison(model_without_privacy, model_with_privacy, 
                                 test_images, test_labels, train_images=None, 
                                 train_labels=None):
    """
    Compare privacy protection between models with and without privacy layers.
    
    Args:
        model_without_privacy: Model trained without privacy layers
        model_with_privacy: Model trained with privacy layers
        test_images: Test images
        test_labels: Test labels
        train_images: Training images (optional)
        train_labels: Training labels (optional)
    """
    print("\n" + "="*80)
    print("🔬 COMPARING PRIVACY PROTECTION: WITH vs WITHOUT PRIVACY LAYERS")
    print("="*80)
    
    # Attack model WITHOUT privacy
    print("\n\n🎯 ATTACKING MODEL WITHOUT PRIVACY LAYERS")
    print("-"*80)
    attacker1 = ModelInversionAttack(model_without_privacy)
    results_no_privacy = attacker1.comprehensive_privacy_test(
        test_images, test_labels, train_images, train_labels, num_samples=3
    )
    
    # Attack model WITH privacy
    print("\n\n🎯 ATTACKING MODEL WITH PRIVACY LAYERS")
    print("-"*80)
    attacker2 = ModelInversionAttack(model_with_privacy)
    results_with_privacy = attacker2.comprehensive_privacy_test(
        test_images, test_labels, train_images, train_labels, num_samples=3
    )
    
    # Visualize comparison for first sample
    if len(test_images) > 0:
        sample_img = test_images[0:1]
        
        reconstructions = [
            results_no_privacy['gradient_reconstructions'][0],
            results_with_privacy['gradient_reconstructions'][0],
            results_no_privacy['feature_reconstructions'][0],
            results_with_privacy['feature_reconstructions'][0]
        ]
        
        names = [
            'No Privacy\n(Gradient)',
            'With Privacy\n(Gradient)',
            'No Privacy\n(Feature)',
            'With Privacy\n(Feature)'
        ]
        
        attacker1.visualize_attack_results(
            sample_img[0], reconstructions, names,
            save_path='privacy_comparison.png'
        )
    
    return results_no_privacy, results_with_privacy