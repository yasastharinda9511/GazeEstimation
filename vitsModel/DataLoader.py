
import numpy as np
import os
import cv2

class DataLoader:
    
    @staticmethod
    def load_grayscale_images_label_pairs(image_dir, label_dir, step=0, step_size=1000, 
                                    target_size=(224, 224), add_noise=False, 
                                    normalize_on_load=False):
        # Get sorted file lists
        image_files = sorted([f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        label_files = sorted([f for f in os.listdir(label_dir) 
                            if f.lower().endswith(('.npy', '.png', '.jpg', '.jpeg'))])
        
        print(f"Total images found: {len(image_files)}")
        print(f"Total labels found: {len(label_files)}")
        
        # Validate file count match
        if len(image_files) != len(label_files):
            print(f"⚠️  Warning: Mismatch in file counts - Images: {len(image_files)}, Labels: {len(label_files)}")
        
        # Calculate slice indices
        start_idx = step * step_size
        end_idx = min((step + 1) * step_size, len(image_files))
        
        print(f"📊 Loading batch {step + 1}: indices {start_idx} to {end_idx-1}")
        
        images = []
        labels = []
        failed_files = []
        
        # Process the subset
        for i, (img_file, label_file) in enumerate(zip(
            image_files[start_idx:end_idx], 
            label_files[start_idx:end_idx]
        )):
            try:
                # Load grayscale image
                img_path = os.path.join(image_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"❌ Could not load image: {img_file}")
                    failed_files.append(img_file)
                    continue
                
                # Resize image
                img = cv2.resize(img, target_size)
                
                # Add channel dimension: (H, W) -> (H, W, 1)
                img = np.expand_dims(img, axis=-1)
                
                # Optional: normalize on load
                if normalize_on_load:
                    img = img.astype(np.float32) / 255.0
                
                # Add noise if requested
                if add_noise:
                    if normalize_on_load:
                        noise = np.random.normal(loc=0, scale=0.02, size=img.shape).astype(np.float32)
                        img = np.clip(img + noise, 0.0, 1.0)
                    else:
                        noise = np.random.normal(loc=0, scale=10, size=img.shape).astype(np.float32)
                        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                
                images.append(img)
                
                # Load label/mask
                label_path = os.path.join(label_dir, label_file)
                
                if label_file.endswith('.npy'):
                    label = np.load(label_path)
                else:
                    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    if label is None:
                        print(f"❌ Could not load label: {label_file}")
                        failed_files.append(label_file)
                        # Remove corresponding image
                        images.pop()
                        continue
                
                # Resize label to match target size
                label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)
                
                # Ensure labels are integers and in valid range [0, 3]
                label = label.astype(np.int32)
                
                # Validate and clip label values
                unique_labels = np.unique(label)
                if np.max(unique_labels) > 3 or np.min(unique_labels) < 0:
                    print(f"⚠️  Label {label_file} has invalid values {unique_labels}, clipping to [0,3]")
                    label = np.clip(label, 0, 3)
                
                labels.append(label)
                    
            except Exception as e:
                print(f"💥 Error processing {img_file}: {str(e)}")
                failed_files.append(img_file)
                # Remove corresponding image if it was added
                if len(images) > len(labels):
                    images.pop()
                continue
        
        # Final validation
        if len(images) != len(labels):
            min_len = min(len(images), len(labels))
            images = images[:min_len]
            labels = labels[:min_len]
            print(f"⚠️  Trimmed to {min_len} matching pairs")
        
        if len(images) == 0:
            raise ValueError("No valid image-label pairs were loaded!")
        
        # Convert to numpy arrays
        images = np.array(images, dtype=np.float32 if normalize_on_load else np.uint8)
        labels = np.array(labels, dtype=np.int32)
        
        return images, labels