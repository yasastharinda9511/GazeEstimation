import os
import cv2
import numpy as np

class DataLoader:
    def __init__(self):
        self.images = None
        self.labels = None

    def load_images_label_pairs(self, image_dir, label_dir, step=1, step_size=1000, add_noise=False):
        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))

        images = []
        labels = []

        print("Images count: " + str(len(image_files)))
        for img, label in zip(image_files[step * step_size: (step + 1) * step_size], label_files[step * step_size: (step + 1) * step_size]):
            img_path = os.path.join(image_dir, img)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if add_noise:
                noise = np.random.normal(loc=0, scale=10, size=img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            img = cv2.resize(img, (128, 128))
            images.append(img)

            label_path = os.path.join(label_dir, label)
            label = np.load(label_path)
            label = cv2.resize(label, (128, 128))
            labels.append(label)

        self.images = np.array(images)
        self.labels = np.array(labels)
        return self.images, self.labels

    def get_images(self):
        return self.images

    def get_labels(self):
        return self.labels
    
    def get_image_label_pairs(self):
        if self.images is None or self.labels is None:
            raise ValueError("Images and labels must be loaded before getting pairs.")
        return self.images, self.labels