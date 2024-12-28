import os
import glob
import cv2

def load_images_from_folder(folder_path, extensions=('.png', '.jpg', '.jpeg')):
    """
    Load all images from a folder into a list of (image_path, image_data).
    """
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))

    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            images.append((img_path, img))
    return images
