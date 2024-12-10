# Import the libraries
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, 
    Conv2DTranspose, BatchNormalization, Dropout, Lambda,
    Activation, Dense, GlobalAveragePooling2D, Reshape, Multiply, Add, Average
)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from patchify import patchify, unpatchify
import random
import shutil
from PIL import Image

# Global variables
patch_size = 16*8
patch_dir = 'dataset_patched'
scaling_factor = 1
raw_base = os.path.join('dataset raw', 'Y2B_23')  # Fixed path with space

# Enable mixed precision training for better performance
set_global_policy('mixed_float16')

def create_directory_structure():
    """Create the required directory structure if it doesn't exist"""
    base_dir = 'dataset'
    subdirs = ['train_images/train', 'train_masks/train', 'val_images/val', 'val_masks/val']
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)

def combine_masks(image_name, source_mask_dir):
    """Combine individual class masks into a single mask"""
    # Initialize empty mask
    mask = None
    
    # Load and combine masks for each class
    classes = ['root', 'shoot', 'seed']
    for i, class_name in enumerate(classes, start=1):
        mask_path = os.path.join(source_mask_dir, f"{image_name}_{class_name}_mask.tif")
        if os.path.exists(mask_path):
            try:
                class_mask = np.array(Image.open(mask_path))
                if mask is None:
                    mask = np.zeros_like(class_mask)
                # Add class label (1, 2, or 3) where mask is True
                mask[class_mask > 0] = i
            except Exception as e:
                print(f"Error processing mask {mask_path}: {str(e)}")
    
    return mask

def process_dataset():
    """Process and organize the dataset"""
    print(f"Looking for dataset in: {raw_base}")
    if not os.path.exists(raw_base):
        raise FileNotFoundError(f"Dataset directory not found at {raw_base}. Please check the path.")
    
    # Process training data
    process_split('train', raw_base)
    
    # Process validation data (using test split)
    process_split('test', raw_base, dest_split='val')

def process_split(split, raw_base, dest_split=None):
    """Process a specific data split"""
    if dest_split is None:
        dest_split = split
        
    source_img_dir = os.path.join(raw_base, 'images', split)
    source_mask_dir = os.path.join(raw_base, 'masks')
    
    dest_img_dir = os.path.join('dataset', f'{dest_split}_images', dest_split)
    dest_mask_dir = os.path.join('dataset', f'{dest_split}_masks', dest_split)
    
    # Ensure source directory exists
    if not os.path.exists(source_img_dir):
        raise FileNotFoundError(f"Source image directory not found at {source_img_dir}")
    
    print(f"Processing {split} split:")
    print(f"Source image directory: {source_img_dir}")
    print(f"Source mask directory: {source_mask_dir}")
    print(f"Destination image directory: {dest_img_dir}")
    print(f"Destination mask directory: {dest_mask_dir}")
    
    # Create destination directories if they don't exist
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)
    
    # Process each image in the split
    try:
        image_files = [f for f in os.listdir(source_img_dir) if f.endswith('.png')]
    except Exception as e:
        print(f"Error reading directory {source_img_dir}: {str(e)}")
        image_files = []
    
    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {source_img_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    for img_name in image_files:
        try:
            print(f"Processing {img_name}")
            # Copy image
            src_img_path = os.path.join(source_img_dir, img_name)
            dst_img_path = os.path.join(dest_img_dir, img_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Process masks
            base_name = os.path.splitext(img_name)[0]
            combined_mask = combine_masks(base_name, source_mask_dir)
            
            if combined_mask is not None:
                # Save combined mask
                mask_path = os.path.join(dest_mask_dir, f"{base_name}.png")
                cv2.imwrite(mask_path, combined_mask)
            else:
                print(f"Warning: No mask found for {img_name}")
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

# print("Creating directory structure...")
# create_directory_structure()

# print("Processing dataset...")
# process_dataset()

# print("Dataset organization complete!")



# Create a new directory for patches
# patch_dir = 'dataset_patched'
# for subdir in ['train_images/train', 'train_masks/train', 'val_images/val', 'val_masks/val']:
#     os.makedirs(os.path.join(patch_dir, subdir), exist_ok=True)

def padder(image, patch_size):
    """
    Adds padding to an image to make its dimensions divisible by a specified patch size.

    This function calculates the amount of padding needed for both the height and width of an image so that its dimensions become divisible by the given patch size. The padding is applied evenly to both sides of each dimension (top and bottom for height, left and right for width). If the padding amount is odd, one extra pixel is added to the bottom or right side. The padding color is set to black (0, 0, 0).

    Parameters:
    - image (numpy.ndarray): The input image as a NumPy array. Expected shape is (height, width, channels).
    - patch_size (int): The patch size to which the image dimensions should be divisible. It's applied to both height and width.

    Returns:
    - numpy.ndarray: The padded image as a NumPy array with the same number of channels as the input. Its dimensions are adjusted to be divisible by the specified patch size.

    Example:
    - padded_image = padder(cv2.imread('example.jpg'), 128)

    """
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def create_and_save_patches(dataset_type, patch_size, scaling_factor):
    """
    Splits images and their corresponding masks from a blood cell dataset into smaller patches and saves them.

    This function takes images and masks from a specified dataset type, scales them if needed, and then splits them into smaller patches. Each patch is saved as a separate file. This is useful for preparing data for tasks like image segmentation in machine learning.

    Parameters:
    - dataset_type (str): The type of the dataset to process (e.g., 'train', 'test'). It expects a directory structure like 'blood_cell_dataset/{dataset_type}_images/{dataset_type}' for images and 'blood_cell_dataset/{dataset_type}_masks/{dataset_type}' for masks.
    - patch_size (int): The size of the patches to be created. Patches will be squares of size patch_size x patch_size.
    - scaling_factor (float): The factor by which the images and masks should be scaled. A value of 1 means no scaling.

    Returns:
    None. The function saves the patches as .png files in directories based on their original paths, but replacing 'blood_cell_dataset' with 'blood_cell_dataset_patched'.

    Note:
    - The function assumes a specific directory structure and naming convention for the dataset.
    """
    for image_path in glob.glob(f'dataset/{dataset_type}_images/{dataset_type}/*.png'):
        mask_path = image_path.replace('images', 'masks')

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        image = padder(image, patch_size)
        if scaling_factor != 1:
            image = cv2.resize(image, (0,0), fx=scaling_factor, fy=scaling_factor)
        patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        patches = patches.reshape(-1, patch_size, patch_size, 3)

        image_patch_path = image_path.replace('dataset', patch_dir)
        for i, patch in enumerate(patches):
            image_patch_path_numbered = f'{image_patch_path[:-4]}_{i}.png'
            cv2.imwrite(image_patch_path_numbered, patch)

        mask_path = image_path.replace('image', 'mask')
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"Failed to load mask: {mask_path}")
            continue
        mask = padder(mask, patch_size)
        if scaling_factor != 1:
            mask = cv2.resize(mask, (0,0), fx=scaling_factor, fy=scaling_factor)
        patches = patchify(mask, (patch_size, patch_size), step=patch_size)
        patches = patches.reshape(-1, patch_size, patch_size, 1)

        mask_patch_path = mask_path.replace('dataset', patch_dir)
        for i, patch in enumerate(patches):
            mask_patch_path_numbered = f'{mask_patch_path[:-4]}_{i}.png'
            cv2.imwrite(mask_patch_path_numbered, patch)

# create_and_save_patches('train', patch_size, scaling_factor)
# create_and_save_patches('val', patch_size, scaling_factor)

# # Find the image dimensions

# # Load a sample image
# img = cv2.imread("./dataset/Y2B_23/images/train/000_43-2-ROOT1-2023-08-08_pvd_OD0001_f6h1_02-Fish Eye Corrected.png")

# # Get the dimensions
# height, width, channels = img.shape

# # Report
# print(f"Height: {height}, Width: {width}, Channels: {channels}")

def apply_augmentation(image, mask):
    """
    Apply random augmentations to both image and mask.
    
    Args:
        image: Input image (H, W, C)
        mask: Input mask (H, W, 1)
    Returns:
        Augmented image and mask
    """
    # Convert to numpy for CPU operations
    image = image.numpy() if isinstance(image, tf.Tensor) else image
    mask = mask.numpy() if isinstance(mask, tf.Tensor) else mask
    
    # Random rotation
    if np.random.random() > 0.5:
        k = np.random.randint(0, 4)  # 0-3 for number of 90-degree rotations
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
    
    # Random flip left-right
    if np.random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    # Random flip up-down
    if np.random.random() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    
    # Convert back to tensors for remaining operations
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    
    # Random brightness (only for image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, 0.2)
    
    # Random contrast (only for image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # Ensure image values stay in valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask

def create_dataset(image_paths, mask_dir, patch_size, scaling_factor, batch_size, is_training=True):
    """
    Creates a tf.data.Dataset that streams data instead of loading it all at once.
    """
    def load_and_preprocess(image_path):
        # Convert string tensor to string
        image_path = image_path.numpy().decode('utf-8')
        mask_path = os.path.join(mask_dir, os.path.basename(image_path))
        
        # Read images
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        
        if image is None or mask is None:
            return None, None
        
        # Process images
        image = padder(image, patch_size)
        mask = padder(mask, patch_size)
        
        if scaling_factor != 1:
            image = cv2.resize(image, (0,0), fx=scaling_factor, fy=scaling_factor)
            mask = cv2.resize(mask, (0,0), fx=scaling_factor, fy=scaling_factor)
        
        # Create patches
        patches = []
        masks = []
        
        for i in range(0, image.shape[0], patch_size):
            for j in range(0, image.shape[1], patch_size):
                if i + patch_size <= image.shape[0] and j + patch_size <= image.shape[1]:
                    img_patch = image[i:i+patch_size, j:j+patch_size].copy()
                    mask_patch = mask[i:i+patch_size, j:j+patch_size].copy()
                    
                    # Normalize and convert to float32
                    img_patch = img_patch.astype('float32') / 255.0
                    mask_patch = mask_patch.astype('float32') / 255.0
                    mask_patch = mask_patch.reshape(patch_size, patch_size, 1)
                    
                    if is_training:
                        img_patch, mask_patch = apply_augmentation(img_patch, mask_patch)
                    
                    patches.append(img_patch)
                    masks.append(mask_patch)
        
        return np.array(patches, dtype='float32'), np.array(masks, dtype='float32')
    
    def generator():
        for image_path in image_paths:
            patches, masks = tf.py_function(
                load_and_preprocess,
                [image_path],
                [tf.float32, tf.float32]
            )
            if patches is not None and masks is not None:
                for i in range(len(patches)):
                    yield patches[i], masks[i]
    
    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(patch_size, patch_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(patch_size, patch_size, 1), dtype=tf.float32)
        )
    )
    
    # Configure dataset
    dataset = dataset.cache()
    if is_training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def f1(y_true, y_pred):
    """Calculate F1 score"""
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_score

def iou_score(y_true, y_pred):
    """Calculate IoU score"""
    y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
    iou = (intersection + K.epsilon()) / (union + K.epsilon())
    return iou

def combined_loss():
    """Combined loss function for better segmentation with mixed precision support"""
    def loss(y_true, y_pred):
        def dice_loss(y_true, y_pred, smooth=1e-6):
            # Cast inputs to float32 for loss calculation
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred)
            return 1 - ((2. * intersection + smooth) / 
                       (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth))

        def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
            # Cast inputs to float32 for loss calculation
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Flatten the inputs
            y_true = tf.reshape(y_true, [-1])
            y_pred = tf.reshape(y_pred, [-1])
            
            # Calculate focal loss
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_factor = tf.ones_like(y_true) * alpha
            alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
            cross_entropy = -tf.math.log(p_t)
            weight = alpha_t * tf.pow((1 - p_t), gamma)
            focal_loss = weight * cross_entropy
            return tf.reduce_mean(focal_loss)

        return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

    return loss

def conv_block_v2(inputs, filters, dropout_rate=0.0):
    """Simplified convolution block with reduced memory usage"""
    x = Conv2D(filters // 2, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)  # Reduced filters
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(filters // 2, (3, 3), padding='same', kernel_initializer='he_normal')(x)  # Reduced filters
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Simplified attention
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // 8, activation='relu')(se)  # Reduced size
    se = Dense(filters // 2, activation='sigmoid')(se)  # Reduced size
    se = Reshape((1, 1, filters // 2))(se)
    x = Multiply()([x, se])
    
    return x

def unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    """
    Enhanced U-Net implementation with:
    - Squeeze-and-Excitation blocks
    - Deep supervision
    - Advanced residual connections
    - Instance normalization
    - Deeper architecture
    """
    if IMG_HEIGHT % 16 != 0 or IMG_WIDTH % 16 != 0:
        raise ValueError(f"Height ({IMG_HEIGHT}) and width ({IMG_WIDTH}) must be divisible by 16")

    # Input
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Initial processing
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Encoder path with SE blocks
    skip1 = conv_block_v2(x, 64, 0.1)
    p1 = MaxPooling2D((2, 2))(skip1)
    
    skip2 = conv_block_v2(p1, 128, 0.1)
    p2 = MaxPooling2D((2, 2))(skip2)
    
    skip3 = conv_block_v2(p2, 256, 0.2)
    p3 = MaxPooling2D((2, 2))(skip3)
    
    skip4 = conv_block_v2(p3, 512, 0.2)
    p4 = MaxPooling2D((2, 2))(skip4)

    # Bridge with attention
    bridge = conv_block_v2(p4, 1024, 0.3)
    
    # Decoder path with deep supervision
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    u6 = concatenate([u6, skip4])
    c6 = conv_block_v2(u6, 512, 0.2)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, skip3])
    c7 = conv_block_v2(u7, 256, 0.2)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, skip2])
    c8 = conv_block_v2(u8, 128, 0.1)
    
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, skip1])
    c9 = conv_block_v2(u9, 64, 0.1)

    # Deep supervision outputs
    ds1 = Conv2D(1, (1, 1), activation='sigmoid')(c6)
    ds1 = UpSampling2D(size=(8, 8))(ds1)
    
    ds2 = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    ds2 = UpSampling2D(size=(4, 4))(ds2)
    
    ds3 = Conv2D(1, (1, 1), activation='sigmoid')(c8)
    ds3 = UpSampling2D(size=(2, 2))(ds3)
    
    main_output = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    # Combine deep supervision outputs
    outputs = Average()([main_output, ds1, ds2, ds3])
    
    # Create and compile model
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def create_model_ensemble(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, num_models=2):  # Reduced number of models
    """Create a smaller ensemble of models"""
    models = []
    
    # Base U-Net with reduced complexity
    def create_base_unet():
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        
        # Encoder - reduced number of filters
        c1 = conv_block_v2(inputs, 32, 0.1)  # Output: 16 filters
        p1 = MaxPooling2D((2, 2))(c1)
        
        c2 = conv_block_v2(p1, 64, 0.1)      # Output: 32 filters
        p2 = MaxPooling2D((2, 2))(c2)
        
        c3 = conv_block_v2(p2, 128, 0.2)     # Output: 64 filters
        p3 = MaxPooling2D((2, 2))(c3)
        
        # Bridge
        b = conv_block_v2(p3, 256, 0.3)      # Output: 128 filters
        
        # Decoder - match encoder dimensions
        u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(b)
        u3 = concatenate([u3, c3])
        c6 = conv_block_v2(u3, 128, 0.2)     # Output: 64 filters
        
        u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
        u2 = concatenate([u2, c2])
        c7 = conv_block_v2(u2, 64, 0.1)      # Output: 32 filters
        
        u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
        u1 = concatenate([u1, c1])
        c8 = conv_block_v2(u1, 32, 0.1)      # Output: 16 filters
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)
        
        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    models.append(create_base_unet())
    
    # Simplified attention U-Net
    def create_attention_unet():
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        
        # Encoder - reduced number of filters
        c1 = conv_block_v2(inputs, 32, 0.1)  # Output: 16 filters
        p1 = MaxPooling2D((2, 2))(c1)
        
        c2 = conv_block_v2(p1, 64, 0.1)      # Output: 32 filters
        p2 = MaxPooling2D((2, 2))(c2)
        
        c3 = conv_block_v2(p2, 128, 0.2)     # Output: 64 filters
        p3 = MaxPooling2D((2, 2))(c3)
        
        # Bridge
        b = conv_block_v2(p3, 256, 0.3)      # Output: 128 filters
        
        # Decoder with simplified attention
        # Match the number of filters with encoder outputs
        u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(b)
        # Add a 1x1 conv to match dimensions before attention
        u3_adj = Conv2D(64, (1, 1), padding='same')(u3)
        a3 = Multiply()([u3_adj, c3])
        c6 = conv_block_v2(a3, 128, 0.2)     # Output: 64 filters
        
        u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
        # Add a 1x1 conv to match dimensions before attention
        u2_adj = Conv2D(32, (1, 1), padding='same')(u2)
        a2 = Multiply()([u2_adj, c2])
        c7 = conv_block_v2(a2, 64, 0.1)      # Output: 32 filters
        
        u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
        # Add a 1x1 conv to match dimensions before attention
        u1_adj = Conv2D(16, (1, 1), padding='same')(u1)
        a1 = Multiply()([u1_adj, c1])
        c8 = conv_block_v2(a1, 32, 0.1)      # Output: 16 filters
        
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c8)
        
        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    models.append(create_attention_unet())
    
    # Compile models with mixed precision
    for model in models:
        base_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=combined_loss(),
            metrics=['accuracy', f1, iou_score]
        )
    
    return models

class EnsemblePredictor:
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.mean(predictions, axis=0)

# Custom F1 Score metric
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

# Custom Binary Accuracy for segmentation
def segmentation_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def train_model(model, train_generator, val_generator, epochs=100):
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            segmentation_accuracy,
            f1_score,
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )
    
    return model, history

if __name__ == "__main__":
    print("Starting the training pipeline...")
    
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                # Further reduce GPU memory limit
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1500)]  # Reduced to 1.5GB
                )
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    
    try:
        # Configure mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Get valid image paths
        train_dir = os.path.join('dataset', 'train_images', 'train')
        val_dir = os.path.join('dataset', 'val_images', 'val')
        train_mask_dir = os.path.join('dataset', 'train_masks', 'train')
        val_mask_dir = os.path.join('dataset', 'val_masks', 'val')
        
        train_paths = glob.glob(os.path.join(train_dir, '*.png'))
        val_paths = glob.glob(os.path.join(val_dir, '*.png'))
        
        # Filter valid paths
        train_paths = [p for p in train_paths if os.path.exists(os.path.join(train_mask_dir, os.path.basename(p)))]
        val_paths = [p for p in val_paths if os.path.exists(os.path.join(val_mask_dir, os.path.basename(p)))]
        
        print(f"Found {len(train_paths)} valid training images")
        print(f"Found {len(val_paths)} valid validation images")
        
        # Create datasets with smaller batch size
        BATCH_SIZE = 2  # Reduced batch size
        train_dataset = create_dataset(train_paths, train_mask_dir, patch_size, scaling_factor, BATCH_SIZE, is_training=True)
        val_dataset = create_dataset(val_paths, val_mask_dir, patch_size, scaling_factor, BATCH_SIZE, is_training=False)
        
        # Create models
        print("\nCreating model ensemble...")
        models = create_model_ensemble(patch_size, patch_size, 3)
        
        print("\nStarting ensemble training...")
        for epoch in range(50):
            print(f"\nEpoch {epoch+1}/50")
            
            for i, model in enumerate(models):
                print(f"Training model {i+1}/{len(models)}")
                
                # Training
                for batch, (X_batch, y_batch) in enumerate(train_dataset):
                    # Ensure inputs are float32
                    X_batch = tf.cast(X_batch, tf.float32)
                    y_batch = tf.cast(y_batch, tf.float32)
                    
                    with tf.GradientTape() as tape:
                        predictions = model(X_batch, training=True)
                        # Cast predictions to float32 for loss calculation
                        predictions = tf.cast(predictions, tf.float32)
                        loss = combined_loss()(y_batch, predictions)
                        # Scale loss for mixed precision
                        scaled_loss = loss
                    
                    gradients = tape.gradient(scaled_loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    if batch % 10 == 0:
                        print(f"Batch {batch}, Loss: {loss.numpy():.4f}")
                    
                    # Clear memory
                    del X_batch, y_batch, predictions, gradients
                    tf.keras.backend.clear_session()
                
                # Validation
                val_metrics = {'loss': [], 'accuracy': [], 'f1': [], 'iou_score': []}
                for X_val, y_val in val_dataset:
                    # Ensure inputs are float32
                    X_val = tf.cast(X_val, tf.float32)
                    y_val = tf.cast(y_val, tf.float32)
                    
                    val_results = model.evaluate(X_val, y_val, verbose=0)
                    for metric, value in zip(model.metrics_names, val_results):
                        val_metrics[metric].append(value)
                    
                    # Clear memory
                    del X_val, y_val
                    tf.keras.backend.clear_session()
                
                # Average validation metrics
                avg_val_metrics = {metric: np.mean(values) for metric, values in val_metrics.items()}
                print(f"Validation metrics: {avg_val_metrics}")
                
                # Save model if it's the best so far
                if not hasattr(model, 'best_val_f1') or avg_val_metrics['f1'] > model.best_val_f1:
                    model.best_val_f1 = avg_val_metrics['f1']
                    model.save(f'best_model_{i+1}.keras')
                
                # Clear GPU memory after training each model
                tf.keras.backend.clear_session()
        
        print("\nCreating ensemble predictor...")
        ensemble = EnsemblePredictor(models)
        
        # Predict in smaller batches using the dataset
        print("\nGenerating and saving ensemble predictions...")
        all_predictions = []
        for X_batch, _ in val_dataset:
            # Ensure inputs are float32
            X_batch = tf.cast(X_batch, tf.float32)
            pred = ensemble.predict(X_batch)
            all_predictions.append(pred)
            
            # Clear memory
            del X_batch, pred
            tf.keras.backend.clear_session()
        
        ensemble_predictions = np.concatenate(all_predictions, axis=0)
        np.save('ensemble_predictions.npy', ensemble_predictions)
        
        print("\nTraining pipeline completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise