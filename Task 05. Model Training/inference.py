# import things
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
import patchify

# Patch size
patch_size = 960

def dice_loss(y_true, y_pred, smooth=1.0):
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Flatten the inputs
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    gamma = tf.cast(gamma, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    
    # Clip prediction values to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    
    # Calculate focal loss
    pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_factor = tf.ones_like(y_true, dtype=tf.float32) * alpha
    alpha_factor = tf.where(y_true == 1, alpha_factor, 1 - alpha_factor)
    
    focal_weight = alpha_factor * tf.pow(1 - pt, gamma)
    
    bce = -tf.math.log(pt)
    loss = focal_weight * bce
    
    return tf.reduce_mean(loss)
    
def weighted_bce(y_true, y_pred, pos_weight=10.0):
    """Weighted binary cross-entropy to handle class imbalance"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    
    # Calculate weighted BCE
    pos_weight = tf.constant(pos_weight, dtype=tf.float32)
    bce = -(pos_weight * y_true * tf.math.log(y_pred) + 
            (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(bce)

def precision_focused_loss(y_true, y_pred, beta=2.0):
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    beta = tf.cast(beta, tf.float32)
    
    # Calculate true positives, false positives, and false negatives
    TP = tf.reduce_sum(y_true * y_pred)
    FP = tf.reduce_sum((1 - y_true) * y_pred)
    FN = tf.reduce_sum(y_true * (1 - y_pred))
    
    # Calculate precision and recall
    precision = TP / (TP + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())
    
    # Calculate F-beta score
    beta_squared = beta * beta
    f_beta = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall + K.epsilon())
    
    return 1 - f_beta

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-5):
    """Tversky loss with asymmetric weighting for better handling of FP and FN"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Calculate True Positives, False Positives, and False Negatives
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    # Calculate Tversky index
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky

def boundary_loss(y_true, y_pred, theta=1.5):
    """Boundary-aware loss to focus on root edges"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Calculate gradients using Sobel filters
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    
    # Calculate boundary magnitude
    boundary_true = tf.sqrt(tf.square(dx_true) + tf.square(dy_true))
    boundary_pred = tf.sqrt(tf.square(dx_pred) + tf.square(dy_pred))
    
    # Calculate boundary loss
    boundary_diff = tf.abs(boundary_true - boundary_pred)
    weighted_diff = boundary_diff * tf.pow(boundary_true, theta)
    
    return tf.reduce_mean(weighted_diff)

def combined_loss(y_true, y_pred):
    """Comprehensive loss function for plant root segmentation"""
    # Cast inputs to float32 and clip predictions
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    
    # Weight factors for each loss component
    dice_weight = tf.constant(1.0, dtype=tf.float32)
    focal_weight = tf.constant(2.5, dtype=tf.float32)
    precision_weight = tf.constant(1.5, dtype=tf.float32)
    tversky_weight = tf.constant(1.5, dtype=tf.float32)
    boundary_weight = tf.constant(0.5, dtype=tf.float32)
    
    try:
        # Calculate individual losses with error handling
        d_loss = dice_loss(y_true, y_pred, smooth=1e-5)
        f_loss = focal_loss(y_true, y_pred, gamma=3.0, alpha=0.75)
        p_loss = precision_focused_loss(y_true, y_pred, beta=0.75)
        t_loss = tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3)
        bound_loss = boundary_loss(y_true, y_pred, theta=1.5)
        
        # Check for NaN values
        losses = [d_loss, f_loss, p_loss, t_loss, bound_loss]
        for loss in losses:
            if tf.reduce_any(tf.math.is_nan(loss)):
                # If NaN detected, return a safe fallback loss
                return dice_loss(y_true, y_pred, smooth=1e-5)
        
        # Combine losses with fixed weights
        total_loss = (
            dice_weight * d_loss +
            focal_weight * f_loss +
            precision_weight * p_loss +
            tversky_weight * t_loss +
            boundary_weight * bound_loss
        )
        
        # Final NaN check
        if tf.reduce_any(tf.math.is_nan(total_loss)):
            return dice_loss(y_true, y_pred, smooth=1e-5)
            
        return total_loss
        
    except Exception as e:
        # If any error occurs, fallback to dice loss
        print(f"Error in combined loss: {e}")
        return dice_loss(y_true, y_pred, smooth=1e-5)

def f1(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # Input layer 
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Convert grayscale to RGB and resize
    if IMG_CHANNELS == 1:
        x = tf.keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)
    else:
        x = inputs
    x = tf.keras.layers.Resizing(224, 224)(x)
    
    # Load MobileNet with the correct input shape
    base_model = tf.keras.applications.MobileNet(
        input_tensor=x,
        include_top=False,
        weights='imagenet'
    )
    
    # Get skip connections
    s1 = base_model.get_layer('conv_pw_1_relu').output  # 112x112
    s2 = base_model.get_layer('conv_pw_3_relu').output  # 56x56
    s3 = base_model.get_layer('conv_pw_5_relu').output  # 28x28
    s4 = base_model.get_layer('conv_pw_11_relu').output # 14x14
    
    # Bridge
    bridge = base_model.get_layer('conv_pw_13_relu').output  # 7x7
    
    # Decoder path with batch normalization and dropout
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bridge)
    u6 = concatenate([u6, s4])
    u6 = tf.keras.layers.BatchNormalization()(u6)
    u6 = tf.keras.layers.Dropout(0.3)(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, s3])
    u7 = tf.keras.layers.BatchNormalization()(u7)
    u7 = tf.keras.layers.Dropout(0.3)(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, s2])
    u8 = tf.keras.layers.BatchNormalization()(u8)
    u8 = tf.keras.layers.Dropout(0.2)(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, s1])
    u9 = tf.keras.layers.BatchNormalization()(u9)
    u9 = tf.keras.layers.Dropout(0.1)(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # Final layers with stronger activation
    outputs = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c9)
    outputs = Conv2D(32, (3, 3), activation='relu', padding='same')(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = Conv2D(16, (3, 3), activation='relu', padding='same')(outputs)
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(outputs)

    # Add batch normalization and dropout for better training
    outputs = tf.keras.layers.BatchNormalization()(outputs)
    outputs = tf.keras.layers.Dropout(0.3)(outputs)
    
    # Resize back to original dimensions
    final_outputs = tf.keras.layers.Resizing(IMG_HEIGHT, IMG_WIDTH)(outputs)
    
    model = Model(inputs=inputs, outputs=final_outputs)
    model.compile(
        optimizer='adam', 
        loss=combined_loss, 
        metrics=[
            f1,
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model

def extract_petri_dish(image):
    """
    Extracts a square region containing the Petri dish from the input image.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Cropped square image containing the Petri dish
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's thresholding to separate dish from background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (should be the Petri dish)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate the size of the square crop (use larger dimension)
    size = max(w, h)
    
    # Calculate new coordinates to center the dish
    center_x = x + w//2
    center_y = y + h//2
    
    # Calculate crop coordinates
    x1 = max(0, center_x - size//2)
    y1 = max(0, center_y - size//2)
    x2 = min(image.shape[1], x1 + size)
    y2 = min(image.shape[0], y1 + size)
    
    # Crop the image
    cropped = image[y1:y2, x1:x2]
    
    return cropped, (x1, y1, x2, y2)

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


def create_patches(image, patch_size):
    # Add channel dimension if needed
    if len(image.shape) == 2:
        image = image[..., np.newaxis]  # Add channel dimension
    
    # Create patches
    patches = patchify.patchify(image, (patch_size, patch_size, 1), step=patch_size)
    
    # Remove the extra dimensions when returning
    patches = patches[:, :, 0, :, :, 0]
    
    print(f"Original shape: {image.shape}")
    print(f"Patches shape: {patches.shape}")
    return patches, image[:, :, 0]  # Return 2D image

# Z score normalization for image
def z_score_normalization(image):
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image

# Reverse the cropping and padding on all of the images
def reverse_padding_and_cropping(predicted_mask, original_shape, petri_coords):
    """
    Reverses the padding and cropping operations to align the predicted mask with the original image.
    
    Args:
        predicted_mask: The mask after prediction (padded)
        original_shape: Shape of the original image (h, w)
        petri_coords: Coordinates of petri dish crop (x1, y1, x2, y2)
    
    Returns:
        Final mask aligned with the original image
    """
    # 1. Remove padding from the predicted mask
    x1, y1, x2, y2 = petri_coords
    crop_height = y2 - y1
    crop_width = x2 - x1
    
    # Calculate padding amounts that were added
    h_pad = predicted_mask.shape[0] - crop_height
    w_pad = predicted_mask.shape[1] - crop_width
    
    # Remove padding
    top_pad = h_pad // 2
    left_pad = w_pad // 2
    unpadded_mask = predicted_mask[top_pad:top_pad+crop_height, 
                                 left_pad:left_pad+crop_width]
    
    # 2. Place the unpadded mask back in the original image size
    final_mask = np.zeros(original_shape, dtype=np.float32)
    final_mask[y1:y2, x1:x2] = unpadded_mask
    
    return final_mask

def predict_roots(image_path, model, patch_size=960, threshold=0.6):
    """
    Predicts root segments in a plant image using a trained U-Net model.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the trained model weights
        patch_size (int): Size of patches for processing (default: 960)
        threshold (float): Threshold for binary mask creation (default: 0.6)
    
    Returns:
        tuple: (predicted_mask, overlay_image) where:
            - predicted_mask: Binary mask of predicted roots (same size as input)
            - overlay_image: Original image with predicted roots highlighted
    """
    # Load the image
    image = cv2.imread(image_path, 0)

    # Increase the contrast of the image
    # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    # image = clahe.apply(image)
    
    # Extract petri dish
    petri_dish, petri_coords = extract_petri_dish(image)
    
    # Add padding
    padded_petri = padder(petri_dish, patch_size)
    
    # Create patches
    patches, _ = create_patches(padded_petri, patch_size)
    
    # Prepare patches for prediction
    n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
    patches_reshaped = patches.reshape(-1, patch_size, patch_size)
    
    # Load model if path is provided
    # if isinstance(model_path, str):
    #     model = load_model(model_path, custom_objects={"f1": f1})
    # else:
    #     model = model_path  # Assume model object was passed directly
    
    # Predict patches
    predictions = []
    for patch in patches_reshaped:
        patch_input = patch[np.newaxis, ..., np.newaxis] #/ 255.0
        patch_input = z_score_normalization(patch_input)
        pred = model.predict(patch_input, verbose=0)
        predictions.append(pred[0, ..., 0])
    
    # Reshape predictions
    predictions = np.array(predictions)
    predictions = predictions.reshape(n_patches_h, n_patches_w, patch_size, patch_size)
    
    # Reconstruct full image
    reconstructed = patchify.unpatchify(predictions, 
                                      (n_patches_h * patch_size, 
                                       n_patches_w * patch_size))
    
    # Create binary mask
    binary_mask = (reconstructed > threshold).astype(np.uint8) * 255
    
    # Clean up mask
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Reverse padding and cropping
    predicted_mask = reverse_padding_and_cropping(
        binary_mask,
        image.shape,
        petri_coords
    )
    
    # Create overlay
    overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay_image[predicted_mask > 0] = [255, 0, 0] 
    
    return predicted_mask, overlay_image


def predict_roots_optimized(image_path, model, patch_size=960, threshold=0.6):
    """
    Predicts root segments in a plant image using a trained U-Net model.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the trained model weights
        patch_size (int): Size of patches for processing (default: 960)
        threshold (float): Threshold for binary mask creation (default: 0.6)
    
    Returns:
        tuple: (petri_dish, unpadded_mask) where:
            - petri_dish: Extracted Petri dish image before padding
            - unpadded_mask: Binary mask corresponding to petri_dish
    """
    # Load the image
    image = cv2.imread(image_path, 0)
    
    # Extract petri dish
    petri_dish, _ = extract_petri_dish(image)
    
    # Store original dimensions
    original_height, original_width = petri_dish.shape
    
    # Add padding
    padded_petri = padder(petri_dish, patch_size)
    padded_height, padded_width = padded_petri.shape
    
    # Calculate padding amounts
    height_padding = padded_height - original_height
    width_padding = padded_width - original_width
    top_padding = height_padding // 2
    left_padding = width_padding // 2
    
    # Create patches
    patches, _ = create_patches(padded_petri, patch_size)
    
    # Prepare patches for prediction
    n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
    patches_reshaped = patches.reshape(-1, patch_size, patch_size)
    
    # Predict patches
    predictions = []
    for patch in patches_reshaped:
        patch_input = patch[np.newaxis, ..., np.newaxis]
        patch_input = z_score_normalization(patch_input)
        pred = model.predict(patch_input, verbose=0)
        predictions.append(pred[0, ..., 0])
    
    # Reshape predictions
    predictions = np.array(predictions)
    predictions = predictions.reshape(n_patches_h, n_patches_w, patch_size, patch_size)
    
    # Reconstruct full image
    reconstructed = patchify.unpatchify(predictions, 
                                      (n_patches_h * patch_size, 
                                       n_patches_w * patch_size))
    
    # Create binary mask
    binary_mask = (reconstructed > threshold).astype(np.uint8) * 255
    
    # Clean up mask
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove padding to match original petri dish dimensions
    unpadded_mask = binary_mask[top_padding:top_padding + original_height,
                               left_padding:left_padding + original_width]
    
    return petri_dish, unpadded_mask


# if __name__ == '__main__':  

    # # Build U-Net
    # model = unet_model(
    #     IMG_HEIGHT=patch_size,
    #     IMG_WIDTH=patch_size,
    #     IMG_CHANNELS=1
    # )

    # # Load the weight if it exists
    # # model.load_weights(f'soheil_6989490_unet_model_{patch_size}px.h5')
    # model.load_weights('best_model.weights.h5')

    # # Example usage:
    # image_path = 'task5_test_image.png'
    # model_path = f'soheil_6989490_unet_model_{patch_size}px.h5'

    # # Get predictions
    # mask, overlay = predict_roots(image_path, model)

    # # Visualize results
    # plt.figure(figsize=(25, 25))
    # plt.subplot(311)
    # plt.imshow(cv2.imread(image_path, 0), cmap='gray')
    # plt.title('Original Image', fontsize=14)
    # plt.subplot(312)    
    # plt.imshow(mask, cmap='gray')
    # plt.title('Predicted Mask', fontsize=14)
    # plt.subplot(313)
    # plt.imshow(overlay)
    # plt.title('Overlay', fontsize=14)
    # plt.show()

