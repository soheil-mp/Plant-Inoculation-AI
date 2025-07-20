"""
U-Net model implementation for plant root segmentation.

This module provides a U-Net architecture with MobileNet backbone
for efficient and accurate root segmentation.
"""

from typing import List, Optional, Any
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, concatenate, 
    BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class UNetModel:
    """U-Net model with MobileNet backbone for root segmentation."""
    
    def __init__(
        self, 
        img_height: int = 960, 
        img_width: int = 960, 
        img_channels: int = 1
    ):
        """
        Initialize U-Net model.
        
        Args:
            img_height: Input image height
            img_width: Input image width  
            img_channels: Number of input channels
        """
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.model = None
    
    def build_model(self) -> Model:
        """
        Build the U-Net model with MobileNet backbone.
        
        Returns:
            Compiled Keras model
        """
        # Input layer 
        inputs = Input((self.img_height, self.img_width, self.img_channels))
        
        # Convert grayscale to RGB and resize for MobileNet
        if self.img_channels == 1:
            x = tf.keras.layers.Lambda(
                lambda x: tf.image.grayscale_to_rgb(x)
            )(inputs)
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
        s4 = base_model.get_layer('conv_pw_11_relu').output  # 14x14
        
        # Bridge
        bridge = base_model.get_layer('conv_pw_13_relu').output  # 7x7
        
        # Decoder path with batch normalization and dropout
        u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(
            bridge
        )
        u6 = concatenate([u6, s4])
        u6 = BatchNormalization()(u6)
        u6 = Dropout(0.3)(u6)
        c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
        
        u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, s3])
        u7 = BatchNormalization()(u7)
        u7 = Dropout(0.3)(u7)
        c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        
        u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, s2])
        u8 = BatchNormalization()(u8)
        u8 = Dropout(0.3)(u8)
        c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
        
        u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, s1])
        u9 = BatchNormalization()(u9)
        u9 = Dropout(0.3)(u9)
        c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        
        # Resize back to original dimensions
        c9_resized = tf.keras.layers.Resizing(
            self.img_height, self.img_width
        )(c9)
        
        # Output layer
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9_resized)
        
        # Create model
        model = Model(inputs, outputs)
        
        self.model = model
        return model
    
    def compile_model(
        self, 
        optimizer: str = 'adam',
        loss: str = 'combined_loss',
        metrics: Optional[List[Any]] = None
    ) -> None:
        """
        Compile the model with loss function and optimizer.
        
        Args:
            optimizer: Optimizer name
            loss: Loss function name
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ['accuracy', self.dice_coefficient, self.f1_score]
        
        if loss == 'combined_loss':
            loss_fn = self.combined_loss
        else:
            loss_fn = loss
            
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
    
    @staticmethod
    def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-5) -> tf.Tensor:
        """Dice loss for segmentation."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice_score
    
    @staticmethod
    def focal_loss(
        y_true: tf.Tensor, 
        y_pred: tf.Tensor, 
        gamma: float = 2.0, 
        alpha: float = 0.75
    ) -> tf.Tensor:
        """Focal loss for addressing class imbalance."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # Calculate focal loss
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        
        # Calculate cross entropy
        ce_loss = -tf.math.log(p_t)
        
        return tf.reduce_mean(focal_weight * ce_loss)
    
    @staticmethod
    def tversky_loss(
        y_true: tf.Tensor, 
        y_pred: tf.Tensor, 
        alpha: float = 0.7, 
        beta: float = 0.3,
        smooth: float = 1e-5
    ) -> tf.Tensor:
        """Tversky loss for handling false positives and negatives."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        true_pos = tf.reduce_sum(y_true * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)
        
        tversky_score = (true_pos + smooth) / (
            true_pos + alpha * false_neg + beta * false_pos + smooth
        )
        
        return 1.0 - tversky_score
    
    @staticmethod
    def boundary_loss(
        y_true: tf.Tensor, 
        y_pred: tf.Tensor, 
        theta: float = 1.5
    ) -> tf.Tensor:
        """Boundary-aware loss to focus on root edges."""
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
    
    def combined_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Comprehensive loss function for plant root segmentation."""
        # Cast inputs to float32 and clip predictions
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # Weight factors for each loss component
        dice_weight = tf.constant(1.0, dtype=tf.float32)
        focal_weight = tf.constant(2.5, dtype=tf.float32)
        tversky_weight = tf.constant(1.5, dtype=tf.float32)
        boundary_weight = tf.constant(0.5, dtype=tf.float32)
        
        try:
            # Calculate individual losses
            d_loss = self.dice_loss(y_true, y_pred, smooth=1e-5)
            f_loss = self.focal_loss(y_true, y_pred, gamma=3.0, alpha=0.75)
            t_loss = self.tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3)
            bound_loss = self.boundary_loss(y_true, y_pred, theta=1.5)
            
            # Check for NaN values
            losses = [d_loss, f_loss, t_loss, bound_loss]
            for loss in losses:
                if tf.reduce_any(tf.math.is_nan(loss)):
                    return self.dice_loss(y_true, y_pred, smooth=1e-5)
            
            # Combine losses with weights
            total_loss = (
                dice_weight * d_loss +
                focal_weight * f_loss +
                tversky_weight * t_loss +
                boundary_weight * bound_loss
            )
            
            # Final NaN check
            if tf.reduce_any(tf.math.is_nan(total_loss)):
                return self.dice_loss(y_true, y_pred, smooth=1e-5)
                
            return total_loss
            
        except Exception:
            return self.dice_loss(y_true, y_pred, smooth=1e-5)
    
    @staticmethod
    def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Dice coefficient metric."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        
        return (2.0 * intersection + 1e-5) / (union + 1e-5)
    
    @staticmethod
    def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """F1 score metric."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Convert predictions to binary
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        
        # Calculate precision and recall
        true_positives = tf.reduce_sum(y_true * y_pred_binary)
        predicted_positives = tf.reduce_sum(y_pred_binary)
        actual_positives = tf.reduce_sum(y_true)
        
        precision = true_positives / (predicted_positives + 1e-7)
        recall = true_positives / (actual_positives + 1e-7)
        
        return 2 * (precision * recall) / (precision + recall + 1e-7)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is not None:
            self.model.save_weights(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load pre-trained model weights."""
        if self.model is not None:
            self.model.load_weights(filepath)


def unet_model(
    IMG_HEIGHT: int = 960, 
    IMG_WIDTH: int = 960, 
    IMG_CHANNELS: int = 1
) -> Model:
    """
    Convenience function to create U-Net model.
    
    Args:
        IMG_HEIGHT: Input image height
        IMG_WIDTH: Input image width
        IMG_CHANNELS: Number of input channels
        
    Returns:
        Compiled U-Net model
    """
    unet = UNetModel(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    model = unet.build_model()
    unet.compile_model()
    return model 