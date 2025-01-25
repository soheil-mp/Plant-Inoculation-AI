
# Import the libraries
import os
import cv2
import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import os
import sys

# Add the paths to task5 and task7 directories and add them to Python path
task5_dir = os.path.join(os.getcwd(), '..', 'task5')
task7_dir = os.path.join(os.getcwd(), '..', 'task7')
import sys
sys.path.append(task5_dir)
sys.path.append(task7_dir)

# Import modules from task5 and task7
import inference
from RAA import RootArchitectureAnalyzer


def cv_inference(image_path):

    # Patch size
    patch_size = 960

    # Build U-Net
    model = inference.unet_model(
        IMG_HEIGHT=patch_size,
        IMG_WIDTH=patch_size,
        IMG_CHANNELS=1
    )

    # Load the weight if it exists
    # model.load_weights(f'soheil_6989490_unet_model_{patch_size}px.h5')
    model.load_weights('./../../task5/best_model.weights.h5')

    # Get predictions
    # mask, overlay = inference.predict_roots(image_path, model)
    image, mask = inference.predict_roots_optimized(image_path, model)

    # After getting petri_dish and mask from predict_roots_optimized
    analyzer = RootArchitectureAnalyzer()
    root_data, root_df, tip_coordinates = analyzer.process_image(mask=mask, petri_dish=image)

    # Visualize the results
    # analyzer.visualize_results(root_data, original_mask=mask, petri_dish=petri_dish)

    return image, mask, analyzer, root_df, root_data, tip_coordinates