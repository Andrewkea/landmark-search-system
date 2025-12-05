"""
Configuration file for Landmark Search System
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Default cities
DEFAULT_CITIES = {
    'EKB': 'Екатеринбург',
    'NN': 'Нижний Новгород', 
    'Vladimir': 'Владимир',
    'Yaroslavl': 'Ярославль'
}

# Image cleaning thresholds
CLEAN_THRESHOLDS = {
    'saturation': 15,
    'laplacian': 50,
    'brightness_low': 30,
    'brightness_high': 220,
    'min_resolution': 150,
    'min_aspect_ratio': 0.3,
    'max_aspect_ratio': 3.0,
    'edge_density_max': 0.6,
    'colorfulness': 8,
    'entropy': 4.5,
    'contrast': 25,
    'noise_threshold': 30,
    'min_unique_colors': 100,
    'max_file_size_kb': 50,
}
