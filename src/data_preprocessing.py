"""
Data preprocessing utilities for the EN3150 CNN Classification project.

This module contains reusable functions for:
- Loading the Fashion-MNIST dataset
- Combining the original train and test sets
- Creating a 70/15/15 train-validation-test split
- Normalizing image pixel values
- Adding the channel dimension required by CNN models
"""

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    return x_train, y_train, x_test, y_test


def combine_dataset(x_train, y_train, x_test, y_test):
    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    return X, y


# -------------------------------------------------------------------
# Train / Validation / Test Split
# -------------------------------------------------------------------

def split_dataset(X, y, random_state=42):
    # First split:
    # 70% training
    # 30% temporary set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y
    )

    # Second split:
    # 50% of the temporary set -> 15% validation
    # 50% of the temporary set -> 15% testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# -------------------------------------------------------------------
# Image Preprocessing
# -------------------------------------------------------------------

def normalize_images(X):
    return X.astype(np.float32) / 255.0


def add_channel_dimension(X):
    return np.expand_dims(X, axis=-1)


# -------------------------------------------------------------------
# Complete Preprocessing Pipeline
# -------------------------------------------------------------------

def preprocess_dataset(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    
    # Normalize images
    X_train = normalize_images(X_train)
    X_val = normalize_images(X_val)
    X_test = normalize_images(X_test)

    # Add channel dimension
    X_train = add_channel_dimension(X_train)
    X_val = add_channel_dimension(X_val)
    X_test = add_channel_dimension(X_test)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )