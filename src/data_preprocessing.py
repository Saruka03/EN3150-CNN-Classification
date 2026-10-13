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
from pathlib import Path


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


if __name__ == "__main__":
    # Ensure the output directory exists
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Fashion-MNIST dataset...")
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    print("Combining dataset splits...")
    X, y = combine_dataset(x_train, y_train, x_test, y_test)

    print("Performing 70/15/15 stratified split...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print("Running preprocessing pipeline (normalization & channel expansion)...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_dataset(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # Save to .npz archive
    output_path = output_dir / "fashion_mnist_processed.npz"
    np.savez(
        output_path,
        x_train=X_train, y_train=y_train,
        x_val=X_val, y_val=y_val,
        x_test=X_test, y_test=y_test
    )
    print(f"Successfully generated and saved: {output_path}")
