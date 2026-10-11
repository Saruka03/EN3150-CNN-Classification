import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def preprocess_for_efficientnet(X):
    """
    Adapts 28x28 grayscale Fashion-MNIST images to 96x96 3-channel RGB 
    images required by EfficientNet-B0.
    """
    # 1. Resize from 28x28 to 96x96
    X_resized = tf.image.resize(X, (96, 96))

    # 2. Convert 1-channel grayscale to 3-channel RGB
    X_rgb = tf.repeat(X_resized, 3, axis=-1)

    # 3. Scale pixel values to [0, 255] as expected by EfficientNet preprocessing
    X_scaled = X_rgb * 255.0
    return X_scaled


def create_efficientnet_b0_model():
    """
    Constructs a SOTA model using EfficientNet-B0 as a pretrained base 
    with a custom classification head for Fashion-MNIST (10 classes).
    """
    # Load EfficientNet-B0 pretrained on ImageNet without the top classification layer
    base_model = EfficientNetB0(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers initially for transfer learning
    base_model.trainable = False

    # Build the full model architecture
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Lambda(preprocess_for_efficientnet),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = create_efficientnet_b0_model()
    model.summary()
