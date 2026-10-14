import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def preprocess_for_mobilenet(X):
    """
    Adapts 28x28 grayscale Fashion-MNIST images to 96x96 3-channel RGB 
    images required by MobileNetV2, and normalizes pixel values.
    """
    # 1. Resize from 28x28 to 96x96 (Minimum requirement for MobileNetV2 is 32x32, 96x96 is optimal)
    X_resized = tf.image.resize(X, (96, 96))

    # 2. Convert 1-channel grayscale to 3-channel RGB by duplicating the channel dimension
    X_rgb = tf.repeat(X_resized, 3, axis=-1)

    # 3. Scale pixel values to [-1, 1] as expected by MobileNetV2 preprocess_input / scaling
    X_normalized = (X_rgb / 127.5) - 1.0
    return X_normalized


def create_mobilenet_v2_model():
    """
    Constructs a SOTA model using MobileNetV2 as a pretrained base 
    with a custom classification head for Fashion-MNIST (10 classes).
    """
    # Load MobileNetV2 pretrained on ImageNet without the top classification layer
    base_model = MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers initially for transfer learning
    base_model.trainable = False

    # Build the full model architecture
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # Lambda layer to apply input adaptation on-the-fly during training/inference
        layers.Lambda(preprocess_for_mobilenet),
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
    model = create_mobilenet_v2_model()
    model.summary()
