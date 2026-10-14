import tensorflow as tf
from tensorflow.keras import layers, models


def create_model_b():
    """
    Constructs an optimized CNN architecture for Fashion-MNIST 
    with total trainable parameters kept strictly under 100,000.
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu',
                      padding='same', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Use Global Average Pooling to drastically reduce parameter count
        # instead of a massive Flatten -> Dense layer transition
        layers.GlobalAveragePooling2D(),

        # Dense classification layers
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
    model_b = create_model_b()
    model_b.summary()
