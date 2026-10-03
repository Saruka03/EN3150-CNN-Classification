from tensorflow import keras  # type: ignore[import-not-found]
from tensorflow.keras import layers  # type: ignore[import-not-found]

# -------------------------------------------------------------------------
# Model configuration
# -------------------------------------------------------------------------

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

CONV_FILTERS = [16, 32, 64]
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_UNITS = 64

LEARNING_RATE = 0.001

# -------------------------------------------------------------------------
# Model creation
# -------------------------------------------------------------------------

def build_model_A(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
    learning_rate=LEARNING_RATE
):
    
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            # Convolution block 1
            layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv1"
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool1"
            ),

            # Convolution block 2
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv2"
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool2"
            ),

            # Convolution block 3
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv3"
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool3"
            ),

            # Classification head
            layers.Flatten(name="flatten"),

            layers.Dense(
                units=64,
                activation="relu",
                name="dense1"
            ),

            layers.Dense(
                units=num_classes,
                activation="softmax",
                name="output"
            ),
        ],
        name="Model_A_Standard_CNN"
    )

    # Compile using Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------------------------------------------------
# Parameter calculation
# -------------------------------------------------------------------------

def calculate_parameters(input_shape=INPUT_SHAPE):
    
    # Conv1
    conv1_params = (3 * 3 * 1 * 16) + 16

    # Conv2
    conv2_params = (3 * 3 * 16 * 32) + 32

    # Conv3
    conv3_params = (3 * 3 * 32 * 64) + 64

    flattened_units = 3 * 3 * 64

    # Dense layer
    dense_params = (flattened_units * 64) + 64

    # Output layer
    output_params = (64 * 10) + 10

    total_params = (
        conv1_params
        + conv2_params
        + conv3_params
        + dense_params
        + output_params
    )

    return {
        "conv1": conv1_params,
        "conv2": conv2_params,
        "conv3": conv3_params,
        "dense1": dense_params,
        "output": output_params,
        "total": total_params,
    }


# -------------------------------------------------------------------------
# Model summary
# -------------------------------------------------------------------------

def print_model_info(model):
    
    print("\n" + "=" * 70)
    print("MODEL A - STANDARD CNN")
    print("=" * 70)

    model.summary()

    manual_params = calculate_parameters()
    keras_params = model.count_params()

    print("\nManual parameter calculation:")
    print("-" * 70)

    for layer_name, params in manual_params.items():
        print(f"{layer_name:15s}: {params:,}")

    print("\nKeras parameter count:")
    print(f"{keras_params:,}")

    print("\nParameter verification:")

    if manual_params["total"] == keras_params:
        print("PASS - Manual and Keras parameter counts match.")
    else:
        print("WARNING - Parameter counts do not match.")

    print("=" * 70)


# -------------------------------------------------------------------------
# Model creation when this file is executed directly
# -------------------------------------------------------------------------

if __name__ == "__main__":

    model_A = build_model_A()

    print_model_info(model_A)