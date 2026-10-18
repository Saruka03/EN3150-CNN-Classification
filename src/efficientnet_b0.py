"""
Standalone Python script converted from the corresponding EN3150 Assignment 03 notebook.
Run from the repository root or inside 'notebooks/' so data/processed/fashion_mnist_processed.npz is found.
"""

# # EN3150 Assignment 03 — SOTA Model 2: EfficientNetB0
# ## Fashion-MNIST Transfer Learning
#
# This script evaluates EfficientNetB0 for Fashion-MNIST classification using transfer learning from pretrained ImageNet weights.
# The same preprocessed training, validation, and test splits used for the custom CNNs and MobileNetV2 are retained.
# Images are resized to 64 × 64 pixels and converted to three channels.
# Training consists of two stages: training a new classification head with the backbone frozen,
# followed by fine-tuning selected pretrained layers with a lower learning rate.

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from pathlib import Path
import os
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Fix the working directory to the repository root if running from inside 'notebooks/'
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

# Reproducibility setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

DATA_PATH = Path("data/processed/fashion_mnist_processed.npz")
OUTPUT_DIR = Path("outputs/sota")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset path: {DATA_PATH.resolve()}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH.resolve()}. Ensure the dataset is present."
    )

# Verify NPZ archive contents and load using lowercase keys
with np.load(DATA_PATH) as data:
    print("\nFiles found in archive:", data.files)
    X_train = data["x_train"]
    y_train = data["y_train"]
    X_val = data["x_val"]
    y_val = data["y_val"]
    X_test = data["x_test"]
    y_test = data["y_test"]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")


# ## Load, resize, and prepare shared dataset tensors

def prepare_and_resize_images(images, target_size=(64, 64)):
    # Convert to float32 and 3 channels if needed
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)

    images = images.astype(np.float32)

    # Resize images from 28x28 to 64x64 using TensorFlow
    images_resized = tf.image.resize(images, target_size).numpy()
    return images_resized


X_train_prep = prepare_and_resize_images(X_train, (64, 64))
X_val_prep = prepare_and_resize_images(X_val, (64, 64))
X_test_prep = prepare_and_resize_images(X_test, (64, 64))

print("Prepared and resized shapes:")
print(
    f"X_train: {X_train_prep.shape}, Range: [{X_train_prep.min():.1f}, {X_train_prep.max():.1f}]")
print(
    f"X_val:   {X_val_prep.shape}, Range: [{X_val_prep.min():.1f}, {X_val_prep.max():.1f}]")
print(
    f"X_test:  {X_test_prep.shape}, Range: [{X_test_prep.min():.1f}, {X_test_prep.max():.1f}]")


# ## Build EfficientNetB0

MODEL_NAME = "EfficientNetB0"
MODEL_STEM = "efficientnetb0"
NUM_CLASSES = 10

BATCH_SIZE = 128
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LAST_LAYERS = 20


def preprocess_batch(images, labels):
    # EfficientNetB0 includes rescaling inside the Keras application.
    # Input images should remain in the [0, 255] range.
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int64)
    return images, labels


def make_dataset(images, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(images), 10000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(
        preprocess_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(X_train_prep, y_train, training=True)
val_ds = make_dataset(X_val_prep, y_val)
test_ds = make_dataset(X_test_prep, y_test)

# Load pretrained EfficientNetB0.
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(64, 64, 3)
)

backbone.trainable = False

inputs = tf.keras.Input(shape=(64, 64, 3), name="image")
x = backbone(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="class_predictions"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="EfficientNetB0_FashionMNIST"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Total parameters:", f"{model.count_params():,}")
print(
    "Initially trainable parameters:",
    f"{sum(np.prod(v.shape) for v in model.trainable_weights):,}"
)


# ## Train and fine-tune EfficientNetB0

best_model_path = OUTPUT_DIR / "efficientnetb0_best.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        mode="min",
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# Phase 1: Train the classification head.
start = time.perf_counter()

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

head_seconds = time.perf_counter() - start

# Phase 2: Fine-tune selected backbone layers.
backbone.trainable = True

for layer in backbone.layers[:-FINE_TUNE_LAST_LAYERS]:
    layer.trainable = False

# Keep Batch Normalization layers frozen.
for layer in backbone.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(
    "Trainable parameters during fine-tuning:",
    f"{sum(np.prod(v.shape) for v in model.trainable_weights):,}"
)

start = time.perf_counter()

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

finetune_seconds = time.perf_counter() - start

total_training_seconds = head_seconds + finetune_seconds

total_completed_epochs = (
    len(history_head.history["loss"])
    + len(history_finetune.history["loss"])
)

average_epoch_seconds = (
    total_training_seconds / total_completed_epochs
)

print(f"Head training time: {head_seconds:.2f} seconds")
print(f"Fine-tuning time: {finetune_seconds:.2f} seconds")
print(f"Total training time: {total_training_seconds:.2f} seconds")
print(f"Completed epochs: {total_completed_epochs}")
print(f"Average time per epoch: {average_epoch_seconds:.2f} seconds")


# ## Plot training and validation curves

def combine_histories(h1, h2):
    combined = {}
    for key in set(h1.history) | set(h2.history):
        combined[key] = (
            h1.history.get(key, []) + h2.history.get(key, [])
        )
    return combined


hist = combine_histories(history_head, history_finetune)
epochs = range(1, len(hist["loss"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, hist["loss"], label="Training Loss")
plt.plot(epochs, hist["val_loss"], label="Validation Loss")
plt.title("EfficientNetB0 - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "efficientnetb0_loss.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, hist["accuracy"], label="Training Accuracy")
plt.plot(epochs, hist["val_accuracy"], label="Validation Accuracy")
plt.title("EfficientNetB0 - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "efficientnetb0_accuracy.png", dpi=150)
plt.show()

pd.DataFrame(hist).to_csv(
    OUTPUT_DIR / "efficientnetb0_training_history.csv",
    index=False
)


# ## Final test-set evaluation

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)
y_true = np.asarray(y_test).reshape(-1)

weighted_precision = precision_score(
    y_true, y_pred, average="weighted", zero_division=0
)
weighted_recall = recall_score(
    y_true, y_pred, average="weighted", zero_division=0
)
macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4%}")
print(f"Weighted precision: {weighted_precision:.4f}")
print(f"Weighted recall: {weighted_recall:.4f}")
print(f"Macro precision: {macro_precision:.4f}")
print(f"Macro recall: {macro_recall:.4f}")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=np.arange(10),
    target_names=class_names,
    zero_division=0,
    digits=4
))


# ## Confusion matrix

cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("EfficientNetB0 - Confusion Matrix")
plt.colorbar()

ticks = np.arange(10)
plt.xticks(ticks, class_names, rotation=45, ha="right")
plt.yticks(ticks, class_names)

plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "efficientnetb0_confusion_matrix.png",
    dpi=150
)
plt.show()


# ## Save model and export metrics

final_model_path = OUTPUT_DIR / "efficientnetb0_final.keras"
model.save(final_model_path)

model_size_bytes = final_model_path.stat().st_size
model_size_kb = model_size_bytes / 1024
model_size_mb = model_size_bytes / (1024 ** 2)

total_params = model.count_params()
trainable_params = sum(
    int(np.prod(v.shape)) for v in model.trainable_weights
)

summary = {
    "model": MODEL_NAME,
    "input_resolution": "64x64x3",
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "total_parameters": int(total_params),
    "trainable_parameters_after_finetuning": int(trainable_params),
    "model_size_bytes": int(model_size_bytes),
    "model_size_kb": round(model_size_kb, 2),
    "model_size_mb": round(model_size_mb, 4),
    "completed_epochs": int(total_completed_epochs),
    "head_training_seconds": round(head_seconds, 2),
    "fine_tuning_seconds": round(finetune_seconds, 2),
    "total_training_seconds": round(total_training_seconds, 2),
    "average_epoch_seconds": round(average_epoch_seconds, 2),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall)
}

summary_path = OUTPUT_DIR / "efficientnetb0_metrics.json"
summary_path.write_text(json.dumps(summary, indent=2))

print(f"Model saved: {final_model_path}")
print(f"Metrics saved: {summary_path}")
