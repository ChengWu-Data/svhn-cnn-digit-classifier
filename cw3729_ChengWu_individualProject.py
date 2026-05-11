"""
Individual Project: CNN Classifier for Street View House Numbers (SVHN)

This file trains and evaluates a convolutional neural network on the SVHN
Format 2 cropped digit dataset.

Expected dataset files:
    data/train_32x32.mat
    data/test_32x32.mat

The script can download the files automatically if they are missing.
The dataset files are not meant to be submitted with the project.
"""

from pathlib import Path
import urllib.request
import time
import random

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# I keep these folders simple because the report needs the figures later.
DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

TRAIN_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
TEST_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

TRAIN_FILE = DATA_DIR / "train_32x32.mat"
TEST_FILE = DATA_DIR / "test_32x32.mat"

RANDOM_SEED = 42
IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


def make_project_folders():
    """Create local folders used by the script."""
    DATA_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)


def set_reproducibility(seed=RANDOM_SEED):
    """
    Set random seeds.

    This does not make every GPU run perfectly identical, but it keeps the
    results much more stable from one run to the next.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def download_file(url, output_path):
    """
    Download one dataset file if it is not already available.

    I do not put the dataset in GitHub because the assignment says not to
    submit the data. The code only keeps a local copy for running the model.
    """
    if output_path.exists():
        print(f"Found {output_path}.")
        return

    print(f"Downloading {output_path.name}...")
    print(f"Source: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}.")


def prepare_dataset_files():
    """Make sure the two SVHN files exist locally."""
    download_file(TRAIN_URL, TRAIN_FILE)
    download_file(TEST_URL, TEST_FILE)


def convert_svhn_labels(y):
    """
    Convert SVHN labels into digits 0-9.

    SVHN uses label 10 for the digit 0. That is easy to miss, and if it is
    not fixed here, the class labels do not match the model output.
    """
    y = y.reshape(-1).astype("int64")
    y[y == 10] = 0
    return y


def load_svhn_mat_file(path):
    """
    Load one SVHN .mat file.

    In the .mat file, images are stored as:
        height, width, channels, number_of_images

    TensorFlow/Keras expects:
        number_of_images, height, width, channels

    So the image array has to be transposed before training.
    """
    mat = sio.loadmat(path)

    x = mat["X"]
    y = mat["y"]

    x = np.transpose(x, (3, 0, 1, 2)).astype("float32")
    x = x / 255.0

    y = convert_svhn_labels(y)

    return x, y


def load_data():
    """Load the training and test data."""
    print("Loading SVHN data...")
    x_train, y_train = load_svhn_mat_file(TRAIN_FILE)
    x_test, y_test = load_svhn_mat_file(TEST_FILE)

    print(f"Training images: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images:     {x_test.shape}")
    print(f"Test labels:     {y_test.shape}")

    return x_train, y_train, x_test, y_test


def plot_sample_training_images(x_train, y_train):
    """Save a grid of training images for the report."""
    plt.figure(figsize=(8, 5))

    indices = np.random.choice(len(x_train), size=20, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(4, 5, i + 1)
        plt.imshow(x_train[idx])
        plt.title(f"Label: {y_train[idx]}")
        plt.axis("off")

    plt.suptitle("Sample SVHN Training Images")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sample_training_images.png", dpi=200)
    plt.close()


def build_baseline_model():
    """
    Build a simple starting model.

    This is close in spirit to a basic MNIST CNN: a few convolution layers,
    max pooling, one dense layer, and a softmax output. I use it as the
    starting point because the task is digit classification, but SVHN is
    more difficult than MNIST, so this model is not the final version.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=IMAGE_SHAPE),

            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="baseline_svhn_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_final_model():
    """
    Build the final CNN model.

    The final model is deeper than the baseline model. I added repeated
    convolution blocks because SVHN has color, background noise, different
    fonts, and uneven lighting. Batch normalization helps training stay
    stable, and dropout keeps the model from relying too much on the
    training images only.
    """
    inputs = keras.Input(shape=IMAGE_SHAPE)

    # Light augmentation. I keep it small because the image is only 32x32.
    x = layers.RandomTranslation(height_factor=0.08, width_factor=0.08)(inputs)
    x = layers.RandomZoom(height_factor=0.08, width_factor=0.08)(x)
    x = layers.RandomContrast(factor=0.10)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="block1_conv1")(x)
    x = layers.BatchNormalization(name="block1_bn1")(x)
    x = layers.Activation("relu", name="block1_relu1")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="block1_conv2")(x)
    x = layers.BatchNormalization(name="block1_bn2")(x)
    x = layers.Activation("relu", name="block1_relu2")(x)

    x = layers.MaxPooling2D((2, 2), name="block1_pool")(x)
    x = layers.Dropout(0.20, name="block1_dropout")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="block2_conv1")(x)
    x = layers.BatchNormalization(name="block2_bn1")(x)
    x = layers.Activation("relu", name="block2_relu1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="block2_conv2")(x)
    x = layers.BatchNormalization(name="block2_bn2")(x)
    x = layers.Activation("relu", name="block2_relu2")(x)

    x = layers.MaxPooling2D((2, 2), name="block2_pool")(x)
    x = layers.Dropout(0.30, name="block2_dropout")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="block3_conv1")(x)
    x = layers.BatchNormalization(name="block3_bn1")(x)
    x = layers.Activation("relu", name="block3_relu1")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="block3_conv2")(x)
    x = layers.BatchNormalization(name="block3_bn2")(x)
    x = layers.Activation("relu", name="block3_relu2")(x)

    x = layers.MaxPooling2D((2, 2), name="block3_pool")(x)
    x = layers.Dropout(0.40, name="block3_dropout")(x)

    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(256, use_bias=False, name="dense_256")(x)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.Activation("relu", name="dense_relu")(x)
    x = layers.Dropout(0.50, name="dense_dropout")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="digit_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="final_svhn_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks(model_name):
    """
    Training callbacks.

    I save the best validation model, lower the learning rate when validation
    loss gets stuck, and stop early if the model is no longer improving.
    """
    checkpoint_path = MODEL_DIR / f"best_{model_name}.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    return callbacks


def plot_learning_curves(history, output_name, title):
    """Save loss and accuracy curves from model training."""
    history_dict = history.history

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict["loss"], label="Training loss")
    plt.plot(history_dict["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict["accuracy"], label="Training accuracy")
    plt.plot(history_dict["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name, dpi=200)
    plt.close()


def train_model(model, model_name, x_train, y_train, epochs, batch_size):
    """Train one model and return the fitted model and training history."""
    print("\n" + "=" * 70)
    print(f"Training model: {model_name}")
    print("=" * 70)
    model.summary()

    start_time = time.time()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )

    elapsed = time.time() - start_time
    print(f"Training time for {model_name}: {elapsed / 60:.2f} minutes")

    return model, history


def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model on the official SVHN test set."""
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy


def plot_confusion_matrix(model, x_test, y_test):
    """Save a confusion matrix for the final model."""
    probabilities = model.predict(x_test, batch_size=256, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    cm = confusion_matrix(y_test, predictions, labels=list(range(NUM_CLASSES)))

    plt.figure(figsize=(8, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(NUM_CLASSES)))
    display.plot(values_format="d", cmap="Blues", ax=plt.gca(), colorbar=False)
    plt.title("Confusion Matrix on SVHN Test Set")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    return predictions


def plot_sample_predictions(x_test, y_test, predictions):
    """Save a small grid of correct and incorrect predictions."""
    correct = np.where(predictions == y_test)[0]
    incorrect = np.where(predictions != y_test)[0]

    chosen_correct = np.random.choice(correct, size=8, replace=False)
    chosen_incorrect = np.random.choice(incorrect, size=8, replace=False)
    chosen = np.concatenate([chosen_correct, chosen_incorrect])

    plt.figure(figsize=(10, 5))

    for i, idx in enumerate(chosen):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x_test[idx])
        plt.title(f"True: {y_test[idx]} | Pred: {predictions[idx]}")
        plt.axis("off")

    plt.suptitle("Sample Predictions: Correct Examples First, Then Mistakes")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sample_predictions.png", dpi=200)
    plt.close()


def normalize_feature_map(feature_map):
    """
    Normalize one feature map for plotting.

    This is only for visualization. It does not change the trained model.
    """
    feature_map = feature_map - feature_map.min()
    denominator = feature_map.max() + 1e-8
    feature_map = feature_map / denominator
    return feature_map


def plot_feature_maps(model, image, layer_name, output_name, max_maps=16):
    """
    Save feature maps from one convolution layer.

    The goal is not to make every filter look beautiful. The report needs to
    show that earlier and later convolution layers respond to different image
    patterns, so a small grid of activations is enough.
    """
    feature_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output,
    )

    feature_maps = feature_model.predict(image[np.newaxis, ...], verbose=0)[0]
    number_of_maps = min(max_maps, feature_maps.shape[-1])

    grid_size = int(np.ceil(np.sqrt(number_of_maps)))
    plt.figure(figsize=(8, 8))

    for i in range(number_of_maps):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(normalize_feature_map(feature_maps[:, :, i]), cmap="viridis")
        plt.axis("off")

    plt.suptitle(f"Feature Maps from {layer_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name, dpi=200)
    plt.close()


def save_feature_map_examples(model, x_test):
    """Save feature map examples from several convolution layers."""
    sample_image = x_test[0]

    plot_feature_maps(
        model=model,
        image=sample_image,
        layer_name="block1_conv1",
        output_name="feature_maps_block1_conv1.png",
        max_maps=16,
    )

    plot_feature_maps(
        model=model,
        image=sample_image,
        layer_name="block2_conv1",
        output_name="feature_maps_block2_conv1.png",
        max_maps=16,
    )

    plot_feature_maps(
        model=model,
        image=sample_image,
        layer_name="block3_conv1",
        output_name="feature_maps_block3_conv1.png",
        max_maps=16,
    )


def write_experiment_summary(
    baseline_test_loss,
    baseline_test_accuracy,
    final_test_loss,
    final_test_accuracy,
):
    """Write a simple text summary that can be used while writing the report."""
    summary_path = RESULT_DIR / "experiment_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("SVHN CNN Individual Project Summary\n")
        file.write("===================================\n\n")

        file.write("Dataset\n")
        file.write("-------\n")
        file.write("SVHN Format 2 cropped digit dataset.\n")
        file.write("Training file: train_32x32.mat\n")
        file.write("Test file: test_32x32.mat\n")
        file.write("Digit label 10 was converted to digit 0.\n\n")

        file.write("Baseline model\n")
        file.write("--------------\n")
        file.write(f"Test loss: {baseline_test_loss:.4f}\n")
        file.write(f"Test accuracy: {baseline_test_accuracy:.4f}\n\n")

        file.write("Final model\n")
        file.write("-----------\n")
        file.write(f"Test loss: {final_test_loss:.4f}\n")
        file.write(f"Test accuracy: {final_test_accuracy:.4f}\n\n")

        file.write("Generated figures\n")
        file.write("-----------------\n")
        file.write("figures/sample_training_images.png\n")
        file.write("figures/baseline_learning_curves.png\n")
        file.write("figures/final_learning_curves.png\n")
        file.write("figures/confusion_matrix.png\n")
        file.write("figures/sample_predictions.png\n")
        file.write("figures/feature_maps_block1_conv1.png\n")
        file.write("figures/feature_maps_block2_conv1.png\n")
        file.write("figures/feature_maps_block3_conv1.png\n\n")

        file.write("Note\n")
        file.write("----\n")
        file.write(
            "The final accuracy can vary slightly by machine, TensorFlow version, "
            "and random initialization. If the result is just below 91%, rerunning "
            "the final model or increasing the number of epochs can help.\n"
        )

    print(f"Saved summary to {summary_path}")


def main():
    """Run the whole project from data loading to final evaluation."""
    make_project_folders()
    set_reproducibility()

    prepare_dataset_files()
    x_train, y_train, x_test, y_test = load_data()

    plot_sample_training_images(x_train, y_train)

    baseline_model = build_baseline_model()
    baseline_model, baseline_history = train_model(
        model=baseline_model,
        model_name="baseline_svhn_cnn",
        x_train=x_train,
        y_train=y_train,
        epochs=12,
        batch_size=128,
    )

    plot_learning_curves(
        history=baseline_history,
        output_name="baseline_learning_curves.png",
        title="Baseline CNN Learning Curves",
    )

    baseline_test_loss, baseline_test_accuracy = evaluate_model(
        baseline_model,
        x_test,
        y_test,
    )

    final_model = build_final_model()
    final_model, final_history = train_model(
        model=final_model,
        model_name="svhn_cnn",
        x_train=x_train,
        y_train=y_train,
        epochs=35,
        batch_size=128,
    )

    plot_learning_curves(
        history=final_history,
        output_name="final_learning_curves.png",
        title="Final CNN Learning Curves",
    )

    final_test_loss, final_test_accuracy = evaluate_model(
        final_model,
        x_test,
        y_test,
    )

    final_model.save(MODEL_DIR / "best_svhn_cnn.keras")

    predictions = plot_confusion_matrix(final_model, x_test, y_test)
    plot_sample_predictions(x_test, y_test, predictions)
    save_feature_map_examples(final_model, x_test)

    write_experiment_summary(
        baseline_test_loss=baseline_test_loss,
        baseline_test_accuracy=baseline_test_accuracy,
        final_test_loss=final_test_loss,
        final_test_accuracy=final_test_accuracy,
    )

    print("\nDone.")
    print(f"Final test accuracy: {final_test_accuracy:.4f}")

    if final_test_accuracy >= 0.91:
        print("The final model reached the required 91% test accuracy.")
    else:
        print(
            "The final model did not reach 91% in this run. "
            "Try rerunning, training for more epochs, or tuning dropout/learning rate."
        )


if __name__ == "__main__":
    main()
