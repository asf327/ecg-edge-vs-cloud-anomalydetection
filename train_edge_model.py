import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# configuration
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/edge"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

# setup
def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)

# load preprocessed data
def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

    X_val = np.load(os.path.join(PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_edge_model(input_shape=(3600, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(16, kernel_size=7, strides=2, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(1, activation="sigmoid")
    ])
    return model

def representative_data_gen(X_train):
    limit = min(100, len(X_train))
    for i in range(limit):
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]

def convert_to_tflite_int8(model, X_train, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data_gen(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved INT8 TFLite model to {output_path}")


def main():
    ensure_dirs()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("Shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    model = build_edge_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", 
                  metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

    checkpoint_path = os.path.join(MODEL_DIR, "best_edge_model.keras")
    final_model_path = os.path.join(MODEL_DIR, "edge_model.keras")
    tflite_ouput_path = os.path.join(MODEL_DIR, "edge_model_int8.tflite")

    callback_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callback_list,
        verbose=1
    )

    results = model.evaluate(X_test, y_test, verbose=1)
    print("\nTest Results:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    model.save(final_model_path)
    print(f"Saved trained model to {final_model_path}")

    convert_to_tflite_int8(model, X_train, tflite_ouput_path)

    history_path = os.path.join(MODEL_DIR, "training_history.npy")
    np.savez(history_path, **history.history)
    print(f"Saved training history to {history_path}")

if __name__ == "__main__":
    main()