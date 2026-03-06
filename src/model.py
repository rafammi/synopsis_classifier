import os
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, hamming_loss

plt.style.use('ggplot')

class genrePredictor():
    def __init__(self, num_classes: int, class_names: np.ndarray, max_tokens: int = 20000, output_sequence_length: int = 100, ):
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.class_names = class_names
        self.num_classes = num_classes
        self.model = None 
        self.history = None
        self.thresholds = None

    def initialize(self, X: np.ndarray, output_dim: int = 128, dropout_rate: float = 0.3):
        print("Initializing model...")
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=self.max_tokens,
            output_sequence_length=self.output_sequence_length
        )
        vectorize_layer.adapt(X) # passing only training dataset
        model = Sequential([
            vectorize_layer,
            Embedding(input_dim = self.max_tokens + 1, output_dim = output_dim, input_length = self.output_sequence_length),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(output_dim // 2, return_sequences = False,
                                kernel_regularizer = l2(1e-4),
                                recurrent_regularizer=l2(1e-4))),
            Dropout(dropout_rate),
            Dense(output_dim // 2, activation = "relu", kernel_regularizer=l2(1e-4)),
            Dense(self.num_classes, activation = "sigmoid")
    ])
        self.model = model

    def compile(self) -> None:
        if self.model:
            print("Compiling model with macro F1 metric...")
            self.model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                metrics=[tf.keras.metrics.F1Score(average="macro", threshold=0.5, name="f1_macro")]
            )
            self.model.summary()
        else:
            print("Model not declared yet!")

    def train(self, X, y, class_weights: np.ndarray = None, early_stopping_rounds: int = 10) -> np.ndarray:
        if self.model:
            print("Training model...")
            early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_rounds,
            restore_best_weights=True
            )

            history = self.model.fit(X, y, epochs = 50,
                                    batch_size= 64, 
                                    validation_split=0.2,
                                    class_weight = class_weights,
                                    callbacks=[early_stopping])
            self.history = history
            print("Model trained!")
        else:
            print("Model not declared yet!")

    def plot_training(self) -> None:
        if self.history:
            f, axs = plt.subplots(1,2)
            axs = axs.ravel()

            axs[0].plot(self.history.history["loss"])
            axs[0].plot(self.history.history["val_loss"])
            axs[0].set_title("Model Training Loss")
            axs[0].set_ylabel('Loss')
            axs[0].set_xlabel('Epoch')
            axs[0].legend(['Train', 'Validation'])

            axs[1].plot(self.history.history["f1_macro"])
            axs[1].plot(self.history.history["val_f1_macro"])
            axs[1].set_title("Model Training F1 (Macro)")
            axs[1].set_ylabel('F1')
            axs[1].set_xlabel('Epoch')
            axs[1].legend(['Train', 'Validation'])

            plt.show()
        else:
            print("model not trained yet!")

    def evaluate(self, X_test, y_test) -> Tuple[float,float,float]:
        if self.model:
            predicted = self.model.predict(X_test)
            thresholds = self.thresholds if self.thresholds is not None else 0.5
            predicted_labels = (predicted >= thresholds).astype(int)
            f1_macro = f1_score(y_test, predicted_labels, average="macro")
            f1_micro = f1_score(y_test, predicted_labels, average="micro")
            hamming_acc = 1 - hamming_loss(y_test, predicted_labels)
            return f1_macro, f1_micro, hamming_acc
        else:
            print("model not trained!")


    def tune_thresholds(self, y_prob_test: np.ndarray, y_test: np.ndarray, n_steps=50) -> None:
        n_classes = y_test.shape[1]
        thresholds = np.zeros(n_classes)

        for i in range(n_classes):
            best_t, best_f1 = 0.05, 0.0
            for t in np.linspace(0.05, 0.95, n_steps):
                preds = (y_prob_test[:, i] >= t).astype(int)
                f1 = f1_score(y_test[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            thresholds[i] = best_t

        self.thresholds = thresholds # shape (n_classes,) — broadcast directly against y_prob

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_prob = self.model.predict(X)
        return y_prob

    def predict(self, overview: str) -> pd.Series:
        if self.model:
            sample_text = tf.constant([overview])
            probs = self.model.predict(sample_text)[0]  # shape: (num_classes,)
            return pd.Series(probs, index=self.class_names).sort_values(ascending=False)
        else:
            print("Model not declared yet!")

    def save_model(self) -> None:
        if self.model:
            current_datetime = datetime.now()
            filename = f"lstm_classifier_{current_datetime.strftime('%Y%m%d_%H%M%S')}"
            path = Path(os.path.dirname(__file__)) / ".." / "models" / "lstm" / filename
            path.mkdir(parents=True, exist_ok=True)
            self.model.save(path / "model.keras")
            metadata = {
                "num_classes": self.num_classes,
                "class_names": list(self.class_names),
                "max_tokens": self.max_tokens,
                "output_sequence_length": self.output_sequence_length,
                "thresholds": self.thresholds.tolist() if self.thresholds is not None else None
            }
            with open(path / "metadata.json", "w") as f:
                json.dump(metadata, f)
            print(f"Model saved at {path}!")
        else:
            print("model not trained yet!")

    def load_model(self, path: str = None) -> None:
        if path is None:
            models_dir = Path(__file__).parent / ".." / "models" / "lstm"
            saved_models = sorted(models_dir.glob("lstm_classifier_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not saved_models:
                print("No saved models found!")
                return
            path = saved_models[0]
            print(f"Loading most recent model: {path.name}")
        path = Path(path)
        self.model = tf.keras.models.load_model(path / "model.keras")
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        self.num_classes = metadata["num_classes"]
        self.class_names = np.array(metadata["class_names"])
        self.max_tokens = metadata["max_tokens"]
        self.output_sequence_length = metadata["output_sequence_length"]
        self.thresholds = np.array(metadata["thresholds"]) if metadata.get("thresholds") else None
        print(f"Model loaded from {path}!")