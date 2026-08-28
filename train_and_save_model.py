import os
import json
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import gensim.downloader as api
from tqdm import tqdm

def main():
    os.makedirs("model_artifacts", exist_ok=True)
    
    print("1. Loading datasets...")
    train_df = pd.read_csv("train_data_final.csv")
    val_df = pd.read_csv("val_data_final.csv")
    test_df = pd.read_csv("test_data_final.csv")
    raw_df = pd.read_csv("stock_trend.csv")

    label2idx = {"downtrend": 0, "flat": 1, "uptrend": 2}
    idx2label = {v: k for k, v in label2idx.items()}

    y_train = train_df["Trend"].map(label2idx).values
    y_val = val_df["Trend"].map(label2idx).values
    y_test = test_df["Trend"].map(label2idx).values

    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    print("2. Loading GloVe embeddings (glove-wiki-gigaword-50)...")
    glove_model = api.load("glove-wiki-gigaword-50")
    embedding_dim = glove_model.vector_size
    max_len = 30
    tokenizer = TweetTokenizer()

    def texts_to_embeddings(texts, max_len, glove_model, emb_dim):
        seqs = []
        for text in tqdm(texts, desc="Embedding"):
            tokens = tokenizer.tokenize(str(text).lower())
            vectors = []
            for tok in tokens:
                if tok in glove_model:
                    vectors.append(glove_model[tok])
                else:
                    # OOV handling with Gaussian vector
                    vectors.append(np.random.normal(scale=0.6, size=emb_dim))
            if len(vectors) == 0:
                vectors.append(np.zeros(emb_dim))
            seqs.append(np.array(vectors))

        return pad_sequences(
            seqs,
            maxlen=max_len,
            dtype="float32",
            padding="post",
            truncating="post"
        )

    print("3. Vectorizing text datasets...")
    X_train = texts_to_embeddings(train_df["Cleaned_Title"], max_len, glove_model, embedding_dim)
    X_val = texts_to_embeddings(val_df["Cleaned_Title"], max_len, glove_model, embedding_dim)
    X_test = texts_to_embeddings(test_df["Cleaned_Title"], max_len, glove_model, embedding_dim)

    print("4. Building LSTM architecture...")
    inputs = Input(shape=(max_len, embedding_dim), name="input_embeddings")
    x = LSTM(64, return_sequences=True, name="lstm_layer")(inputs)
    x = Flatten(name="flatten_layer")(x)
    x = Dense(32, activation="relu", name="dense_32")(x)
    x = Dropout(0.5, name="dropout")(x)
    outputs = Dense(3, activation="softmax", name="output_softmax")(x)

    model = Model(inputs, outputs, name="StockTrend_LSTM")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    print("5. Training model (10 epochs)...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    print("6. Evaluating on Test Set...")
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test, y_pred,
        target_names=["downtrend", "flat", "uptrend"],
        output_dict=True
    )
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    print(f"Test Accuracy: {acc:.4f}")

    # Save model
    model_path = os.path.join("model_artifacts", "lstm_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save evaluation metrics and history
    metrics_data = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "labels": ["downtrend", "flat", "uptrend"],
        "history": {
            "epochs": list(range(1, 11)),
            "train_accuracy": [float(v) for v in history.history.get("accuracy", [])],
            "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
            "train_loss": [float(v) for v in history.history.get("loss", [])],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])]
        },
        "model_summary": [
            {"layer": "Input Layer", "shape": f"(None, {max_len}, {embedding_dim})", "params": 0},
            {"layer": "LSTM (64 units, return_sequences=True)", "shape": f"(None, {max_len}, 64)", "params": 29440},
            {"layer": "Flatten", "shape": "(None, 1920)", "params": 0},
            {"layer": "Dense (32 units, ReLU)", "shape": "(None, 32)", "params": 61472},
            {"layer": "Dropout (0.5 rate)", "shape": "(None, 32)", "params": 0},
            {"layer": "Dense Output (3 units, Softmax)", "shape": "(None, 3)", "params": 99}
        ]
    }

    with open(os.path.join("model_artifacts", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Save dataset summary
    dataset_summary = {
        "total_raw_rows": len(raw_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_distribution": train_df["Trend"].value_counts().to_dict(),
        "val_distribution": val_df["Trend"].value_counts().to_dict(),
        "test_distribution": test_df["Trend"].value_counts().to_dict(),
        "top_companies": raw_df["Name"].value_counts().head(10).to_dict() if "Name" in raw_df.columns else {}
    }
    with open(os.path.join("model_artifacts", "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    print("All artifacts generated successfully!")

if __name__ == "__main__":
    main()

