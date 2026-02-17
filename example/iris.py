from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from zenml import pipeline, step


@step
def load_data() -> pd.DataFrame:
    """Load the dataset."""
    # For this example, we'll use the iris dataset
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df


@step
def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Split and preprocess the data."""
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


@step
def train_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train the model."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


@step
def evaluate_model(
    model: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series
) -> float:
    """Evaluate the model and return the accuracy."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.2f}")
    return accuracy


@step
def save_checkpoint(
    model: RandomForestClassifier, checkpoint_path: str = "./model_checkpoint.pkl"
) -> str:
    """Save the trained model to disk."""
    import pickle

    with open(checkpoint_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model checkpoint saved to {checkpoint_path}")
    return checkpoint_path


@step
def summarize_data(df: pd.DataFrame) -> tuple[int, int]:
    """Log dataset shape inside a step to avoid StepArtifact access."""
    shape = df.shape
    print(f"Dataset shape: {shape}")
    return shape


@pipeline(enable_cache=False, name="rf_iris_classifier_training_pipeline")
def training_pipeline():
    """Define the pipeline steps."""
    df = load_data()
    summarize_data(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    save_checkpoint(model)


@step
def load_checkpoint(
    checkpoint_path: str = "./model_checkpoint.pkl",
) -> RandomForestClassifier:
    """Load a trained model from checkpoint."""
    import pickle

    with open(checkpoint_path, "rb") as f:
        model = pickle.load(f)
    return model


@step
def inference(model: RandomForestClassifier, X_data: np.ndarray) -> np.ndarray:
    """Run inference on input data."""
    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    return predictions


@pipeline(enable_cache=False, name="rf_iris_classifier_inference_pipeline")
def inference_pipeline():
    """Define the inference pipeline steps."""
    df = load_data()
    _, X_test, _, _ = preprocess_data(df)
    model = load_checkpoint()
    predictions = inference(model, X_test)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["train", "inference"], nargs="?", default="train"
    )
    parser.add_argument("--config", default="iris.yaml")
    args = parser.parse_args()
    if args.mode == "inference":
        inference_pipeline.with_options(config_path=args.config)()
    else:
        training_pipeline.with_options(config_path=args.config)()
