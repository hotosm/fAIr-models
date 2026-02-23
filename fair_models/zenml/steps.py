"""Reusable ZenML steps for model developers.

load_model resolves a model from the ZenML artifact store using
either a direct artifact version ID or a URI fallback. The materializer
registered at training time handles deserialization — PyTorch, Keras,
TensorFlow, or any custom materializer works transparently.
"""

from typing import Any

from zenml import step
from zenml.client import Client


@step
def load_model(
    model_uri: str,
    zenml_artifact_version_id: str = "",
) -> Any:
    """Resolve model from ZenML artifact store. Framework-agnostic via materializer."""
    client = Client()
    if zenml_artifact_version_id:
        art = client.get_artifact_version(zenml_artifact_version_id)
    else:
        results = client.list_artifact_versions(uri=model_uri)
        if not results:
            msg = f"No artifact found for URI: {model_uri}"
            raise RuntimeError(msg)
        art = results[0]
    return art.load()
