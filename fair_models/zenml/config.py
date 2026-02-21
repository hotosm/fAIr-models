"""YAML config generation from STAC metadata for ZenML pipeline runs."""

from __future__ import annotations

from typing import Any

import pystac


def _extract_input_spec(mlm_input: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull chip_size, num_bands, and band names from mlm:input."""
    if not mlm_input:
        return {}
    first = mlm_input[0]
    shape = first.get("input", {}).get("shape", [])
    bands = [b["name"] for b in first.get("bands", [])]
    spec: dict[str, Any] = {"bands": bands}
    if len(shape) == 4:
        spec["chip_size"] = shape[-1]
    return spec


def _extract_num_classes(mlm_output: list[dict[str, Any]]) -> int | None:
    """Derive num_classes from the first output's classification:classes list."""
    if not mlm_output:
        return None
    classes = mlm_output[0].get("classification:classes", [])
    return len(classes) if classes else None


def generate_training_config(
    base_model_item: pystac.Item,
    dataset_item: pystac.Item,
    model_name: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate ZenML pipeline run config dict from STAC metadata.

    ZenML YAML config schema ref:
    https://docs.zenml.io/concepts/steps_and_pipelines/yaml_configuration
    """
    props = base_model_item.properties

    hyperparams: dict[str, Any] = dict(props.get("mlm:hyperparameters", {}))
    input_spec = _extract_input_spec(props.get("mlm:input", []))
    num_classes = _extract_num_classes(props.get("mlm:output", []))

    ds_props = dataset_item.assets
    chips_href = ds_props["chips"].href
    labels_href = ds_props["labels"].href

    parameters: dict[str, Any] = {
        "base_model_weights": base_model_item.assets["model"].href,
        "dataset_chips": chips_href,
        "dataset_labels": labels_href,
        **hyperparams,
        **input_spec,
    }
    if num_classes is not None:
        parameters["num_classes"] = num_classes

    if overrides:
        parameters.update(overrides)

    config: dict[str, Any] = {
        "model": {"name": model_name},
        "parameters": parameters,
    }

    if "training-runtime" in base_model_item.assets:
        config["settings"] = {"docker": {"parent_image": base_model_item.assets["training-runtime"].href}}

    return config


def generate_inference_config(
    model_item: pystac.Item,
    input_images_path: str,
) -> dict[str, Any]:
    """Generate ZenML inference pipeline run config dict from a STAC model item.

    Works for both base-model and local-model items since both carry the
    same MLM asset/property structure.
    """
    props = model_item.properties
    input_spec = _extract_input_spec(props.get("mlm:input", []))

    parameters: dict[str, Any] = {
        "model_weights": model_item.assets["model"].href,
        "input_images": input_images_path,
        **input_spec,
    }

    num_classes = _extract_num_classes(props.get("mlm:output", []))
    if num_classes is not None:
        parameters["num_classes"] = num_classes

    mlm_output = props.get("mlm:output", [])
    if mlm_output:
        post_fn = mlm_output[0].get("post_processing_function", {})
        if post_fn.get("expression"):
            parameters["post_processing"] = post_fn["expression"]

    config: dict[str, Any] = {
        "parameters": parameters,
    }

    if "inference-runtime" in model_item.assets:
        config["settings"] = {"docker": {"parent_image": model_item.assets["inference-runtime"].href}}

    return config
