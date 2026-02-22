"""YAML config generation from STAC metadata for ZenML pipeline runs."""

from __future__ import annotations

from typing import Any

import pystac

from fair_models.stac.constants import OCI_MEDIA_TYPE


def _extract_input_spec(mlm_input: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull chip_size from mlm:input shape. Band names are model-internal, not pipeline params."""
    if not mlm_input:
        return {}
    shape = mlm_input[0].get("input", {}).get("shape", [])
    return {"chip_size": shape[-1]} if len(shape) == 4 else {}


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

    runtime = base_model_item.assets.get("training-runtime")
    if runtime and runtime.media_type == OCI_MEDIA_TYPE:
        config["settings"] = {"docker": {"parent_image": runtime.href}}

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

    config: dict[str, Any] = {
        "parameters": parameters,
    }

    runtime = model_item.assets.get("inference-runtime")
    if runtime and runtime.media_type == OCI_MEDIA_TYPE:
        config["settings"] = {"docker": {"parent_image": runtime.href}}

    return config
