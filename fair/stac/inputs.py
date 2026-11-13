"""How a model's `mlm:input` entries name its imagery: input <name> reads the dataset asset `chips-<name>`,
and predictions give imagery keyed by the same names."""

import re
from collections.abc import Iterable, Mapping

import pystac

INPUT_CHIPS_ASSET_PREFIX = "chips-"
# A dataset whose chips are one `chips` asset gives them to the `image` input of a single-input model.
SINGLE_CHIPS_ASSET = "chips"
IMAGE_INPUT = "image"
# Input names become asset keys and storage paths, so they are restricted to slugs.
INPUT_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


def input_names(model_item: pystac.Item) -> list[str]:
    return [entry["name"] for entry in model_item.properties.get("mlm:input", [])]


def require_imagery_for_inputs(model_item: pystac.Item, imagery_names: Iterable[str]) -> list[str]:
    """The model's input names; raises ValueError unless they are slugs and the imagery names are exactly those."""
    names = input_names(model_item)
    if invalid := [name for name in names if not INPUT_NAME_PATTERN.fullmatch(name)]:
        raise ValueError(f"Model '{model_item.id}' has input names that are not slugs: {invalid}")
    if sorted(imagery_names) != sorted(names):
        raise ValueError(f"Model '{model_item.id}' reads imagery for inputs {names}, got {sorted(imagery_names)}")
    return names


def input_chips_asset(input_name: str) -> str:
    return f"{INPUT_CHIPS_ASSET_PREFIX}{input_name}"


def input_imagery_title(input_name: str) -> str:
    return f"Imagery for input {input_name}"


def chips_hrefs_by_input(asset_hrefs: Mapping[str, str]) -> dict[str, str]:
    """Chips href of each input among a dataset's asset hrefs; a lone `chips` asset is the `image` input."""
    hrefs = {
        key.removeprefix(INPUT_CHIPS_ASSET_PREFIX): href
        for key, href in asset_hrefs.items()
        if key.startswith(INPUT_CHIPS_ASSET_PREFIX)
    }
    if not hrefs and SINGLE_CHIPS_ASSET in asset_hrefs:
        return {IMAGE_INPUT: asset_hrefs[SINGLE_CHIPS_ASSET]}
    return hrefs


def dataset_chips_assets(model_item: pystac.Item, dataset_item: pystac.Item) -> dict[str, str]:
    """Dataset asset key of the chips for each model input that the dataset has chips for."""
    names = input_names(model_item)
    found = {name: input_chips_asset(name) for name in names if input_chips_asset(name) in dataset_item.assets}
    if not found and names == [IMAGE_INPUT] and SINGLE_CHIPS_ASSET in dataset_item.assets:
        return {names[0]: SINGLE_CHIPS_ASSET}
    return found
