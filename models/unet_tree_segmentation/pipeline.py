"""unet-tree-segmentation: a prediction-only U-Net that segments tree crowns in RGB imagery."""

from pathlib import Path
from typing import Annotated, Any

from zenml import pipeline, step

from fair.stac.inputs import IMAGE_INPUT
from fair.utils.data import resolve_directory

CHIP_SIZE = 256
TREE_CHANNEL = 1


def _chip_paths(chips_href: str) -> list[Path]:
    chips = resolve_directory(chips_href)
    return [chips] if chips.is_file() else sorted(chips.rglob("*.tif"))


def preprocess(image_path: Any) -> Any:
    """Read a 256 x 256 RGB chip as a (1, 3, 256, 256) float32 tensor scaled to 0 to 1."""
    import numpy as np
    import rasterio

    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
    if rgb.shape[1:] != (CHIP_SIZE, CHIP_SIZE):
        msg = f"{image_path}: chip is {rgb.shape[2]}x{rgb.shape[1]}, the model takes {CHIP_SIZE}x{CHIP_SIZE}"
        raise ValueError(msg)
    return rgb[np.newaxis]


def postprocess(logits: Any, threshold: float) -> Any:
    """Turn (2, H, W) background and tree logits into a uint8 tree mask."""
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    # Softmax over two classes is a sigmoid of the logit difference.
    tree_probability = 1.0 / (1.0 + np.exp(logits[1 - TREE_CHANNEL] - logits[TREE_CHANNEL]))
    return (tree_probability >= threshold).astype(np.uint8)


def predict(session: Any, input_images: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    import rasterio
    from pyproj import Transformer
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform

    threshold = float(params.get("confidence_threshold", 0.5))
    input_name = session.get_inputs()[0].name
    chip_paths = _chip_paths(input_images[IMAGE_INPUT])
    if not chip_paths:
        msg = f"No georeferenced (.tif) chips found in {input_images[IMAGE_INPUT]}"
        raise FileNotFoundError(msg)

    features: list[dict[str, Any]] = []
    for chip_path in chip_paths:
        with rasterio.open(chip_path) as src:
            transform = src.transform
            to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        logits = session.run(None, {input_name: preprocess(chip_path)})[0][0]
        mask = postprocess(logits, threshold)
        for geom, _ in shapes(mask, mask=mask.astype(bool), transform=transform):
            polygon = shapely_transform(lambda x, y, _z=None, t=to_wgs84: t.transform(x, y), shape(geom))
            features.append(
                {"type": "Feature", "properties": {"class": 1, "label": "tree"}, "geometry": mapping(polygon)}
            )
    return {"type": "FeatureCollection", "features": features}


@step
def run_inference(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    return predict(load_session(model_uri), input_images, inference_params)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: dict[str, str],
    inference_params: dict[str, Any],
) -> None:
    run_inference(model_uri=model_uri, input_images=input_images, inference_params=inference_params)
