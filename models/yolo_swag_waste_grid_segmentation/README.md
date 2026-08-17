# YOLO Solid Waste Grid Segmentation (SWAG)

## Overview

SWAG screens , georeferenced, very-high-resolution RGB aerial or UAV imagery for visible, openly dumped solid waste. It classifies 5 m × 5 m cells as `waste` or `background` and returns the cells as GeoJSON for review. The pretrained model covers urban and peri-urban settings from 60 globally distributed OpenAerialMap scenes; fine-tune it before use in a geography or imagery setting that is not well represented by that data.

## Pretrained source

The pretrained weights, labels, and preparation utilities come from the [YOLO Solid Waste Assessment on Grids source repository](https://github.com/GIScience/solid-waste-detection-for-fAIr). Its training data comprises 60 OpenAerialMap scenes; the source repository is released under the MIT License. The model is based on the accompanying [SWAG preprint](https://doi.org/10.48550/arXiv.2605.02316). Review the licence of each source image before reusing training imagery.

## Training coverage and examples

The pretrained SWAG dataset covers geographically diverse OpenAerialMap scenes. The map below shows the source training-scene distribution.

[![World map of SWAG training-scene locations](https://raw.githubusercontent.com/GIScience/solid-waste-detection-for-fAIr/dc826f335b5403ed8161f0b6f5656e89aae9b2b8/data/images/Overview.png)](https://github.com/GIScience/solid-waste-detection-for-fAIr/blob/dc826f335b5403ed8161f0b6f5656e89aae9b2b8/data/images/Overview.png)

The following examples show 5 m grid-cell waste overlays in different settings. Select an image to view it at full size.

| Riverbank setting | Open-site setting | Built-up setting |
| --- | --- | --- |
| [![Waste grid cells along a riverbank](https://raw.githubusercontent.com/GIScience/solid-waste-detection-for-fAIr/dc826f335b5403ed8161f0b6f5656e89aae9b2b8/data/images/example_senegal.png)](https://github.com/GIScience/solid-waste-detection-for-fAIr/blob/dc826f335b5403ed8161f0b6f5656e89aae9b2b8/data/images/example_senegal.png) | [![Waste grid cells at an open site](https://raw.githubusercontent.com/GIScience/solid-waste-detection-for-fAIr/dc826f335b5403ed8161f0b6f5656e89aae9b2b8/data/images/example.png)](https://github.com/GIScience/solid-waste-detection-for-fAIr/blob/dc826f335b5403ed8161f0b6f5656e89aae9b2b8/data/images/example.png) | [![Waste grid cells in a built-up area](https://raw.githubusercontent.com/GIScience/solid-waste-detection-for-fAIr/dc826f335b5403ed8161f0b6f5656e89aae9b2b8/data/images/example_buildings.png)](https://github.com/GIScience/solid-waste-detection-for-fAIr/blob/dc826f335b5403ed8161f0b6f5656e89aae9b2b8/data/images/example_buildings.png) |

Images are from the [SWAG source repository](https://github.com/GIScience/solid-waste-detection-for-fAIr) and are pinned to its source commit for reproducibility.

## Architecture

SWAG is a two-class YOLO26x classifier. The published ONNX model accepts one RGB grid cell as a `1 × 3 × 128 × 128` `float32` tensor and returns `1 × 2` class scores for `background` and `waste`. The pipeline creates the spatial grid before classification and restores the grid-cell geometry in the GeoJSON output.

Fine-tuning starts with a mosaic of the supplied imagery and a GeoJSON or GeoPackage label file. Features with `label = 1` mark waste and features with `label = 0` mark reviewed background; both are required. A cell is labelled `waste` when at least 80% of its area is covered by a waste label. All cells from one source label polygon stay together in train, validation, or test.


### Labeling guidance

- **Define the positive class consistently.** Label only exposed, openly dumped waste that forms a clearly defined, densely covered area: visible waste should cover about 80% or more of the area being labelled. Do not label sparse, ambiguous, or mostly occluded material as a waste pile.
- **Prioritise variety and spatial coverage.** Select waste piles with varied appearance, material, size, and setting, and distribute them as evenly as practical across the training area. Train, validation, and test are assigned by source polygon, so provide enough separate waste and background polygons to populate all three splits.
- **Handle very large piles selectively.** You may label smaller, representative parts of a very large pile instead of its full extent. The pipeline converts the selected geometry into 5 m × 5 m training cells and keeps all cells from each selected polygon in one split.
- **Adapt the overlap threshold for very small piles.** The default `waste_overlap_threshold` is `0.8`, so a waste polygon must cover at least 80% of a 5 m grid cell for that cell to become a positive training example. If the imagery contains many genuine but very small piles, lowering this hyperparameter can retain more useful positive cells. Treat this as a deliberate local-data choice: a lower threshold also makes each positive cell less visually pure.
- **Add reviewed background polygons.** `label = 0` background polygons are required and are split as complete source polygons, just like waste. Include representative hard negatives—visually noisy vegetation, rubble, beaches, mosaic or patterned ground surfaces, bright water-reflection artefacts, building materials, and clothes laid out to dry—along with ordinary clean background. Do not mass-label unreviewed cells.

The [extra-large checkpoint](https://raw.githubusercontent.com/GIScience/solid-waste-detection-for-fAIr/9f62fd1e4de6905a38620c195a6e62bcef280956/data/checkpoint/checkpoint_v1_extra_large.pt) is the pinned base model used by this pipeline.

### Model size

- **Parameters:** 28.33 M
- **Compute:** 2.21 GMACs (about 4.41 GFLOPs, counting one multiply-add as two FLOPs) per cell
- **Inference input:** fixed batch-1 `float32` tensor of `1 × 3 × 128 × 128`; inference runs one 5 m cell at a time
- **CPU inference:** about 11.7 ms per cell
- **Memory and benchmark host:** about ~ 381 MiB  memory usage after model warm-up; Intel Core Ultra 7 255H (16 logical CPUs), CPU ONNX Runtime

## Intended use

**Target:** visible, openly dumped solid waste in georeferenced, very-high-resolution RGB aerial or UAV imagery. SWAG is intended to screen an urban or peri-urban area and show eg. where a reviewer should look more closely. Fine-tune it with local labels when the imagery or the appearance of waste differs from the training data.

Each output feature is one 5 m × 5 m grid cell by default (`cell_size_m` can change this size):

| Output part | Meaning |
| --- | --- |
| Geometry | The geographic extent of the grid cell in WGS84—not the boundary of a waste pile. |
| `cell_id` | The cell's identifier within this inference run. |
| `label` | `waste` when the waste confidence meets the selected threshold; otherwise `background`. A `waste` label means the cell is a candidate for review, not that all of the cell is waste. |
| `confidence` | The model's estimated probability, from 0 to 1, that the cell contains the target waste class. |

SWAG does not identify waste material types, trace exact pile boundaries, or estimate waste mass or volume.

## Limitations

- Results depend on image quality, ground sampling distance, illumination, occlusion, and whether local waste appearance resembles the OAM training scenes. Local fine-tuning is recommended before operational use in a new geography or sensor setting.
- The fixed 5 m grid trades boundary detail for consistent area coverage. A positive cell can include both waste and non-waste land, and small piles can be missed when they do not cover the configured threshold of a cell.
- Tile zoom is not a fixed ground resolution: metres per pixel vary with latitude, and an OAM service can resample imagery beyond its native resolution. The serving runtime currently accepts its global zoom range without enforcing this model's STAC zoom metadata. Check the source imagery's native ground sampling distance and the number of native pixels per 5 m cell before relying on a prediction.
- Grouping cells by source polygon reduces leakage between splits, but it is not an independent field campaign. Reported accuracy should not be interpreted as a guarantee of performance in another area.
- Predictions should be reviewed by a domain expert before publication, enforcement, or resource-allocation decisions.

## Usage

### Zoom Level

As we deal with accumulations of smaller Objects, it is strongly encouraged to use the finest available zoom level for your imagery and stick to the recommended zoom level of 22. 

### Training parameters

| Parameter                 | Default | Use                                                                                                                                                                                 |
|---------------------------|--------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `epochs`                  |     `5` | Number of fine-tuning passes through the training data.                                                                                                                             |
| `batch_size`              |     `4` | Number of cells per training update; increase it only when GPU memory permits. It does not affect inference.                                                                        |
| `learning_rate`           | `0.001` | Optimizer step size; reduce it when fine-tuning becomes unstable.                                                                                                                   |
| `freeze_layers`           |  `10` | Number of first YOLO layers to freeze. Use `0` to freeze none; a larger value freezes more pretrained layers, which can help when fine-tuning data are limited. |
| `val_ratio`               |   `0.1` | Target fraction of source label-polygon groups reserved for validation. |
| `test_ratio`              |   `0.1` | Target fraction of source label-polygon groups reserved for final evaluation. |
| `waste_overlap_threshold` |   `0.8` | Minimum waste-label coverage required for a 5 m cell to become a positive training example. Usefull to lower if alot of smaller waste piles have been labeled                       |
| `cell_size_m`             |   `5.0` | Training-grid edge length in metres; keep it at `5.0` unless deliberately retraining for another spatial scale.                                                                     |

`optimizer` (`AdamW`), `scheduler` (`cosine`), `weight_decay` (`0.0001`), and `sample_fraction` (`1.0`) are also configurable; their defaults are appropriate for the supplied fine-tuning workflow.

### Inference parameters

| Parameter | Default | Use |
| --- | ---: | --- |
| `confidence_threshold` | `0.5` | Required threshold for assigning `waste`: raise it to reduce false positives, or lower it to find more candidate cells. |
| `cell_size_m` | `5.0` | Inference-grid edge length in metres. Smaller cells give a denser output grid and require more model calls; use the training value unless there is a clear reason to change it. |

Inference is fixed at batch 1; there is no inference `batch_size` setting.

## Citation

If you use this model or its pretrained weights, cite [*Open-access model for detecting openly dumped dispersed municipal solid waste from crowdsourced UAV imagery in Sub-Saharan Africa*](https://doi.org/10.48550/arXiv.2605.02316).

In addtion a citation of the global sourced training data [*Global YOLO SWAG *](https://zenodo.org/records/21874456) is welcome. 
## License

SWAG is released under the [MIT License](https://spdx.org/licenses/MIT.html). The accompanying data-preparation source repository is also MIT-licensed by the GIScience Research Group and HeiGIT.
