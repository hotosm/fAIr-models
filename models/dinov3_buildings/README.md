# dinov3-buildings

Binary building-footprint segmentation for very high resolution RGB aerial imagery, built on a frozen DINOv3-ViT-L/16 encoder with a UperNet decoder. Output is GeoJSON polygons in EPSG:4326, one per detected building, with a per-polygon confidence score.

## Summary

| | |
|---|---|
| Task | Building footprint extraction |
| Input | 3-band RGB GeoTIFF chips, VHR (~30 cm GSD) |
| Output | GeoJSON `FeatureCollection` of `Polygon` features in EPSG:4326, each with `class` and `score` |
| Coverage | Global; trained on the HOT VHR Building Segmentation dataset (~58k chips worldwide) |
| Use cases | Humanitarian mapping, OSM data preparation, disaster-response building inventory |
| License | Apache-2.0 |

## Intended use

This model is for direct inference on OpenAerialMap tiles, and for optional decoder fine-tuning on small downstream labelled sets (~100-200 chips) when an area shows imagery characteristics that differ from the global training distribution.

A typical workflow:

1. Bring a TMS URL plus a bounding box, or a directory of georeferenced RGB chips.
2. Send a request to the inference container, or invoke the inference pipeline programmatically.
3. Receive building polygons with per-polygon confidence scores, ready to merge into OSM or another vector store.

Optional second step: fine-tune the decoder on a small labelled subset of your own area (the encoder stays frozen). This adapts the model to local imagery characteristics while preserving the global feature representation.

## Where it works best

- Very high resolution aerial imagery, around 30 cm GSD, sourced from OpenAerialMap.
- Dense and rural settlement patterns from any continent represented in the HOT VHR training set.
- 3-band RGB input at the native 256x256 tile size matching DINOv3-L's operating range.
- Buildings with reasonably orthogonal footprints; the DP+MBR-safe vectorisation step squares up corners that the raw mask would otherwise leave jagged.

## Where to use caution

- Its a geo foundation model requires significant resources if you wanna finetune , use with caution in constrained infrastructure
- Regions visually distinct from the HOT global mix (extreme arctic settlements, irregular vernacular architecture, dense informal settlements with overlapping roofs) typically benefit from a per-area decoder finetune.
- The regularised output averages 5-7 vertices per polygon. Buildings with fine architectural detail (courtyards, balconies, complex roofs) appear as smoothed footprints by design. Careful with rounded buildings this might not work 
- Multispectral, SAR, DEM, or grayscale inputs sit outside the design envelope of this model; bring 3-band RGB.

## How to use

### Live inference

A long-running container exposes `POST /predict` on port 8080. Request body:

```json
{
  "model_uri":  "https://huggingface.co/kshitijrajsharma/dinov3-hot-buildings/resolve/main/dinov3_buildings.onnx",
  "image_uri":  "https://tiles.openaerialmap.org/.../{z}/{x}/{y}",
  "bbox":       [west, south, east, north],
  "zoom":       18,
  "params":     {"confidence_threshold": 0.5}
}
```

The server fetches the tiles for the requested bbox, runs sliding-window inference, and returns a GeoJSON `FeatureCollection`. Each feature has `properties.class = 1` and `properties.score` in `[0, 1]`.

### Fine-tuning on a local area

Drop a directory of RGB OAM chips and a single `labels.geojson` of OSM building polygons into the platform, then trigger `training_pipeline`. Default budget is 15 epochs at batch size 8 with `lr=5e-5`. The pipeline produces a fine-tuned ONNX and a metrics report.

### Inference parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `confidence_threshold` | 0.5 | Mask probability above which a pixel counts as foreground during watershed seeding |
| `simplify_m` | 1.0 | Douglas-Peucker simplification tolerance in metres (EPSG:3857) |
| `regularize_area_threshold` | 0.55 | Minimum polygon-area / MBR-area ratio for the rectangle-substitution step |
| `regularize_overlap_tol_m2` | 1.0 | Maximum new neighbour overlap allowed when substituting a polygon with its MBR |
| `min_area_m2` | 1.0 | Polygons smaller than this area are dropped |

## Inputs and outputs

**Input contract.** A directory of georeferenced RGB GeoTIFFs (`.tif` / `.tiff` / `.png` with `.aux.xml` sidecars). The platform's tile downloader, `geomltoolkits.downloader.tms`, produces this layout automatically for any TMS URL plus a bounding box.

**Output contract.** A GeoJSON `FeatureCollection` in EPSG:4326. Each feature:

```json
{
  "type": "Feature",
  "properties": {"class": 1, "score": 0.84},
  "geometry":   {"type": "Polygon", "coordinates": [...]}
}
```

The `score` is the mean predicted mask probability across the pixels belonging to that polygon's source watershed instance. Higher = the model assigned more pixels with high foreground probability to that building.

## Compute footprint

Measured on an Intel Core i9-14900HX (32 threads, CPU-only inference) for the live serve image.

| | Value |
|---|---:|
| Total parameters | ~343M (300M frozen encoder + 43M trainable decoder, neck, heads) |
| Trainable parameters | ~43M |
| ONNX file size | 1.42 GB |
| ONNX session load time | ~4 s |
| Per-chip inference (one 256x256 forward) | 1.1 s median (best 0.72 s, worst 1.57 s) |
| Banepa AOI end-to-end POST `/predict`, cold cache | 228 s (includes 1.4 GB ONNX download) |
| Banepa AOI end-to-end POST `/predict`, warm cache | 196 s |
| Peak container RAM during inference | ~4.9 GB |
| Working CRS for all geometric math | EPSG:3857 |

The Banepa AOI here covers `[85.5162, 27.6312, 85.5244, 27.6385]` at zoom 18, producing 2035 building polygons over ~36 input tiles merged into a ~1500x1500 raster (121 sliding windows at 256/128).

## Architecture

- **Encoder**: DINOv3-ViT-L/16 (LVD1689M, ~300M params), frozen during training and inference. Intermediate features tap blocks [5, 11, 17, 23] via `get_intermediate_layers(norm=True, reshape=True, return_class_token=False)`, applying the backbone's final LayerNorm to every tap.
- **Neck**: terratorch `LearnedInterpolateToPyramidal`, a ConvTranspose-based learned upsampling at scales [4x, 2x, 1x, 0.5x], producing a 4-level feature pyramid.
- **Decoder**: terratorch `UperNetDecoder` with `decoder_channels=512`, `pool_scales=(1, 2, 3, 6)`, BN+ReLU lateral and FPN convs.
- **Heads**:
  - **Main**: 1x1 Conv -> 3 channels (mask logit, boundary logit, signed distance via tanh).
  - **Auxiliary**: FCN head on tap-2 (3x3 Conv -> BN -> ReLU -> Dropout(0.1) -> 1x1 Conv) -> 3 channels. Used during training; dropped at ONNX export.

The native input size is 256x256, matching the HOT VHR Building Segmentation tile size and DINOv3-L's expected operating range. fAIr-standard 256x256 chips feed the model directly without resizing.

## Training data and recipe

- **Source**: [hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation) (~58k train, 7.2k val, 7.2k test, 256x256 chips).
- **Normalisation**: dataset-computed mean `[0.4297, 0.4002, 0.3433]` and std `[0.2056, 0.1674, 0.1599]` from `norm_stats.json` in the HF dataset repo.
- **Hard-negative handling**: null-mask tiles (no buildings in the chip) are kept in training as hard negatives, suppressing false positives over rural land. Tiles with fully-null RGB are dropped during data preparation.
- **Loss**: BCE+Dice on the mask logit, BCE on a 2-pixel boundary band, Huber on a signed distance map clipped to ±15 px and normalised to [-1, 1]. The auxiliary head receives the same multi-task loss at weight 0.4. Loss weights were chosen by Optuna TPE over 12 trials on 10% data x 8 epochs (`conf/experiments/v5_hpo_best.yaml` in the upstream repo).
- **Optimiser**: AdamW with OneCycleLR cosine schedule, max_epochs=50, early-stop patience=5 on val/loss. Best checkpoint at epoch 5.

## Train/val split (for the local finetune step)

Spatial-block split, not random. Chip filenames parse as `OAM-{x}-{y}-{z}.tif` to extract TMS tile coordinates; chips are bucketed into `(x // block_size, y // block_size)` blocks and whole blocks are assigned to train or val until `val_ratio` is reached. The default `block_size=4` keeps adjacent 16-chip neighbourhoods together, so the catalog-facing val set never shares spatial context with train. Filenames that fall outside the OAM pattern get a seeded random split.

Two val sets coexist by design:

- **Spatial val** (`split_info.val_chip_names`): the held-out set `evaluate_model` measures. Reported as `fair:pixel_iou` and the instance metrics on the local-model STAC item.
- **Inner val** (`hyperparameters["inner_val_frac"]`, default 0.1): a small random slice inside the spatial-train set that upstream `dinov3_hot.finetune.finetune` uses for Lightning's per-epoch `val_loss` / `val_iou` and early stopping. Upstream sees a chips directory containing only spatial-train chips (symlinked), so the inner val never overlaps the spatial val.

## Evaluation

Numbers below come from two independent code paths to cross-validate the model.

### HF VHR test split (7236 global tiles, raster-level instance matching from `dinov3-hot-ablation`)

| Pixel IoU | Precision | Recall | F1@0.5 | avg vertices | orthogonality |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **0.441** | 0.212 | 0.345 | **0.262** | 4.91 | 0.71 |

GT reference (HF mask polygons): avg vertices 4.30, orthogonality 0.91.

### Banepa, Nepal (1536x1536 OAM scene, 2720 OSM GT polygons)

Two implementations agree to within ~0.02 across every shared metric. The upstream raster-level numbers come from `dinov3_hot.metrics.instance_prf`; the polygon-level numbers come from [polymetrics](https://github.com/kshitijrajsharma/polymetrics)

| Metric | Upstream (raster) | polymetrics (polygon) |
| --- | ---: | ---: |
| Precision | 0.483 | **0.495** |
| Recall | 0.351 | **0.370** |
| F1@0.5 | **0.407** | **0.424** |
| Mean IoU (TP) | n/a | 0.676 |
| mAP@0.5 | n/a | **0.219** |
| mAP@0.5:0.95 | n/a | **0.071** |
| avg vertices | 5.99 | 6.18 |
| orthogonality | 0.63 | 0.61 |

GT reference (OSM polygons): avg vertices 5.30, orthogonality 0.95.

## Fine-tuning details

The `train_model` ZenML step expects a directory of OAM RGB chips plus a single GeoJSON of OSM building polygons. Labels are burned to per-chip rasters via `geomltoolkits.raster.burn.burn_labels`, then the decoder is fine-tuned with the encoder kept frozen. fAIr's local finetune supervises the mask channel only (BCE+Dice); the boundary and distance heads stay at their pretrained values.

Default fine-tuning budget exposed by the pipeline:

- 100-200 chips, `epochs=15`, `batch_size=8`, `learning_rate=5e-5`, `weight_decay=0.01`.
- Observed val/iou lift on the Banepa sample: 0 to 5 percentage points. The HF pretraining already covers dense-urban data well, so the per-area finetune delta is often small. Areas visually distinct from the HOT global mix show the largest lift.

The full multi-task fine-tune recipe (boundary + distance + auxiliary head supervision) lives in the upstream `dinov3-hot finetune` CLI.

## License

Apache-2.0. The DINOv3-ViT-L/16 encoder weights come from Facebook Research's DINOv3 (Oquab et al., 2025), Apache-2.0. The decoder architecture, multi-task training recipe, post-processing pipeline (DP+MBR-safe vectorisation, watershed instance separation), and per-area finetune scaffolding come from [`dinov3-hot-ablation`](https://github.com/kshitijrajsharma/dinov3-hot-ablation) by Kshitij Raj Sharma, Apache-2.0.

## Citation

- Encoder: [DINOv3](https://arxiv.org/abs/2508.10104) (Oquab et al., 2025).
- Decoder + training + post-processing recipe: [`dinov3-hot-ablation`](https://github.com/kshitijrajsharma/dinov3-hot-ablation) by Kshitij Raj Sharma.
- Independent metric reproduction: [polymetrics](https://github.com/kshitijrajsharma/polymetrics).
