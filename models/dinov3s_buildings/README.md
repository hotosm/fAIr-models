# dinov3s-buildings

Binary building-footprint segmentation for very high resolution RGB aerial imagery, built on a frozen DINOv3-**ViT-S/16** encoder with a UperNet decoder. Output is GeoJSON polygons in EPSG:4326, one per detected building, with a per-polygon confidence score. Approximately **6.6x smaller** than the [dinov3l-buildings](../dinov3l_buildings) ViT-L variant, trading ~10 pp Banepa instance F1 for size and inference cost.

## Summary

| | |
|---|---|
| Task | Building footprint extraction |
| Input | 3-band RGB GeoTIFF chips, VHR (~30 cm GSD) |
| Output | GeoJSON `FeatureCollection` of `Polygon` features in EPSG:4326, each with `class` and `score` |
| Coverage | Global; trained on the HOT VHR Building Segmentation dataset (~37k chips worldwide) |
| Use cases | Edge / constrained-infra inference, batch building inventory where size matters more than the last few pp of F1 |
| License | Apache-2.0 |

## When to pick this over `dinov3l-buildings`

- Edge or CPU-only deployment where the larger model's 1.4 GB ONNX is impractical.
- Batch jobs over very large areas where ~3x faster per-tile inference materially shifts the wall clock.
- Acceptable trade: ~10 pp lower Banepa instance F1 vs the ViT-L variant. Pixel quality on matched buildings is the same; this model simply matches fewer of them.

Pick `dinov3l-buildings` if instance F1 is the headline metric and you have GPU or sufficient CPU budget.

## Intended use

Direct inference on OpenAerialMap tiles, and optional decoder fine-tuning on small downstream labelled sets (~100-200 chips) when local imagery differs from the global training distribution.

A typical workflow:

1. Bring a TMS URL plus a bounding box, or a directory of georeferenced RGB chips.
2. Send a request to the inference container, or invoke the inference pipeline programmatically.
3. Receive building polygons with per-polygon confidence scores, ready to merge into OSM or another vector store.

## Where it works best

- VHR sat/aerial imagery, ~30-50 cm GSD, sourced from OpenAerialMap.
- 3-band RGB at the native 256x256 tile size matching DINOv3-S's operating range.
- Reasonably orthogonal building footprints; the DP+MBR-safe vectorisation step squares up corners that the raw mask leaves jagged.

## Where to use caution

- Dense urban with many touching buildings: this model catches ~26% fewer instances than the ViT-L variant at Banepa scale.
- Regions visually distinct from the HOT global mix (extreme arctic settlements, irregular vernacular architecture) benefit from a per-area decoder finetune.
- Inputs outside the 3-band RGB envelope (multispectral, SAR, DEM, grayscale) sit outside the model's design envelope.

## How to use

### Live inference

A long-running container exposes `POST /predict` on port 8080. Request body:

```json
{
  "model_uri":  "https://huggingface.co/kshitijrajsharma/dinov3-hot-buildings/resolve/main/dinov3s_buildings.onnx",
  "image_uri":  "https://tiles.openaerialmap.org/.../{z}/{x}/{y}",
  "bbox":       [west, south, east, north],
  "zoom":       18,
  "params":     {"confidence_threshold": 0.4371}
}
```

The server fetches the tiles for the requested bbox, runs sliding-window inference, and returns a GeoJSON `FeatureCollection`. Each feature has `properties.class = 1` and `properties.score` in `[0, 1]`.

### Fine-tuning on a local area

Drop a directory of RGB OAM chips and a single `labels.geojson` of OSM building polygons into the platform, then trigger `training_pipeline`. Default budget is 15 epochs at batch size 8 with `lr=5e-5`. The pipeline produces a fine-tuned ONNX and a metrics report.

### Inference parameters

Catalog defaults come from a 30-trial Optuna TPE post-process HPO on a 1000-chip project-stratified sample of `hotosm/vhr-building-segmentation` (train+test, 70 projects, max 15 chips/project). Objective is the shape-aware composite scored via [polymetrics](https://github.com/kshitijrajsharma/polymetrics) (see below). They differ from `dinov3l-buildings`' defaults because the smaller backbone produces slightly different mask characteristics.

| Parameter | Default | Meaning |
|---|---:|---|
| `confidence_threshold` | 0.4371 | Mask probability above which a pixel counts as foreground during watershed seeding |
| `simplify_m` | 0.9626 | Douglas-Peucker simplification tolerance in metres (EPSG:3857) |
| `regularize_area_threshold` | 0.4949 | Minimum polygon-area / MBR-area ratio for the rectangle-substitution step |
| `regularize_overlap_tol_m2` | 3.9251 | Maximum new neighbour overlap allowed when substituting a polygon with its MBR |
| `min_area_m2` | 2.6465 | Polygons smaller than this area are dropped |
| `seed_min_distance` | 6 | Minimum pixel distance between watershed seeds in the predicted distance map |

Per-area fine-tune re-tunes these via the `tune_postprocess` step on the user's own val set.

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

The `score` is the mean predicted mask probability across the pixels belonging to that polygon's source watershed instance.

## Compute footprint

### Model size

| | DINO Buildings Small (this) | DINO Buildings Large |
|---|---:|---:|
| Total parameters | ~53 M | ~343 M |
| Trainable parameters (decoder + neck + heads) | ~31 M | ~43 M |
| Frozen encoder | ~22 M (ViT-S/16) | ~300 M (ViT-L/16) |
| Lightning ckpt size (state_dict + AdamW moments) | 462 MB | 1.73 GB |
| ONNX file size (self-contained, inference-ready) | 210 MB | 1.42 GB |

### Reference inference benchmark

Standardised CPU-only baseline for capacity planning. Single-threaded ONNX runtime, cold session, synthetic RGB input. Measured on Intel Core i9-14900HX, 64 GB RAM.

**Workload**: One 512x512 RGB tile, sliding window 256 stride 128 = 9 forward passes per tile.

| Metric | Small (this) | Large | Ratio |
|---|---:|---:|---:|
| Cold session load | 0.64 s | 3.31 s | 5.2x |
| Per-window forward (one 256x256) | 1.20 s median | 3.02 s median | 2.5x |
| End-to-end one 512x512 tile | **11.01 s** | 27.54 s | **2.5x** |
| Peak process RAM during inference | 591 MB | 3036 MB | 5.1x |

### Estimating larger AOIs

For an N x N raster (px) at stride 128:
- forwards per tile = ceil((N - 256) / 128 + 1)^2
- total time ≈ session_load + forwards × per_window_time + ~0.5 s vectorise overhead

Examples on this baseline (single-thread):

| Raster | Forwards | Small | Large |
|---|---:|---:|---:|
| 512 x 512 | 9 | ~11 s | ~28 s |
| 1024 x 1024 | 49 | ~59 s | ~2.5 min |
| 2048 x 2048 | 225 | ~4.5 min | ~11 min |
| 4096 x 4096 | 961 | ~19 min | ~48 min |

## Architecture

- **Encoder**: DINOv3-ViT-S/16 (LVD1689M, ~22 M params), frozen during training and inference. Intermediate features tap blocks `[2, 5, 8, 11]` (matched quartile depth for ViT-S's 12 blocks) via `get_intermediate_layers(norm=True, reshape=True, return_class_token=False)`.
- **Neck**: terratorch `LearnedInterpolateToPyramidal`, ConvTranspose-based learned upsampling at scales [4x, 2x, 1x, 0.5x], producing a 4-level feature pyramid. Channels per level auto-adapt to the backbone's embed dim (384 for ViT-S).
- **Decoder**: terratorch `UperNetDecoder` with `decoder_channels=512`, `pool_scales=(1, 2, 3, 6)`, BN+ReLU lateral and FPN convs (identical to the L variant).
- **Heads**:
  - **Main**: 1x1 Conv -> 3 channels (mask logit, boundary logit, signed distance via tanh).
  - **Auxiliary**: FCN head on tap-2 (3x3 Conv -> BN -> ReLU -> Dropout(0.1) -> 1x1 Conv) -> 3 channels. Training only; dropped at ONNX export.

Native input 256x256

## Training data and recipe

- **Source**: [hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation) (~37k train after drop-null, plus val + test).
- **Normalisation**: mean `[0.4297, 0.4002, 0.3433]`, std `[0.2056, 0.1674, 0.1599]` (matches the L variant; same dataset).
- **Loss**: BCE+Dice on the mask logit, BCE on a 2-pixel boundary band, Huber on a signed distance map clipped to ±15 px. Auxiliary head receives the same multi-task loss at weight 0.218. Boundary weight 0.268, distance weight 0.472, TV 0.049. Loss weights chosen by Optuna TPE over 12 trials on 10% data x 8 epochs; TPE rediscovered the L variant's loss-weight regime with LR linearly scaled for the doubled batch.
- **Optimiser**: AdamW, `lr=2.377e-3`, `weight_decay=6.978e-3`, OneCycleLR cosine schedule, max_epochs=50, early-stop patience=15 on val/loss. Best ckpt at epoch 16.
- **Regularisation**: photometric augmentation (`ColorJitter`, `GaussianBlur`, `RandomAdjustSharpness`, `RandomAutocontrast`, `JPEG` quality jitter) plus BCE label smoothing (eps=0.05). Targets robustness to OAM lighting / sensor / compression variance.
- **Batch size**: 64 (doubled from the L variant's 32 thanks to the smaller backbone footprint).

## Evaluation

### HF VHR test split (7236 global tiles, raster-level instance matching)

| Pixel IoU | Precision | Recall | F1@0.5 | avg vertices | orthogonality |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **0.4276** | 0.154 | 0.300 | **0.204** | 4.90 | 0.687 |

GT reference (HF mask polygons): avg vertices 4.30, orthogonality 0.91.

### Banepa, Nepal (1536x1536 OAM scene, 2720 OSM GT polygons, catalog defaults applied)

`tune_postprocess` is the per-area equivalent; numbers here use the catalog defaults shipped above (1000-chip HF VHR HPO).

| Metric | Zero-shot | Per-area fine-tuned (15 ep, lr=1e-4, 102 train / 18 val chips) |
| --- | ---: | ---: |
| Pixel IoU | **0.495** | **0.604** |
| Precision | 0.354 | 0.394 |
| Recall | 0.261 | 0.295 |
| F1@0.5 | **0.300** | **0.337** |
| Mean IoU (matched) | 0.677 | 0.690 |
| avg vertices | 5.28 | 5.85 |
| orthogonality | 0.706 | 0.616 |

GT reference (OSM polygons): avg vertices 5.30, orthogonality 0.95.

The fine-tune lifts pixel-IoU by ~11 pp and instance F1 by ~3.7 pp on this scene by recovering 92 additional true positives. Vertex count and orthogonality both regress slightly: the per-area decoder learns to follow the OAM scene's local building shapes, which here trade rectangularity for boundary fidelity. Re-run `tune_postprocess` after fine-tuning to refit the post-process to the new mask characteristics.

## Train/val split (for the local finetune step)

Identical to `dinov3l-buildings`: spatial-block split on OAM tile coords. Chip filenames parse as `OAM-{x}-{y}-{z}.tif`; chips bucket into `(x // block_size, y // block_size)` blocks, whole blocks assigned to train or val until `val_ratio` is reached. Default `block_size=4`. Filenames outside the OAM pattern fall back to a seeded random split.

## Fine-tuning details

The `train_model` ZenML step expects a directory of OAM RGB chips plus a single GeoJSON of OSM building polygons. Labels are burned to per-chip rasters via `geomltoolkits.raster.burn.burn_labels`, then the decoder is fine-tuned with the encoder kept frozen. fAIr's local finetune supervises the mask channel only (BCE+Dice); the boundary and distance heads stay at their pretrained values.

Default fine-tuning budget exposed by the pipeline: 100-200 chips, `epochs=15`, `batch_size=8`, `learning_rate=5e-5`, `weight_decay=0.01`.

### Per-area post-processing tuning

After training, a small Optuna study (30 trials by default, TPE sampler seeded by `split_seed`) searches the six inference post-processing params on the val set, scored by [polymetrics](https://github.com/kshitijrajsharma/polymetrics) via a shape-aware composite:

```
0.30 * F1
+ 0.15 * mean_IoU
+ 0.20 * sq(polygon_count_ratio)
+ 0.15 * sq(vertex_count_ratio)
+ 0.10 * (1 - orthogonality_delta)
+ 0.10 * (1 - compactness_delta)
```

where `sq(r) = exp(-|log(r)|)` peaks at `r=1.0` and penalises ratios symmetrically. The ratio + delta terms keep watershed `seed_min_distance` and Douglas-Peucker `simplify_m` from drifting into block-merging or zigzag-vertex regions that F1 alone wouldn't penalise.

## License

Apache-2.0. The DINOv3-ViT-S/16 encoder weights come from Facebook Research's DINOv3 (Oquab et al., 2025), Apache-2.0. The decoder architecture, multi-task training recipe, post-processing pipeline (DP+MBR-safe vectorisation, watershed instance separation), and per-area finetune scaffolding come from [`dinov3-hot-ablation`](https://github.com/kshitijrajsharma/dinov3-hot-ablation) by Kshitij Raj Sharma, Apache-2.0.

## Citation

- Encoder: [DINOv3](https://arxiv.org/abs/2508.10104) (Oquab et al., 2025).
- Decoder + training + post-processing recipe: [`dinov3-hot-ablation`](https://github.com/kshitijrajsharma/dinov3-hot-ablation) by Kshitij Raj Sharma.
- Independent metric reproduction: [polymetrics](https://github.com/kshitijrajsharma/polymetrics).
