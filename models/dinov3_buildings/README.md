# DINOv3-L + UperNet Building Segmentation

Binary building footprint segmentation from VHR RGB aerial imagery, built on a frozen DINOv3-ViT-L/16 (LVD1689M) encoder with a UperNet decoder, multi-task auxiliary heads, and watershed instance separation. Pretrained on the global HOT VHR Building Segmentation dataset; intended for direct inference on OAM tiles or optional decoder fine-tuning on small downstream labelled sets (~100-200 chips).

## Architecture

- **Encoder**: DINOv3-ViT-L/16 (LVD1689M, ~300M params), frozen during training and inference. Intermediate features are tapped at blocks [5, 11, 17, 23] via `get_intermediate_layers(norm=True, reshape=True, return_class_token=False)`, ensuring the backbone's final LayerNorm is applied to every tap.
- **Neck**: terratorch `LearnedInterpolateToPyramidal`, ConvTranspose-based learned upsampling at scales [4x, 2x, 1x, 0.5x] producing a 4-level feature pyramid.
- **Decoder**: terratorch `UperNetDecoder` with `decoder_channels=512`, `pool_scales=(1, 2, 3, 6)`, BN+ReLU lateral and FPN convs.
- **Heads**:
  - **Main**: 1x1 Conv -> 3 channels (mask logit, boundary logit, signed distance via tanh).
  - **Auxiliary**: FCN head on tap-2 (3x3 Conv -> BN -> ReLU -> Dropout(0.1) -> 1x1 Conv) -> 3 channels. Used during training only; not exported to ONNX.
- **Trainable parameters**: ~43M (decoder + pyramid + head + aux head). Encoder remains frozen, including during downstream FT.

Native input size is 256x256 (matches the HOT VHR Building Segmentation tile size and DINOv3-L's expected operating range). fAIr-standard 256x256 chips are used directly without resizing.

## Training data

- Source: [hotosm/vhr-building-segmentation](https://huggingface.co/datasets/hotosm/vhr-building-segmentation) (~58k train, 7.2k val, 7.2k test, 256x256 chips).
- Normalisation: dataset-computed mean `[0.4297, 0.4002, 0.3433]` and std `[0.2056, 0.1674, 0.1599]` from `norm_stats.json` in the HF dataset repo.
- Null-mask tiles (no buildings in the chip) are kept in training as hard negatives, suppressing false positives over rural land. Tiles with fully-null RGB are dropped (defensive; HOT global has 0 such tiles).
- Loss: BCE+Dice on the mask logit, BCE on a 2-pixel boundary band, Huber on a signed distance map clipped to +/-15 px and normalized to [-1, 1]. The auxiliary head receives the same multi-task loss at weight 0.4. Optuna TPE chose loss weights via 12 trials on 10% data x 8 epochs (`conf/experiments/v5_hpo_best.yaml` in the upstream repo).
- Optimizer: AdamW with OneCycleLR cosine schedule, max_epochs=50, early-stop patience=5 on val/loss. Best ckpt at epoch 5.

## Inference contract

`predict(session, input_images, params)`:

- `input_images`: a directory of georeferenced RGB chips (`.tif`/`.tiff`/`.png` with their `.aux.xml` sidecars). In production `fair.serve.base._fetch_chips` populates this directory from a TMS URL + bbox via `geomltoolkits.downloader.tms`.
- `params`: `{"confidence_threshold": float}` required.
- Output: GeoJSON `FeatureCollection` of `Polygon` features in EPSG:4326.

The serve flow mirrors the upstream `dinov3-hot predict` CLI step for step, only the model forward pass is swapped to an ONNX Runtime session:

1. Merge all chips into one in-memory georeferenced raster (`rasterio.merge.merge`).
2. Sliding-window ONNX inference at 256/128 (window/stride). Gaussian-weighted stitching of the 3-channel main head into full-extent mask probability, boundary probability, and tanh-normalised signed distance maps.
3. Watershed instance separation seeded from local maxima of the distance map (`dinov3_hot.infer.instance_separate`, `seed_min_distance=4`); falls back to connected components if no peaks exist.
4. Vectorise per-instance labels with DP+MBR-safe regularisation (`dinov3_hot.infer.vectorize`, which delegates to `dinov3_hot.postprocess.vectorize_binary_mask`): Douglas-Peucker simplified in metres then substituted with the minimum rotated rectangle when near-rectangular and the swap does not increase neighbour overlap (defaults: 1 m DP, 0.55 area ratio, 1 m² overlap tolerance).
5. Reproject to EPSG:4326.

`preprocess(batch)`: divides image by 255, applies dataset mean/std normalisation, squeezes mask to long.

`postprocess(logits)`: sigmoid on channel 0 of the model output -> threshold at 0.5 -> uint8 binary mask. Used by the training-time evaluator; the serve path reads all three channels (mask, boundary, distance) for watershed seeding.

ONNX export wraps the model to return only the main 3-channel logits (drops the aux head).

## Downstream fine-tuning

The `train_model` ZenML step expects a directory of OAM RGB chips plus a single GeoJSON of OSM building polygons. Labels are burned to per-chip rasters via `geomltoolkits.raster.burn.burn_labels` and the decoder is fine-tuned with the encoder kept frozen. fAIr's local FT supervises the mask channel only (BCE+Dice); the boundary and distance heads stay at their pretrained values.

Default FT budget exposed by the pipeline:

- 100-200 chips, `epochs=15`, `batch_size=8`, `learning_rate=5e-5`, `weight_decay=0.01`.
- Observed val/iou lift on the Banepa sample: 0 to 5 percentage points. The global HF pretraining is already strong on dense-urban data, so the per-area FT delta is often small or zero (see Banepa numbers below).

The full multi-task FT recipe (boundary + distance + aux head supervision) lives in the upstream `dinov3-hot finetune` CLI.

### Train/val split

Spatial block, not random. Chip filenames are parsed as `OAM-{x}-{y}-{z}.tif` to extract TMS tile coordinates; chips are bucketed into `(x // block_size, y // block_size)` blocks and whole blocks are assigned to train or val until `val_ratio` is reached. The default `block_size=4` keeps adjacent 16-chip neighbourhoods together, so the catalog-facing val never shares spatial context with train. Filenames that don't match the OAM pattern fall back to a seeded random split.

Two val sets co-exist by design:

- **Spatial val** (`split_info.val_chip_names`): the held-out set our `evaluate_model` step measures. Reported as `fair:pixel_iou` and the instance metrics on the local-model STAC item.
- **Inner val** (`hyperparameters["inner_val_frac"]`, default 0.1): a small random slice inside the spatial-train set that upstream `dinov3_hot.finetune.finetune` uses for Lightning's per-epoch `val_loss`/`val_iou` and early stopping. It never overlaps the spatial val because we hand upstream a chips directory that contains only spatial-train chips (symlinked).

## Reference results

Numbers below are from the upstream `dinov3-hot` repo on pinned data revisions; pIoU and instance F1 are computed from raster masks, polygon shape stats from the regularised vectorised output (DP+MBR-safe in each tile's local UTM zone).

### HF VHR test split (7236 global tiles, standard benchmark)

| Pixel IoU | Precision | Recall | F1@0.5 | avg vertices | orthogonality |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **0.441** | 0.212 | 0.345 | **0.262** | 4.91 | 0.71 |

GT reference (HF mask polygons): avg vertices 4.30, orthogonality 0.91.

### Banepa, Nepal (1536x1536 OAM scene, 2720 OSM GT polygons)

Full-raster sliding-window inference matches what fAIr will invoke in deployment (the shipped ONNX with per-chip sliding-window stitching).

| Pixel IoU | Precision | Recall | F1@0.5 | avg vertices | orthogonality |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **0.656** | 0.483 | 0.351 | **0.407** | 5.99 | 0.63 |

GT reference (OSM polygons): avg vertices 5.30, orthogonality 0.95.

## License

Apache-2.0. The DINOv3-ViT-L/16 encoder weights come from Facebook Research's DINOv3 (Oquab et al., 2025), Apache-2.0. The decoder architecture, multi-task training recipe, post-processing pipeline (DP+MBR-safe vectorisation, watershed instance separation), and per-area finetune scaffolding come from [`dinov3-hot-ablation`](https://github.com/kshitijrajsharma/dinov3-hot-ablation) by Kshitij Raj Sharma, Apache-2.0.

## Citation

- Encoder: [DINOv3](https://arxiv.org/abs/2508.10104) (Oquab et al., 2025).
- Decoder + training + post-processing recipe: [`dinov3-hot-ablation`](https://github.com/kshitijrajsharma/dinov3-hot-ablation) by Kshitij Raj Sharma.
