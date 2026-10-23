# dinov3s-road-surface

## Overview

Classifies the surface of mapped roads as paved or unpaved from aerial and drone imagery. It does not find roads: it needs the road geometries (OSM ways) together with the imagery and assigns a surface class to each of them. For every road, the model cuts up to four small image patches at points along the road line and judges the road from all of them together, so one patch where the road is hidden by a tree or a building doesn't decide the result (see [How the bagging approach works](#how-the-bagging-approach-works)).

The model is a DINOv3 ViT-S/16 encoder with attention pooling and a linear classifier head, trained with multi-instance learning on these bags of patches. It packages [SUSformer](https://github.com/GIScience/Road-surface-classification-for-FAIR) ("Surface from UAV Scenery") by HeiGIT for fAIr. Meta's [DINOv3](https://arxiv.org/abs/2508.10104) is pretrained on general imagery; for SUSformer, HeiGIT fine-tuned the encoder on labelled road patches from 124 OpenAerialMap scenes so that its features capture road surfaces better than the stock weights do.

### Summary

|           |                                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------------------- |
| Task      | Road surface classification (paved / unpaved) from imagery + road geometry                                    |
| Input     | 3-band RGB imagery chips plus OSM road `LineString`s (as a `labels.geojson`) covering the same scene          |
| Output    | One prediction per road: predicted `class`/`label`, `confidence`, carrying through `osm_id` and `highway`     |
| Coverage  | Global; base model trained on a worldwide catalogue of OpenAerialMap drone/satellite scenes                   |
| Use cases | Road/route-planning decision support where surface type affects vehicle accessibility                         |
| License   | Code: MIT. Encoder weights: [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |

## Classes

SUSformer labels roads with three classes. This model merges the two paved ones and works with two:

| id  | name      | meaning                                                                                                                                                                        |
| --- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0   | `unpaved` | No continuous pavement or segmented pavement e.g. dirt, gravel, ground                                                                                                         |
| 1   | `paved`   | Continuous pavement (asphalt, concrete, tarmac) or segmented pavement (sett, cobblestone, paving stones); merges the upstream `paved_continuous` and `paved_segmental` classes |

## Intended use

- A decision-support signal for road/route planning where surface type affects vehicle accessibility.
- Classifies roads that are already mapped in OSM and does not detect new ones. Where an area is poorly mapped, the roads have to be mapped first.
- Works best on OAM-style drone and high-resolution satellite imagery; low-resolution imagery is harder for it to read reliably.

## How the bagging approach works

The model classifies a road from several close-up patches taken at different points along it. Together these patches form a "bag" (multi-instance learning), and attention pooling combines them into one decision, giving the clearest patches the most weight. One patch on its own can mislead: a tree or a building may cover the road, or the image may be blurred there.

![Example patches along roads](https://raw.githubusercontent.com/GIScience/Road-surface-classification-for-FAIR/f92eeb2b4c7903eaaf34ead8879732b9e38f2c96/images/patches_examples.png)

In this fAIr packaging, bags are cut by `road_surface_classifier.data.materialize.materialize_dataset` per labelled road, configured via `PATCH_CONFIG_KWARGS` in `pipeline.py`:

- Patch size: 15 m along the road, rendered at 256x256 px.
- Patches spaced roughly every 20 m along the road, 1-4 patches per road.
- 5 bags per labelled road, each cut with lateral jitter (1.5 m) and bearing jitter (10°), so the model doesn't depend on exactly where along a road a patch is taken.
- Patches too close to junctions (7.5 m clearance) or to a road's endpoints (10% margin) are avoided; junction clearance ignores minor highway values (`footway`, `path`, `cycleway`, `steps`, etc.).

![Patches of one road with augmentation](https://raw.githubusercontent.com/GIScience/Road-surface-classification-for-FAIR/f92eeb2b4c7903eaaf34ead8879732b9e38f2c96/images/patches_with_augmentation.png)

## Pretrained source

The encoder starts from Meta's DINOv3 ViT-S/16, pretrained on the LVD-1689M image collection (Siméoni et al., 2025, see [Citation](#citation)). For SUSformer, HeiGIT fine-tuned this encoder on labelled road patches and trained the attention pooling and the linear classifier head on the same data. Code and training setup are released as Global SUSformer 1.0.0 (Szamek and Langer, 2026, [doi:10.5281/zenodo.23396797](https://doi.org/10.5281/zenodo.23396797)).

### Training dataset

The training dataset consists of 124 OpenAerialMap scenes, mostly drone captures with a smaller share of high-resolution satellite imagery. The scenes cover most world regions; regions without usable OAM scenes remain gaps. For every scene, the OSM `highway=*` ways within its footprint were fetched through HeiGIT's ohsome API, and a small team labelled 4,649 road segments by hand in QGIS. The labels follow three classes (`unpaved`, `paved_continuous`, `paved_segmental`), which this model merges into two (see [Classes](#classes)). The OSM `surface` tag served as a hint during labelling but not as ground truth. Class shares, the regional distribution and the image resolutions are shown under [Training data](#training-data).

### Training setup

SUSformer trains the complete network, encoder included, on bags cut once offline from the labelled roads. Each road gets five bags of 15 m patches at 256 × 256 px, cut with lateral and bearing jitter, and the patches receive pixel-level augmentation during training. Training uses AdamW (weight decay 0.05) with a one-cycle learning-rate schedule peaking at 2e-4. A layer-wise decay of 0.9 lowers the learning rate towards the input layers of the encoder, which stays frozen for the first five epochs. The loss is class-weighted cross-entropy with a label smoothing of 0.1, each batch holds 32 bags, and an exponential moving average of the weights (decay 0.995) is used for validation and for the saved checkpoints. Training runs for up to 100 epochs with early stopping, and the last ten epochs use complete, unaugmented bags, so that the model settles on bags like those it sees at inference. The accuracies under [Reliability under degraded conditions](#reliability-under-degraded-conditions) come from a 5-fold cross-validation split by scene (seed 42), so roads of a validation scene never appear in the training folds of that split. The released model was then trained on all 124 scenes with the same setup.

### Data license

The imagery comes from the Open Imagery Network behind OpenAerialMap and is licensed CC BY 4.0, with attribution to the contributors of the Open Imagery Network. The road geometries are OSM ways (© OpenStreetMap contributors, ODbL), to which the SUSformer labellers assigned the surface classes by hand. <!-- TODO: state the license of the SUSformer surface labels --> The fine-tuned encoder weights are a derivative of DINOv3 and fall under the DINOv3 License (see [License](#license)).

## Training data

The base model is trained on SUSformer's labelled catalogue:

![Distribution of the training scenes](https://raw.githubusercontent.com/GIScience/Road-surface-classification-for-FAIR/f92eeb2b4c7903eaaf34ead8879732b9e38f2c96/images/scenes_distribution.png)

124 labelled scenes from OpenAerialMap with 4,649 labelled road segments, spread across the world.

The hand labels use three classes: `unpaved` 1,784 (38.4%), `paved_continuous` 2,600 (55.9%) and `paved_segmental` 265 (5.7%). After merging the paved classes, that is 1,784 (38.4%) unpaved and 2,865 (61.6%) paved roads. The training loss is class-weighted cross-entropy to make up for the imbalance. Fine-tuning keeps the base model's class weights, so a new region with a very different class balance may need its own weighting.

![Labelled roads per World Bank region](https://raw.githubusercontent.com/GIScience/Road-surface-classification-for-FAIR/f92eeb2b4c7903eaaf34ead8879732b9e38f2c96/images/pie_chart_per_wb_region.png)

![Labels per class and World Bank region](https://raw.githubusercontent.com/GIScience/Road-surface-classification-for-FAIR/f92eeb2b4c7903eaaf34ead8879732b9e38f2c96/images/labels_per_class_and_wb_region.png)

Most labelled scenes are high-resolution (5 cm and below) UAV imagery, but UAV and satellite imagery with a lower resolution is included too:

![Resolution of the training scenes](https://raw.githubusercontent.com/GIScience/Road-surface-classification-for-FAIR/f92eeb2b4c7903eaaf34ead8879732b9e38f2c96/images/resolutions.png)

For a local fAIr fine-tune, `train_model` instead takes your own `dataset_chips` (imagery) and `dataset_labels` (a `labels.geojson` of roads, see [Input data](#input-data)) and re-materializes bags from those.

## Reliability under degraded conditions

Upstream macro-balanced accuracy, mean ± std over 5-fold cross-validation, each fold scored on its held-out validation scenes:

| condition                        | approx. GSD | accuracy     |
| -------------------------------- | ----------: | ------------ |
| normal use (full set of patches) |      5.9 cm | 96.5% ± 0.7% |
| only 1 patch per road            |      5.9 cm | 92.9% ± 0.8% |
| only 2 patches per road          |      5.9 cm | 94.9% ± 0.6% |
| only 4 patches per road          |      5.9 cm | 96.1% ± 0.7% |
| imagery downsampled 2x           |       12 cm | 96.3% ± 1.1% |
| imagery downsampled 4x           |       23 cm | 94.4% ± 1.9% |
| imagery downsampled 8x           |       47 cm | 91.8% ± 1.7% |
| imagery downsampled 16x          |       94 cm | 86.0% ± 2.8% |

The GSD is the patches' effective ground sampling distance: each 15 m patch is cut at 256 px (15 m / 256 px ≈ 5.9 cm per pixel), as in this fAIr pipeline, and the downsampled conditions reduce the patches by the factor before scaling them back up. It is the finest detail the model gets; scenes captured coarser than that carry less.

Macro AUROC under normal use: 99.5% ± 0.3%.

A single patch per road costs about 3.6 percentage points of accuracy, and 2x or 4x downsampling costs little. From 8x downsampling (about 47 cm GSD) on, accuracy drops clearly, so predictions on imagery that coarse need more checking.

## Examples

Prediction of the base model (SUSformer): brown shows unpaved and blue paved roads, the hue shows the model's confidence.

![Example prediction](https://raw.githubusercontent.com/GIScience/Road-surface-classification-for-FAIR/f92eeb2b4c7903eaaf34ead8879732b9e38f2c96/images/example1.png)

## Limitations

- Misalignment between OSM and imagery: training and testing only used patches in which enough of the road was visible in the drone imagery. In practice, OSM road geometries and imagery are often offset, so patches cut along an OSM way can partly or entirely miss the road. Small offsets degrade predictions; larger ones can make them wrong. Check the alignment of roads and imagery before relying on the output.
- Coarse imagery: the training scenes go up to about 30 cm GSD and none are coarser; most are far finer, around 4–5 cm. Training and evaluation at coarser resolutions therefore rely mostly on downsampled versions of fine scenes (see [Reliability under degraded conditions](#reliability-under-degraded-conditions)), not on imagery captured that coarse. At lower resolutions the model struggles with narrow roads and dense urban scenes, where a road covers only a few pixels or is hard to separate from its surroundings. The zoom level of the input imagery only matters indirectly: every patch is resampled to 256 px per 15 m of road, so a lower zoom gives the model the same patch size with less detail in it. Use the highest zoom level the imagery offers.
- Label coverage: roads were labelled only where the labellers could identify the surface without ground reference, either from the imagery or from existing OSM tags. Scenes where the surface cannot be determined from the imagery are probably missing from both training and testing, so the reported accuracy does not cover them.

## Usage

### Input data

Fine-tuning takes two directories (local paths or `s3://` prefixes):

- `dataset_chips`: GeoTIFF imagery (`*.tif`) covering the labelled roads; several chips are mosaicked into one scene.
- `dataset_labels`: a single `labels.geojson` with the roads, following this contract:

| Field                       | Required | Contract                                                                                                                                                                                                                                                                                     |
| --------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| geometry                    | yes      | `LineString` or `MultiLineString` road centrelines; other geometry types are dropped.                                                                                                                                                                                                        |
| `label`                     | yes      | Integer surface class: `0` = unpaved, `1` = paved. Leave it empty (`null`) for roads you don't label: they are not trained or evaluated on, but still count as road network for junctions. No other values are accepted (collapse SUSformer's 3-class `2` = `paved_segmental` to `1` first). |
| `osm_id` or defined Road id | yes      | Road id (the column is set by the `id_column` hyperparameter). All features with the same id stay in the same split, so pieces of one OSM way never leak between train, val and test.                                                                                                        |
| `highway`                   | no       | OSM `highway` value. Lines tagged `footway`, `path`, `cycleway`, `steps`, `pedestrian`, `corridor`, `platform`, `construction` (and other non-carriageway values) are not treated as junctions when patches are placed.                                                                      |

A minimal feature is a `LineString` with the properties `osm_id`, `highway` and `label`, for example a residential road (`highway` `residential`) labelled as paved (`label` 1).

### Fine-tuning (`training_pipeline`)

1. `split_dataset` splits the labelled roads into train/val/test by id (`id_column`, default `osm_id`): the ids form `n_clusters` (default 10) spatially compact clusters of equal size by their centroid, and in seeded random order test gets `test_ratio` and val `val_ratio` of the clusters (at least one each), the rest is train; not stratified by class. Defaults: `val_ratio=0.2`, `test_ratio=0.2`, `split_seed=42`.
2. `train_model` loads the base model (frozen encoder and attention pool, trainable linear classifier) and first checks a gate: if the base model already scores at or above `gate_accuracy` (default 0.95) on all labelled roads in the scene, fine-tuning is skipped and the base model is returned unchanged. Otherwise it fine-tunes the classifier head for `epochs` (default 10) with AdamW at `learning_rate` (default 1e-3). Each epoch is one pass over the train roads in random order, in batches of `batch_size` (default 8) roads with one random bag each; every batch mixes in `replay_bags_per_step` (default 20) bags of cached backbone features from the base model's own training scenes, so the head doesn't drift off the original distribution. It validates after every epoch.
3. `evaluate_model` reports class-weighted cross-entropy loss and accuracy on the held-out test roads.
4. `export_onnx` exports the model as one ONNX file (input: one road's bag of 1–4 patches, output: its logits), with the DINOv3 License and "Built with DINOv3" bundled in the file's metadata.

### Monitoring fine-tuning

- MLflow, per step: `train_loss` (train roads plus replay bags).
- MLflow, after every epoch: `val_loss` and `val_accuracy`, and once per run the gate's `base_accuracy`. The gate check runs inside the training run, so a run that keeps the base model still shows up with its parameters.
- Loss history: one point per epoch (mean train loss over its batches, val loss). Promotion publishes it as the local model's training metrics.

### Running locally

The local stack of this repository (`just setup`) runs the full cycle for this model. `just example dinov3s_road_surface_classification` registers the base model, fine-tunes it with the training pipeline on a sample dataset and runs inference with the result. The sample needs road labels following [Input data](#input-data), since the bundled building sample contains no road lines. `just serve dinov3s_road_surface_classification` starts the inference container on port 8090, which answers `POST /predict` requests naming the ONNX model, a tile source for the imagery, a bounding box and a zoom level. For each request, the container fetches the OSM roads within the bounding box, classifies them and returns a GeoJSON `FeatureCollection` with `class`, `label` and `confidence` per road (parameters under [Inference](#inference)). After `just example`, `just test-serve dinov3s_road_surface_classification` sends such a request against the fine-tuned ONNX file.

## Hyperparameters

All defaults live in the STAC item (`mlm:hyperparameters`); fAIr passes the `training.*` values to the training pipeline and the `inference.*` values to `predict`.

### Training

| Parameter              |                                    Default | Meaning                                                                                                                             |
| ---------------------- | -----------------------------------------: | ----------------------------------------------------------------------------------------------------------------------------------- |
| `epochs`               |                                         10 | Passes over the train roads; each road contributes one random bag (of 5) per epoch                                                  |
| `batch_size`           |                                          8 | Train roads per batch (one optimizer step); an epoch is ceil(train roads / `batch_size`) batches, the last one can be smaller       |
| `learning_rate`        |                                      0.001 | AdamW learning rate for the classifier head (encoder and attention pooling stay frozen)                                             |
| `replay_bags_per_step` |                                         20 | Bags from the replay bank mixed into every batch, so the head keeps working on the base model's training scenes                     |
| `gate_accuracy`        |                                       0.95 | If the base model already reaches this accuracy on all labelled roads of the scene, fine-tuning is skipped and the base model kept  |
| `val_ratio`            |                                        0.2 | Share of the road clusters held out for validation                                                                                  |
| `test_ratio`           |                                        0.2 | Share of the road clusters held out for testing (used only by `evaluate_model`)                                                     |
| `id_column`            |                                   `osm_id` | Label column identifying a road; all features with the same id stay in the same split                                               |
| `n_clusters`           |                                         10 | Number of spatially compact road clusters the split assigns to train, val and test                                                  |
| `split_seed`           |                                         42 | Seed for the order in which clusters are assigned to the splits                                                                     |
| `encoder_uri`          |           `dinov3_encoder.zip` (SUSformer) | Fine-tuned DINOv3 ViT-S/16 encoder `state_dict` (DINOv3 License), as `encoder.pt` or a `.zip` holding it; loaded with the base head |
| `replay_bank_uri`      | `replay_bank_2000_features.pt` (SUSformer) | Replay bank of per-patch encoder features from the base model's training scenes                                                     |

An epoch shows every train road once, with one of its 5 pre-cut bags picked at random and augmented, so 10 epochs show each bag about twice. `batch_size` sets how many roads go into one optimizer step: with 320 train roads and the default 8, an epoch is 40 steps and the default fine-tune 400 steps. Fewer train roads mean fewer steps per epoch. If fine-tuning on a small scene barely changes the model, lower `batch_size` instead of raising `epochs`: that adds updates while each bag is still seen only about twice. Each road costs up to 4 encoder passes, so `batch_size` also bounds the time and memory per step. The replay bags are added on top, so a small `batch_size` with the default 20 replay bags weighs the base scenes more.

Keep `epochs` and `learning_rate` low. Fine-tuning in fAIr sees the labelled roads of a single scene. More epochs or a higher learning rate make the head fit that one scene and drift away from what it learned on the 124 base scenes, so predictions elsewhere get worse. Watch `val_loss` per epoch in MLflow: if it rises while `train_loss` keeps falling, use fewer epochs. The replay bags and the accuracy gate work against this drift as well: raise `replay_bags_per_step` to stay closer to the base model, lower it to adapt more to the new scene.

### Inference

| Parameter              |                                 Default | Meaning                                                                                                                                                                                                            |
| ---------------------- | --------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `confidence_threshold` |                                     0.5 | Below this softmax confidence a road is classed `unknown` (`class` is null); 0.5 accepts every prediction of the two classes                                                                                       |
| `highway_values`       | roads and paths (list in the STAC item) | Comma-separated OSM `highway` values to classify: roads of all classes, `track`, `service`, and smaller ways (`footway`, `path`, `pedestrian`, `cycleway`, `bridleway`); only these ways are fetched and predicted |
| `exclude_tags`         |     `footway=sidewalk,footway=crossing` | Comma-separated `key=value` OSM tags; ways carrying any of them are left out, by default sidewalks and crossings mapped as separate footways                                                                       |

Raise `confidence_threshold` to report only confident predictions (more roads become `unknown`). Footways and paths are classified, but sidewalks are not: in OSM a sidewalk is a `highway=footway` way tagged `footway=sidewalk`, which `exclude_tags` removes. Steps, corridors, platforms and ways under construction are not in `highway_values`.

## Architecture

- Encoder: DINOv3-ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m`, pretrained on LVD-1689M, ~22M params), fine-tuned for SUSformer on its road training data (see [Training data](#training-data)). Frozen during fAIr fine-tuning.
- Pooling: attention pooling over a road's bag of patch encodings; clearer patches get more weight.
- Head: linear classifier on the pooled representation, the only trainable part during fAIr fine-tuning.
- Patch input: 256x256 RGB, ~15 m of road per patch.

### Model size

- Parameters: 21.69 M (encoder, attention pooling and classifier head)
- Compute: 5.62 GMACs (about 11.2 GFLOPs, counting one multiply-add as two FLOPs) per patch; a road of 4 patches takes 22.5 GMACs
- Inference input: `float32` tensor of `patches × 3 × 256 × 256`, the bag of 1 to 4 patches of one road; inference runs one road at a time
- CPU inference: about 30 ms per road with 1 patch, about 109 ms with 4 patches (median, ONNX)
- Memory and benchmark host: about 359 MiB memory usage after model warm-up; Intel Core Ultra 7 255H (16 logical CPUs), CPU ONNX Runtime

## License

Code (this package and the upstream [SUSformer](https://github.com/GIScience/Road-surface-classification-for-FAIR) source) is MIT licensed. The DINOv3-ViT-S/16 encoder weights are fine-tuned from Meta's [DINOv3](https://github.com/facebookresearch/dinov3), and as a derivative of the DINOv3 Materials they are not MIT licensed. They are distributed under the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Among other things, that license requires that any redistribution of the weights includes a copy of the license, that use complies with applicable trade-control laws and excludes the license's prohibited end uses (e.g. military/warfare, nuclear, espionage, weapons, ITAR-controlled activities), and that publications using the weights acknowledge DINOv3. Read the full DINOv3 License before using or redistributing the model weights.

## Citation

If you use this model or its weights, please cite the DINOv3 paper it is based on:

> Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C.,
> Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F.,
> Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T.,
> Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C.,
> Mairal, J., Jégou, H., Labatut, P., & Bojanowski, P. (2025).
> _DINOv3_. arXiv:2508.10104. https://arxiv.org/abs/2508.10104

Source: [GIScience/Road-surface-classification-for-FAIR](https://github.com/GIScience/Road-surface-classification-for-FAIR) (SUSformer), [doi:10.5281/zenodo.23396797](https://doi.org/10.5281/zenodo.23396797).
