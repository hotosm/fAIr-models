# dinov3-damage

Building damage grading from pre-event and post-event RGB imagery. A frozen DINOv3-ViT-L/16 (SAT-493M) encoder runs a siamese pass over both images, a UperNet decoder fuses them, and two heads predict building localization and damage on the xBD Joint Damage Scale. Each building footprint gets one class (no damage, minor damage, major damage, destroyed) and the mean probability of that class over its pixels. Training, inference and pooling come from the [dda](https://github.com/hotosm/dda) package; this directory wires them to fAIr.

## Summary

|          |                                                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Task     | Building damage classification per footprint                                                                                   |
| Input    | Post-event RGB chips (`post` input, dataset asset `chips-post`) and pre-event RGB chips of the same tiles (`pre`, `chips-pre`) |
| Output   | GeoJSON `FeatureCollection` of footprint polygons in EPSG:4326 with `damage_class`, `damage`, `damage_confidence` and `osm_id` |
| Window   | 512 x 512 px; training uses 2 x 2 blocks of 256 px fAIr chips, prediction slides the window over the area mosaic               |
| Training | xBD (xView2) building damage dataset                                                                                           |

## Training on fAIr

A dataset for this model carries post-event chips as `chips-post`, pre-event chips of the same tiles as `chips-pre`, and polygon labels whose `label` is 1 to 4 in the order no damage, minor damage, major damage, destroyed. `split_dataset` groups chips into 2 x 2 tile blocks, keeps the blocks that intersect the labels, and assigns whole blocks to train or val, stratified by the damage classes each block holds. `train_model` builds one window per block whose labelled pixels are at least 90% imaged in both pre and post, masks labels that fall off the imagery, and stretches each window per channel. With `strict_head` on (the default), only the localization and damage heads train. The step returns the base checkpoint unless the val damage macro F1, over the classes the labels contain, improves.

## Prediction

`predict` reads the chips of the `post` and `pre` inputs, mosaics each, and grades the footprints of the area.

| Parameter               | Meaning                                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `buildings_geojson_uri` | GeoJSON of the footprints to grade. Empty uses the OpenStreetMap `building=*` and `destroyed:building=*` footprints within the area. |
| `confidence_threshold`  | Drops buildings below this confidence. No-data buildings are always returned.                                                        |

A footprint with less than 90% of its pixels imaged in both pre and post is returned as `no-data` (class -1) with a null confidence, which keeps imagery gaps out of the four damage classes.

## Limitations

- Pre and post imagery must be co-registered closely enough that a footprint covers the same building in both; the pre image is reprojected onto the post grid at its georeferenced position.
- Minor and major damage are the hardest classes. On the xBD val split the base model reaches per-class F1 of 0.92 (no damage), 0.58 (minor), 0.66 (major) and 0.89 (destroyed), from the [model card](https://huggingface.co/hotosm/earthquake-damage-assessment-model). Those numbers are in-distribution, and the card notes that generalising to an unseen disaster is harder for minor and major damage; fine-tuning the heads on labelled buildings of the event is intended to narrow that gap.
- The ONNX graph is ViT-L sized (about 1.3 GB) and runs one window at a time on CPU when served.

## License

The pipeline code is AGPL-3.0-only, as is the rest of this repository. The pretrained checkpoint is trained on xBD and its model card publishes it under CC BY-NC-SA 4.0, so the STAC item declares `CC-BY-NC-SA-4.0`, which local models fine-tuned from it inherit.

## Citation

- DINOv3: Siméoni et al., 2025, [arXiv:2508.10104](https://arxiv.org/abs/2508.10104).
- xBD: Gupta et al., 2019, [xBD: A Dataset for Assessing Building Damage from Satellite Imagery](https://arxiv.org/abs/1911.09296).
- dda: Kshitij Raj Sharma, [github.com/hotosm/dda](https://github.com/hotosm/dda).
