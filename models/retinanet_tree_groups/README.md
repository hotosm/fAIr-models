# retinanet-tree-groups

## Overview

Finds single trees and groups of trees in very high resolution RGB aerial imagery and returns one box polygon per tree
(class 1, `tree`) and per tree group (class 2, `tree_group`). It is a second tree model next to `deepforest_tree`:
besides single crowns it marks clusters of trees that OpenStreetMap maps as one `natural=tree_group` area, and each tree
says whether its centre lies inside a detected group. The weights were fine-tuned on HOT's tree labels in Freetown,
Sierra Leone, on OpenAerialMap imagery at zoom 19 to 21 (about 30 to 7.5 cm per pixel). They can be used directly on OAM
chips or fine-tuned on a mapper's own OSM tree labels. The model code, preprocessing, sliding-window prediction,
fine-tuning and scoring live in the MIT-licensed [`fair_trees`](https://github.com/OmdenaAI/HOTOSM-fAIr-Trees) library;
this directory connects that library to fAIr.

## Architecture

- torchvision RetinaNet with a ResNet-50 body and FPN, two classes (tree, tree_group). The pack trains it with torch
  2.11.0, with BatchNorm statistics frozen.
- The ONNX graph is the raw network only. It takes one window as `images` (float32, 1 x 3 x H x W) and returns
  `cls_logits` (one logit per anchor and class) and `bbox_regression` (four box deltas per anchor). Anchors, box
  decoding, the confidence filter, NMS, merging across windows and georeferencing to EPSG:4326 run in `fair_trees` with
  numpy, so prediction needs ONNX Runtime and no torch.
- Prediction stitches the chips into a mosaic and slides windows over it: 60 m windows for trees and 120 m windows for
  tree groups. The window sizes are read from the model file's metadata, where the published model and every fine-tune
  made by this pack carry 60 m and 120 m; they are not a setting. Each window is scaled to [0, 1], normalised with the
  ImageNet mean and std, resized (bilinear) so its short side is 800 px and its long side at most 1333 px, and
  zero-padded to a multiple of 32. Because of the resize, a window costs the same at every zoom.
- Each window side that borders another window has a narrow edge band, 2% of the window side (1.2 m for trees, 2.4 m
  for groups). Tree boxes reaching into it are dropped, and tree-group boxes reaching into it are combined with boxes
  from other windows by the `group_merge` rule. Sides with no neighbouring window keep their boxes. The band is set by
  this pack, not a setting: at the default overlap neighbouring tree windows share 7.5 m, and the library's own band of
  5% would drop most tree boxes lying across that strip. With 2%, a tree crown wider than about 5 m lying across a
  window seam can still be missed.
- Each feature carries `class`, `label`, `confidence` and `in_group`. `confidence` is the sigmoid score of the box's
  class, rounded to 4 decimals; a tree group joined across windows carries the highest score of its parts. `in_group` is
  true on a tree whose box centre lies inside a tree-group box predicted on the same chips, false on other trees, and
  null on tree groups. In live predictions it is set before the result is cut to the requested area, so a tree can say
  `true` while its group's centre lies outside the area and the group is not returned.
- Both container images take `fair_trees` from its GitHub repository at one pinned commit, the `FAIR_TREES_REF`
  argument of the Dockerfile: the commit of the library's `v0.1.0` release tag. A library change reaches fAIr only
  after that argument moves to a new commit.

## Pretrained source

The base model is release `v0.1.0` of
[`OmarHelmy/retinanet-tree-groups`](https://huggingface.co/OmarHelmy/retinanet-tree-groups/tree/v0.1.0) on Hugging Face:
`best.pt` is the checkpoint that fine-tuning starts from and `model.onnx` serves predictions. The STAC item pins both by
the commit of that release and by checksum. The model card describes three training steps. The weights start from
DeepForest's `NEON.pt` tree-crown RetinaNet
([`weecology/deepforest-tree`](https://huggingface.co/weecology/deepforest-tree)). They were fine-tuned on
NeonTreeEvaluation and OAM-TCD ([`restor/tcd`](https://huggingface.co/datasets/restor/tcd) and
[`restor/tcd-nc`](https://huggingface.co/datasets/restor/tcd-nc)), and the result was blended as 30% of these weights
and 70% of the starting weights; the card calls this blend the joint base. The joint base was then fine-tuned for 10
epochs on HOT's Freetown tree and tree_group labels
([`hotosm/fAIr-guru-td-trees`](https://huggingface.co/datasets/hotosm/fAIr-guru-td-trees) v0.1.0, OpenStreetMap
`natural=tree` points and `natural=tree_group` polygons) with the backbone frozen and random crops of 30 to 80 m; the
best epoch on validation, epoch 4, is the published model.

The scores below are F1 on HOT's Freetown labels, scored at zoom 20 only: zoom 19 and 21 imagery was used in training
but not scored. Predicted and labelled boxes are matched one-to-one (Hungarian matching) and a pair counts when its IoU
is above 0.4. Some other packs, such as `deepforest_tree`, report at IoU 0.5, so the numbers are not directly
comparable. Predicted tree boxes whose centre lies inside a labelled tree-group polygon are left out of the scores. The
test part, about 8.0 km² of Freetown, was held out from training and from picking thresholds; the validation part covers
about 5.9 km².

| Setting                                                                          | Val trees | Val tree groups | Test trees | Test tree groups |
| -------------------------------------------------------------------------------- | --------- | --------------- | ---------- | ---------------- |
| fAIr defaults: confidence 0.175 for both classes, overlap 0.125, merge `overlap` | 0.559     | 0.455           | 0.508      | 0.376            |
| Model card: confidence 0.2 trees and 0.15 groups, overlap 0.5, merge `drop_edge` | 0.573     | 0.477           | 0.531      | 0.333            |

fAIr serves this pack at the first setting, with the pack's edge band of 2%, so the first row is the one that describes
predictions on fAIr. The model card quotes the second row: the per-class thresholds picked on validation, with the
library's own defaults for window overlap, edge band (5%) and group merge (`drop_edge` drops tree-group boxes cut by a
window edge). The fAIr setting gives a little less tree F1 and more tree-group F1 on test, and the card's setting needs
about 2.7 times its window passes. The default overlap was chosen for serving cost: at overlap 0.25, with the same
threshold, band and merge, tree recall is higher (0.544 against 0.517 on validation, 0.545 against 0.527 on test) for
about a third more window passes.

## Limitations

- Scored in one city, Freetown, at one zoom, 20. Other cities, sensors, seasons and zoom levels are unmeasured.
  Imagery coarser than 2 m per pixel is refused, and imagery coarser than about 0.62 m per pixel, coarser than training
  saw, is predicted with a warning.
- Tree groups are much weaker than trees (test F1 0.376 against 0.508 at the fAIr defaults). Most OSM tree data maps
  single trees only, so a mapper's fine-tune seldom has group labels to learn from.
- Boxes are approximate. An OSM tree is a point, so its label box is a square of its `diameter_crown` tag or, without
  one, of a 4 m default (see Usage); the scores rest on these boxes. The model draws axis-aligned boxes, not crown
  outlines.
- Outside labelled tree-group polygons, every chip of a fAIr dataset counts as fully mapped: a tree there that the
  labels miss is background in training and a false positive in the validation scores. Inside a labelled tree-group
  polygon, tree predictions are neither penalised in training nor scored (for scoring, a predicted tree counts as inside
  when its box centre is).
- The weights are for non-commercial use only (CC-BY-NC-4.0), and so are models fine-tuned from them.
- Sliding windows are slow on CPU. One window took about 1.3 s with ONNX Runtime limited to 2 threads, measured on a
  laptop, not on fAIr's servers. At the default overlap, an area of about 200 m square at zoom 20 needs about 34
  windows, so live predictions suit small areas; larger areas belong in batch predictions. Each prediction stitches its
  chips into mosaics, and the bounding rectangles of those mosaics may add up to at most 268,435,456 pixels
  (16,384 x 16,384).
- Training requests a GPU. Each training crop is resized to 800 px on its short side, so memory per crop is high.

## Usage

On fAIr this pack is used like the other trainable models (see the contributing guide): a mapper registers a dataset
of OAM chips and OSM tree labels, fine-tunes this base model on it, promotes the result and predicts with it. Defaults
and allowed ranges of every setting are in the STAC item's hyperparameter list.

Chips must be square 8-bit RGB GeoTIFFs named `OAM-{x}-{y}-{z}.tif`, as fAIr's tile download names them. Chips with
other names are not read, and a folder that holds only chips with other names is an error.

Labels are read as follows. A feature's OSM `natural` tag decides its class when it is `tree` or `tree_group`; otherwise
fAIr's `label` number does (1 tree, 2 tree_group, missing means tree). A tree point becomes a square crown with the
side given by its `diameter_crown` tag in metres (values such as "3", "3m" or "3,5 m"; above 30 m counts as missing).
Tree points without a usable diameter are handled per labels file: when at least half of the file's tree points have
one, the others are masked (left out of training and scoring); otherwise they become 4 m squares. A tree polygon becomes
one box per polygon part, and a tree_group polygon one box around all of its parts. Tree_group points and lines are
skipped.

fAIr compares the geometry type of a dataset and a model only through the geometry words in their keywords, and this
model's keywords name only `polygon`. A dataset whose keywords name `point` and not `polygon` is therefore refused
before training, even though the pipeline itself can read tree points.

Fine-tuning splits the chips by spatial blocks of their tile coordinates (4 x 4 tiles at the lowest zoom present by
default), so neighbouring chips stay on one side; whole blocks go to validation until about 20% of the chips are there.
A dataset that fits in one block gets no validation chips and trains without validation. Training starts from the
`v0.1.0` checkpoint and by default trains only the FPN and the detection heads (`trainable_layers` 0, as the HOT
fine-tune did), with SGD at learning rate 0.001 and a cosine schedule, for up to 10 epochs. It stops after 3 epochs
without a validation improvement and keeps the best epoch. Each epoch draws random crops 30 to 80 m wide;
`samples_per_epoch` 0 draws one crop per train chip. `sample_fraction` thins only the validation chips, keeping the
first in name order, for quick test runs; the train chips are never thinned. A dataset that labels only one class (for
example single trees without groups) trains that class, and the model keeps the base model's detection of the other.
Data whose labels fall on no train chip is refused before training, with a message to check that the labels cover the
chips' area.

The reported metrics are precision, recall and F1 for each class on the validation chips, at the fAIr defaults for
confidence, window overlap, edge band and group merge, with IoU above 0.4. Two counts, `val_trees` and
`val_tree_groups`, give the number of labelled boxes each class's scores rest on; 0 means that class was not scored, and
its scores read 0.

Prediction takes three parameters. `confidence_threshold` (default 0.175, 0 to 1) keeps the boxes scoring above it, for
trees and tree groups alike; a value the caller sends, including 0, always wins. `window_overlap` (default 0.125, 0 to
0.5) is the overlap of neighbouring windows as a share of the window side; 0.25 needs about a third more window passes
than the default and 0.5 about 2.7 times as many. `group_merge` (default `overlap`) chooses how tree-group boxes cut by
a window edge are combined: `overlap` joins a cut box with boxes from other windows that cover more than half of the
smaller box, `join` with any box from another window it overlaps, and `drop_edge` drops cut boxes. A value outside these
ranges, or a `group_merge` other than these three, is refused with an error that names the parameter. Other parameters
are ignored with a warning, so predictions use the settings the metrics were scored with. The per-class thresholds a
fine-tune picks on validation are stored in its model file, but this pack always hands the library one threshold, the
caller's or 0.175, so the stored ones are not used on fAIr. Live predictions keep only the boxes whose centre lies in
the requested area; batch predictions keep every box on the given chips. Chips of several zoom levels are predicted
separately.

## Citation

- DeepForest: Weinstein et al. 2020, DeepForest: A Python package for RGB deep learning tree crown delineation.
  https://doi.org/10.1111/2041-210X.13472
- OAM-TCD: Veitch-Michaelis et al. 2024, OAM-TCD: A globally diverse dataset of high-resolution tree cover maps.
  https://arxiv.org/abs/2407.11743
- NeonTreeEvaluation: Weinstein et al., Zenodo. https://doi.org/10.5281/zenodo.5914554
- RetinaNet: Lin et al. 2017, Focal Loss for Dense Object Detection. https://arxiv.org/abs/1708.02002
- `fair_trees` and the weights: Omdena, [HOTOSM-fAIr-Trees](https://github.com/OmdenaAI/HOTOSM-fAIr-Trees) and
  [`OmarHelmy/retinanet-tree-groups`](https://huggingface.co/OmarHelmy/retinanet-tree-groups/tree/v0.1.0).

## License

This directory's code is licensed AGPL-3.0-only, the licence of the whole fAIr-models repository. It calls the
`fair_trees` library, which is MIT. The weights are [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/),
non-commercial use only: about 6% of the OAM-TCD tiles used to train the joint base (see Pretrained source) come from
`restor/tcd-nc`, which is CC-BY-NC-4.0. The STAC item declares `CC-BY-NC-4.0`, which models fine-tuned from it inherit.

Attribution:

- HOT tree labels: (c) OpenStreetMap contributors, [ODbL-1.0](https://www.openstreetmap.org/copyright). Imagery of
  `hotosm/fAIr-guru-td-trees`: OpenAerialMap, [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), contributors of
  the Open Imagery Network.
- OAM-TCD: Restor, `restor/tcd` (CC-BY-4.0) and `restor/tcd-nc` (CC-BY-NC-4.0).
- NeonTreeEvaluation: Weinstein et al. (CC-BY-4.0).
- DeepForest `NEON.pt`: Weecology, `weecology/deepforest-tree` (MIT).
