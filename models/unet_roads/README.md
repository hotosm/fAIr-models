# Road Surface (U-Net)

## Overview

unet-roads maps roads in very high resolution RGB aerial and satellite imagery and returns the road surface as polygons. It targets under-mapped areas in low and middle income countries, where drone and satellite imagery on OpenAerialMap is available but the road network is incomplete. Mappers review the predicted road polygons in fAIr and use the ones they confirm to add or correct roads in OpenStreetMap. The model is a base model: it is meant to be fine-tuned in fAIr on local labels before it is relied on in a new area.

## Architecture

A U-Net with a ResNet34 encoder (segmentation-models-pytorch) predicts a single road probability channel for each 256 by 256 chip. fAIr chips are read as 3-band RGB and scaled to the 0 to 1 range. At prediction time the probability maps of all chips in a request are pasted into one mosaic, thresholded and vectorised, so a road comes out as one polygon across chip edges. The polygons are simplified with Douglas-Peucker and filtered by geodesic area. Training labels are burned into masks on the chip grid: polygon labels as drawn, and line or point labels buffered to a configurable road width. The loss is the sum of binary cross-entropy and Dice. Augmentation is random flips and right-angle rotations, plus random zoom and colour jitter (brightness, contrast, saturation, gamma, channel shift, noise), so the model learns road shape rather than the colours of one imagery source. The validation split holds out blocks of neighbouring tiles so a road never runs across train and validation.

## Pretrained source

The encoder starts from ImageNet weights. The released checkpoint was trained in two stages on chips at zoom 19 (about 0.3 m per pixel). The base stage used SpaceNet 3 road centrelines over Khartoum (2,798 chips) and Paris (5,846 chips), licensed CC-BY-SA-4.0, for 14 epochs. The fine-tuning stage ran 8 epochs on OpenAerialMap drone imagery (CC-BY-4.0) with OpenStreetMap highways (ODbL-1.0) over Mafalala in Maputo (90 chips), Freetown (81 chips) and Dar es Salaam (81 chips), mixed with 30 percent of the base chips. All of these labels are centrelines, buffered to 6 m to form the training masks, so the predicted surface is about that wide regardless of the real road width. Because OpenStreetMap misses roads in the drone areas, background pixels of the drone sets were weighted 0.3 in the cross-entropy term. The checkpoint with the best mean validation relaxed F1 was kept.

## Evaluation

The model reports four metrics on the held-out blocks: road pixel IoU, and relaxed precision, recall and F1, which compare the centrelines of the predicted and labelled road masks and count a match within three model pixels (about 0.9 m). On SpaceNet 3 Khartoum the released checkpoint reaches IoU 0.53, relaxed precision 0.43, recall 0.46 and F1 0.44. On SpaceNet 3 Paris it reaches IoU 0.56 and relaxed F1 0.20. On the drone sets the figures against OpenStreetMap are lower, IoU 0.14 to 0.25 and relaxed F1 0.09 to 0.24, and they understate the model because the reference labels are incomplete.

For that reason the road masks were also reviewed by eye, chip by chip, by one reviewer who looked at the predicted mask and the centreline traced from it, and marked each chip as correct, partial, missed, invented road or correctly empty. Counting correct chips fully and partial chips as half, among chips that contain a road, the score was 80 percent on a deliberately hard sample of 60 Khartoum validation chips, 78 percent on the 90 Mafalala chips (most of which were in the fine-tuning set) and 69 percent on 120 chips of Banepa, Nepal, drone imagery the model never saw in training. Chips with an invented road were 3 percent in Khartoum, 6 percent in Banepa and 13 percent in Mafalala. An earlier checkpoint trained on Khartoum alone scored 13 percent on the same Banepa chips, which is the gain from adding Paris, the colour and zoom augmentation and the drone fine-tuning.

## Limitations

The model is strongest on paved roads and wide unpaved streets. Without local fine-tuning it still misses narrow footpaths between terraced fields, tracks under tree canopy and lanes in deep shadow. Its most common false positives on drone imagery are long metal roofs and plot boundaries. Because it was trained on centrelines buffered to a fixed width, the polygons mark where a road runs but are not a measurement of its width. It was trained on a small set of geographies (Khartoum, Paris, Maputo, Freetown, Dar es Salaam), and performance elsewhere should be checked before use. The reference labels of the drone sets are incomplete, so the automatic metrics there are a lower bound, and the visual review was done by a single reviewer. Every polygon carries the same class: the model does not separate paved from unpaved roads or roads from footpaths. It sees RGB only and expects chips at roughly 30 cm per pixel; coarser imagery loses narrow roads. OpenStreetMap stores most roads as centrelines, so fine-tuning in fAIr needs a dataset of road surface polygons. For very large requests the mosaic is skipped and polygons are vectorised chip by chip, which splits a road at chip edges.

## Usage

Register the model with fAIr from its STAC item, then fine-tune it on a dataset of road surface polygons over OpenAerialMap chips built with the fAIr dataset tools. After promotion, run prediction on new chips through the fAIr client or the inference service. The confidence threshold, minimum polygon area and simplification tolerance can be tuned per request. Colour and zoom augmentation is on by default during fine-tuning and can be switched off.

## Citation

Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

Van Etten, A., Lindenbaum, D. and Bacastow, T. M. (2018). SpaceNet: A Remote Sensing Dataset and Challenge Series. arXiv:1807.01232.

Road labels for the fine-tuning stage are copyright OpenStreetMap contributors.

## License

CC-BY-SA-4.0. The checkpoint is derived from SpaceNet 3 data (CC-BY-SA-4.0), OpenStreetMap data (ODbL-1.0) and OpenAerialMap imagery (CC-BY-4.0).
