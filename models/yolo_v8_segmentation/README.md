# YOLOv8 Building Footprint Segmentation

## Overview

YOLOv8 Building Footprint Segmentation is a base model for extracting building footprints from very high resolution RGB aerial imagery. The model is intended for fAIr finetuning workflows where users provide OpenAerialMap chips and matching vector labels, and the system produces polygon-ready outputs through the model's preprocessing and postprocessing entrypoints.

## Architecture

This model uses a YOLOv8 segmentation backbone with a single foreground class for buildings. Training and inference are orchestrated through the fAIr ZenML pipeline contract, while geospatial preprocessing and polygonization are delegated to the shared `hot_fair_utilities` stack. The model expects RGB chip tensors and returns building mask predictions that are postprocessed into GeoJSON building polygons.

## Pretrained Source

The pretrained initialization checkpoint is the YOLOv8 segmentation weight published through the HOT fAIr utilities repository. It is used as the base checkpoint for finetuning on user-provided datasets in the fAIr platform.

## Limitations and Bias

Performance is sensitive to image resolution, acquisition quality, and domain shift between training and inference regions. Dense urban settlements, informal settlements, heavy shadows, cloud cover, and occlusions may reduce recall and boundary quality. Geographic and annotation bias in source training data can propagate to outputs, especially in underrepresented building styles and roof materials. Predictions should therefore be reviewed before operational use in humanitarian mapping workflows.

## Usage

This model is designed to be used through fAIr training and inference pipelines rather than standalone scripts. During training, users provide dataset chip and label assets plus hyperparameter overrides, and the pipeline performs deterministic train/validation splitting, model training, evaluation, and ONNX export. During inference, users provide model URI and input imagery; the pipeline runs segmentation and postprocesses outputs into geospatial building polygons.

## Citation

If you use this model in downstream work, cite the Ultralytics YOLO family and the HOT fAIr model packaging workflow. The STAC item includes a `cite-as` link for model lineage reference.

## License

This model is distributed under the Apache-2.0 license, consistent with the STAC metadata for the model package and source code.
