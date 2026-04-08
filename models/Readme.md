# Base Models

Each subdirectory is one base model contribution. A base model is a reusable ML
blueprint that users finetune on their own datasets through the fAIr platform.

| Model | Task | Architecture | Source | Example |
|---|---|---|---|---|
| [`unet_segmentation`](unet_segmentation/) | Semantic segmentation | UNet (torchgeo) | [OAM-TCD](https://arxiv.org/abs/2407.11743) | [`examples/segmentation`](../examples/segmentation/) |
| [`resnet18_classification`](resnet18_classification/) | Binary classification | ResNet18 (torchvision) | ImageNet | [`examples/classification`](../examples/classification/) |
| [`yolo11n_detection`](yolo11n_detection/) | Object detection | YOLOv11 nano (ultralytics) | COCO | [`examples/detection`](../examples/detection/) |

## Contributing

See [Contributing a Model](../docs/contributing/model.md) for the full guide on
adding a new base model.
