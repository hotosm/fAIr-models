# Base Models

Each subdirectory is one base model contribution. A base model is a reusable ML
blueprint that users finetune on their own datasets through the fAIr platform.

| Model                                                                     | Task                  | Architecture                        | Source                                                              | Example                                          |
| ------------------------------------------------------------------------- | --------------------- | ----------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------ |
| [`dinov3s_buildings`](dinov3s_buildings/)                                 | Semantic segmentation | DINOv3 ViT-S/16 + UperNet (PyTorch) | HOT VHR Building Segmentation                                       | `just example dinov3s_buildings`                 |
| [`yolo_swag_waste_grid_segmentation`](yolo_swag_waste_grid_segmentation/) | Semantic segmentation | YOLO26x classifier (ultralytics)    | [SWAG](https://github.com/GIScience/solid-waste-detection-for-fAIr) | `just example yolo_swag_waste_grid_segmentation` |
| [`sklearn_rgb_segmentation`](sklearn_rgb_segmentation/)                   | Semantic segmentation | Logistic regression (scikit-learn)  | Minimal example, no deep-learning stack                             | `just example sklearn_rgb_segmentation`          |

## Contributing

See [Contributing a Model](../docs/contributing/model.md) for the full guide on
adding a new base model.
