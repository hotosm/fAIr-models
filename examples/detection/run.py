from fair.zenml.runner import DatasetConfig, FairWorkflowRunner

runner = FairWorkflowRunner(
    base_model_id="yolo11n-detection",
    model_name="yolo11n-detection-finetuned-banepa",
    stac_item_path="models/yolo11n_detection/stac-item.json",
    pipeline_module="models.yolo11n_detection.pipeline",
    config_dir="examples/detection/config",
    dataset=DatasetConfig(
        title="buildings-banepa-detection",
        description=("COCO-format building detection labels derived from the Banepa OAM+OSM segmentation dataset."),
        label_type="vector",
        label_tasks=["object-detection"],
        label_classes=[{"name": "building", "classes": ["building"]}],
        keywords=["building", "object-detection", "polygon"],
        train_chips_path="data/sample/train/oam",
        train_labels_path="data/sample/train/detection_labels.json",
        predict_images_path="data/sample/predict/oam",
        source_imagery_href=(
            "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
        ),
        label_description="Building detection labels in COCO format derived from segmentation masks",
        label_methods=["automated"],
    ),
    finetune_overrides={"learning_rate": 0.01, "batch_size": 8, "chip_size": 640},
    promote_description="YOLOv11n detection finetuned on buildings-banepa-detection",
)

if __name__ == "__main__":
    runner.run()
