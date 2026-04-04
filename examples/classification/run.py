from fair.zenml.runner import DatasetConfig, FairWorkflowRunner

runner = FairWorkflowRunner(
    base_model_id="resnet18-classification",
    model_name="resnet18-classification-finetuned-banepa",
    stac_item_path="models/resnet18_classification/stac-item.json",
    pipeline_module="models.resnet18_classification.pipeline",
    config_dir="examples/classification/config",
    dataset=DatasetConfig(
        title="buildings-banepa-classification",
        description=(
            "Binary classification labels (building/no_building) derived from the Banepa OAM+OSM segmentation dataset."
        ),
        label_type="raster",
        label_tasks=["classification"],
        label_classes=[
            {"name": "building", "classes": ["building", "no_building"]},
        ],
        keywords=["building", "classification", "polygon"],
        train_chips_path="data/sample/train/oam",
        train_labels_path="data/sample/train/classification_labels.csv",
        predict_images_path="data/sample/predict/oam",
        source_imagery_href=(
            "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
        ),
        label_description="Binary building labels derived from segmentation masks",
        label_methods=["automated"],
    ),
    finetune_overrides={"learning_rate": 0.001, "batch_size": 8},
    promote_description="ResNet18 classification finetuned on buildings-banepa-classification",
)

if __name__ == "__main__":
    runner.run()
