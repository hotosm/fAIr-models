from fair.client import DatasetConfig, FairClient

client = FairClient(
    zenml_store_url=None,
    stac_api_url=None,
    dsn=None,
    user_id="anonymous",
    config_dir="examples/classification/config",
)

if __name__ == "__main__":
    from fair.utils import install_s3_cleanup_handler

    install_s3_cleanup_handler()

    client.setup()

    base_model_id = client.register_base_model("models/resnet18_classification/stac-item.json")

    dataset_id = client.register_dataset(
        DatasetConfig(
            title="buildings-banepa-classification",
            description=(
                "Binary classification labels (building/no_building) derived from the "
                "Banepa OAM+OSM segmentation dataset."
            ),
            label_type="raster",
            label_tasks=["classification"],
            label_classes=[{"name": "building", "classes": ["building", "no_building"]}],
            keywords=["building", "classification", "polygon"],
            train_chips_path="data/sample/train/oam",
            train_labels_path="data/sample/train/classification_labels.csv",
            predict_images_path="data/sample/predict/oam",
            source_imagery_href=(
                "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
            ),
            label_description="Binary building labels derived from segmentation masks",
            label_methods=["automated"],
        )
    )

    finetuned_model_id = client.finetune(
        base_model_id=base_model_id,
        dataset_id=dataset_id,
        model_name="resnet18-classification-finetuned-banepa",
        overrides={"learning_rate": 0.001, "batch_size": 8},
    )

    local_model_id = client.promote(
        finetuned_model_id,
        description="ResNet18 classification finetuned on buildings-banepa-classification",
    )

    client.predict(local_model_id, image_path="data/sample/predict/oam")
