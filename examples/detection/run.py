from fair.client import DatasetConfig, FairClient

client = FairClient(
    zenml_store_url=None,
    stac_api_url=None,
    dsn=None,
    user_id="anonymous",
    config_dir="examples/detection/config",
)

if __name__ == "__main__":
    from fair.utils import install_s3_cleanup_handler

    install_s3_cleanup_handler()

    client.setup()

    base_model_id = client.register_base_model("models/yolo11n_detection/stac-item.json")

    dataset_id = client.register_dataset(
        DatasetConfig(
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
        )
    )

    finetuned_model_id = client.finetune(
        base_model_id=base_model_id,
        dataset_id=dataset_id,
        model_name="yolo11n-detection-finetuned-banepa",
        overrides={"learning_rate": 0.01, "batch_size": 8, "chip_size": 640},
    )

    local_model_id = client.promote(
        finetuned_model_id,
        description="YOLOv11n detection finetuned on buildings-banepa-detection",
    )

    client.predict(local_model_id, image_path="data/sample/predict/oam")
