from fair.client import DatasetConfig, FairClient

client = FairClient(
    zenml_store_url=None,
    stac_api_url=None,
    dsn=None,
    user_id="anonymous",
    config_dir="examples/segmentation/config",
)

if __name__ == "__main__":
    from fair.utils import install_s3_cleanup_handler

    install_s3_cleanup_handler()

    client.setup()

    base_model_id = client.register_base_model("models/unet_segmentation/stac-item.json")

    dataset_id = client.register_dataset(
        DatasetConfig(
            title="buildings-banepa",
            description="OpenAerialMap chips with OSM building footprints for Banepa, Nepal.",
            label_type="vector",
            label_tasks=["segmentation"],
            label_classes=[{"name": "building", "classes": ["building"]}],
            keywords=["building", "semantic-segmentation", "polygon"],
            train_chips_path="data/sample/train/oam",
            train_labels_path="data/sample/train/osm",
            predict_images_path="data/sample/predict/oam",
            labels_pattern="*.geojson",
            source_imagery_href=(
                "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
            ),
            label_description="Building footprints manually labeled from OpenAerialMap imagery",
        )
    )

    finetuned_model_id = client.finetune(
        base_model_id=base_model_id,
        dataset_id=dataset_id,
        model_name="unet-segmentation-finetuned-banepa",
        overrides={"learning_rate": 0.001},
    )

    local_model_id = client.promote(
        finetuned_model_id,
        description="UNet segmentation finetuned on buildings-banepa",
    )

    client.predict(local_model_id, image_path="data/sample/predict/oam")
