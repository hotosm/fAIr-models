from fair.client import FairClient

client = FairClient(
    zenml_store_url=None,
    stac_api_url=None,
    dsn=None,
    user_id="anonymous",
    config_dir="examples/segmentation/config",
    upload_artifacts=False,
)

if __name__ == "__main__":
    from fair.utils import install_s3_cleanup_handler

    install_s3_cleanup_handler()

    client.setup()

    base_model_id = client.register_base_model("models/unet_segmentation/stac-item.json")

    dataset_id = client.register_dataset("data/sample/buildings-banepa/stac-item.json")

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
