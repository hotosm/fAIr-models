from fair.client import FairClient

client = FairClient(
    zenml_store_url=None,
    stac_api_url=None,
    dsn=None,
    user_id="anonymous",
    config_dir="examples/detection/config",
    upload_artifacts=False,
)

if __name__ == "__main__":
    from fair.utils import install_s3_cleanup_handler

    install_s3_cleanup_handler()

    client.setup()

    base_model_id = client.register_base_model("models/yolo11n_detection/stac-item.json")

    dataset_id = client.register_dataset("data/sample/buildings-banepa-detection/stac-item.json")

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
