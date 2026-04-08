import os

from fair.client import FairClient

client = FairClient(
    zenml_store_url=os.environ.get("FAIR_ZENML_STORE_URL"),
    stac_api_url=os.environ.get("FAIR_STAC_API_URL"),
    dsn=os.environ.get("FAIR_DSN"),
    user_id=os.environ.get("FAIR_USER_ID", "anonymous"),
    config_dir="examples/detection/config",
    upload_artifacts=os.environ.get("FAIR_UPLOAD_ARTIFACTS", "").lower() == "true",
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
        overrides={"learning_rate": 0.01, "batch_size": 2, "chip_size": 640},
    )

    local_model_id = client.promote(
        finetuned_model_id,
        description="YOLOv11n detection finetuned on buildings-banepa-detection",
    )

    client.predict(local_model_id, image_path="data/sample/predict/oam")
