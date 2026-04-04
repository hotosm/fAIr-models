from fair.zenml.runner import DatasetConfig, FairWorkflowRunner

runner = FairWorkflowRunner(
    base_model_id="unet-segmentation",
    model_name="unet-segmentation-finetuned-banepa",
    stac_item_path="models/unet_segmentation/stac-item.json",
    pipeline_module="models.unet_segmentation.pipeline",
    config_dir="examples/segmentation/config",
    dataset=DatasetConfig(
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
    ),
    finetune_overrides={"learning_rate": 0.001},
    promote_description="UNet segmentation finetuned on buildings-banepa",
)

if __name__ == "__main__":
    runner.run()
