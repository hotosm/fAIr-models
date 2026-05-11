class DatasetStoragePaths:
    ROOT = "datasets"
    DOWNLOAD_SUBDIR = "download"

    @classmethod
    def download_dir(cls, prefix: str, item_id: str) -> str:
        return f"{prefix}/{cls.ROOT}/{item_id}/{cls.DOWNLOAD_SUBDIR}"


class LocalModelStoragePaths:
    ROOT = "local-models"
    CHECKPOINT_SUBDIR = "checkpoint"
    MODEL_SUBDIR = "model"
    METRICS_SUBDIR = "training-metrics"

    @classmethod
    def item_dir(cls, prefix: str, item_id: str) -> str:
        return f"{prefix}/{cls.ROOT}/{item_id}"

    @classmethod
    def checkpoint_dir(cls, prefix: str, item_id: str) -> str:
        return f"{cls.item_dir(prefix, item_id)}/{cls.CHECKPOINT_SUBDIR}"

    @classmethod
    def model_dir(cls, prefix: str, item_id: str) -> str:
        return f"{cls.item_dir(prefix, item_id)}/{cls.MODEL_SUBDIR}"

    @classmethod
    def metrics_file(cls, prefix: str, item_id: str) -> str:
        return f"{cls.item_dir(prefix, item_id)}/{cls.METRICS_SUBDIR}/{item_id}.json"
